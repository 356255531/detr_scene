# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import numpy as np

import torch
from util import box_ops

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

from datasets.sg_eval import SGRecall,SGMeanRecall
from datasets.do_sgg_eval import do_sgg_eval
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def sgg_evaluate(model, criterion,postprocessors,
                    data_loader,device):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    print_freq = 10

    evaluator = {}
    result_dict = {}
    eval_recall = SGRecall(result_dict)
    mode = 'sgdet'
    eval_recall.register_container(mode)
    evaluator['eval_recall'] = eval_recall
    num_rel_category = 50
    ind_to_predicates = np.arange(50)
    eval_mean_recall = SGMeanRecall(result_dict, num_rel_category, ind_to_predicates, print_detail=True)
    eval_mean_recall.register_container(mode)
    evaluator['eval_mean_recall'] = eval_mean_recall
    result_str = '\n' + '=' * 100 + '\n'

    fauxcoco = COCO()
    predictions = []
    groundtruths = []

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()


        results = postprocessors['bbox'](outputs)

        for i, (target, result) in enumerate(zip(targets,results)):
            predictions.append(result)
            groundtruths.append(target)
            do_sgg_eval(target,result,device,mode,evaluator,result_dict)


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
    # gather the stats from all processes
    anns = []
    for image_id, gt in enumerate(groundtruths):
        labels = gt['labels'].tolist()  # integer
        boxes = box_ops.box_cxcywh_to_xyxy(gt['boxes'] * 1000).tolist()  # xyxy
        for cls, box in zip(labels, boxes):
            anns.append({
                'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1],  # xywh
                'category_id': cls,
                'id': len(anns),
                'image_id': image_id,
                'iscrowd': 0,
            })

    fauxcoco.dataset = {
        'info': {'description': 'use coco script for vg detection evaluation'},
        'images': [{'id': i} for i in range(len(groundtruths))],
        'categories': [
            {'supercategory': 'person', 'id': i, 'name': i}
            for i in range(150)
        ],
        'annotations': anns,
    }
    fauxcoco.createIndex()

    # format predictions to coco-like
    cocolike_predictions = []

    for image_id, prediction in enumerate(predictions):
        box = box_ops.box_cxcywh_to_xywh(prediction['boxes'] * 1000).detach().cpu().numpy()  # xywh
        score = prediction['classification scores'].detach().cpu().numpy()  # (#objs,)
        label = prediction['classification labels'].detach().cpu().numpy()  # (#objs,)
        # for predcls, we set label and score to groundtruth

        image_id = np.asarray([image_id] * len(box))
        cocolike_predictions.append(
            np.column_stack((image_id, box, score, label))
        )
        # logger.info(cocolike_predictions)
    cocolike_predictions = np.concatenate(cocolike_predictions, 0)
    # evaluate via coco API
    res = fauxcoco.loadRes(cocolike_predictions)
    coco_eval = COCOeval(fauxcoco, res, 'bbox')
    coco_eval.params.imgIds = list(range(len(groundtruths)))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mAp = coco_eval.stats[1]

    result_str += 'Detection evaluation mAp=%.4f\n' % mAp
    result_str += '=' * 100 + '\n'
    result_str += eval_recall.generate_print_string(mode)
    eval_mean_recall.calculate_mean_recall(mode)
    result_str += eval_mean_recall.generate_print_string(mode)
    print(result_str)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, result_str
