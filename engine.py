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
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)
        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator

@torch.no_grad()
def sgg_evaluate(model: torch.nn.Module, criterion: torch.nn.Module,postprocessors,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    ii = 0

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

    OBJECT_NAME_DICT = {'man': 79697, 'window': 54583, 'person': 52876, 'woman': 35742, 'building': 35231,
                        'shirt': 32121, 'wall': 31638, 'tree': 29918, 'sign': 24021, 'head': 23832, 'ground': 23043,
                        'table': 23035, 'hand': 21122, 'grass': 20272, 'sky': 19985, 'water': 18989, 'pole': 18665,
                        'light': 17443, 'leg': 17269, 'car': 17205, 'people': 15635, 'hair': 15543, 'clouds': 14590,
                        'ear': 14526, 'plate': 13797, 'street': 13470, 'trees': 13083, 'road': 12860,
                        'shadow': 12611, 'eye': 12524, 'leaves': 12079, 'snow': 11919, 'train': 11692, 'hat': 11680,
                        'door': 11552, 'boy': 11104, 'pants': 10953, 'wheel': 10730, 'nose': 10629, 'fence': 10334,
                        'sidewalk': 10233, 'girl': 9826, 'jacket': 9813, 'field': 9698, 'floor': 9549, 'tail': 9532,
                        'chair': 9308, 'clock': 9144, 'handle': 9083, 'face': 8846, 'boat': 8794, 'line': 8777,
                        'arm': 8743, 'plane': 8285, 'horse': 8156, 'bus': 8136, 'dog': 8100, 'windows': 7995,
                        'giraffe': 7950, 'bird': 7892, 'cloud': 7880, 'elephant': 7822, 'helmet': 7748,
                        'shorts': 7587, 'food': 7277, 'leaf': 7210, 'shoe': 7155, 'zebra': 7031, 'glass': 7021,
                        'cat': 6990, 'bench': 6757, 'glasses': 6723, 'bag': 6713, 'flower': 6615,
                        'background': 6539, 'rock': 6213, 'cow': 6190, 'foot': 6165, 'sheep': 6161, 'letter': 6140,
                        'picture': 6126, 'logo': 6116, 'player': 6065, 'bottle': 6020, 'tire': 6017,
                        'skateboard': 6017, 'stripe': 6001, 'umbrella': 5979, 'surfboard': 5954, 'shelf': 5944,
                        'bike': 5868, 'number': 5828, 'part': 5820, 'motorcycle': 5818, 'tracks': 5801,
                        'mirror': 5747, 'truck': 5610, 'tile': 5602, 'mouth': 5584, 'bowl': 5522, 'pizza': 5521,
                        'bear': 5389, 'spot': 5328, 'kite': 5307, 'bed': 5295, 'roof': 5256, 'counter': 5252,
                        'post': 5230, 'dirt': 5204, 'beach': 5102, 'flowers': 5101, 'jeans': 5018, 'top': 5016,
                        'legs': 4975, 'cap': 4860, 'pillow': 4775, 'box': 4748, 'neck': 4697, 'house': 4629,
                        'reflection': 4612, 'lights': 4554, 'plant': 4515, 'trunk': 4465, 'sand': 4451, 'cup': 4416,
                        'child': 4368, 'button': 4334, 'wing': 4325, 'shoes': 4323, 'writing': 4284, 'sink': 4204,
                        'desk': 4176, 'board': 4168, 'wave': 4147, 'sunglasses': 4129, 'edge': 4119, 'paper': 3994,
                        'vase': 3983, 'lamp': 3950, 'lines': 3936, 'brick': 3907, 'phone': 3888, 'ceiling': 3860,
                        'book': 3785, 'airplane': 3695, 'laptop': 3691, 'vehicle': 3686, 'headlight': 3678,
                        'coat': 3639,'a':222}
    PREDICATE_DICT = {'on': 284572, 'have': 124330, 'in': 89147, 'wearing': 80107, 'of': 72283, 'with': 20517,
                      'behind': 18353, 'holding': 12781, 'standing': 12307, 'near': 11739, 'sitting': 11569,
                      'next': 10376, 'walking': 6991, 'riding': 6813, 'are': 6718, 'by': 6517, 'under': 6319,
                      'in front of': 5539, 'on side of': 5370, 'above': 5224, 'hanging': 4719, 'at': 3536,
                      'parked': 3308, 'beside': 3210, 'flying': 3032, 'attached to': 2925, 'eating': 2727,
                      'looking': 2407, 'carrying': 2389, 'laying': 2333, 'over': 2252, 'inside': 2104,
                      'belonging': 1976, 'covered': 1891, 'growing': 1678, 'covering': 1642, 'driving': 1536,
                      'lying': 1456, 'around': 1454, 'below': 1408, 'painted': 1386, 'against': 1381, 'along': 1353,
                      'for': 1272, 'crossing': 1134, 'mounted': 1083, 'playing': 1053, 'outside': 1012,
                      'watching': 992}

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        from PIL import Image
        import matplotlib.pyplot as plt
        from datasets.coco import make_coco_transforms
        import datasets.transforms as T
        from datasets.do_sgg_eval import non_max_suppression
        image = Image.open('visualization/1.jpg')
        (x, y) = image.size
        # image_array = np.array(image)
        # im_output = Image.fromarray(image_array)
        # plt.imshow(im_output)
        # plt.show()
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        ttt = {}
        im = transform(image,ttt)
        ima = []
        ima.append(im[0])
        outputs = model(ima)
        results = postprocessors['bbox'](outputs)
        pre_class_mask = results[0]['classification scores'].gt(0.6)
        pre_class = results[0]['classification labels'][pre_class_mask]
        pre_class_idx = torch.arange(0, 100)[pre_class_mask]

        pre_class_idx_pick = non_max_suppression(pre_class_idx, pre_class, results[0]['boxes'][pre_class_idx],
                                                 results[0]['classification scores'][pre_class_idx], threshold=0.4)
        pre_class_idx = pre_class_idx[pre_class_idx_pick]
        pre_class = results[0]['classification labels'][pre_class_idx]

        # get rel_idx
        preboxes = results[0]['boxes'][pre_class_idx]
        box = box_ops.box_cxcywh_to_xywh(preboxes)

        image_array = np.array(image)
        im_output = Image.fromarray(image_array)
        plt.imshow(im_output)

        ax = plt.gca()

        li = list(OBJECT_NAME_DICT.keys())

        for n in range(len(box)):
            ax.add_patch(plt.Rectangle((box[n][0] * x, box[n][1] * y),
                                       box[n][2] * x, box[n][3] * y, fill=False, color='red', linewidth=3))
            text = f'{li[pre_class[n]]}'
            ax.text(box[n][0]*x,box[n][1]*y,text,fontsize=15,bbox=dict(facecolor='yellow',alpha=0.5))
        plt.axis('off')
        plt.savefig('save.jpg')
        plt.show()

        import pdb
        pdb.set_trace()



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
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        ii += 1
        # if ii == 5:
        #     break
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
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
