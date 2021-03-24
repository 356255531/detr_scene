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
    recall = torch.zeros(3)
    mean_recall20_dict = dict()
    mean_recall50_dict = dict()
    mean_recall100_dict = dict()
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

        #calculate recall

        #get gt_traid
        sub_idx = targets[0]['relationships'][:,0]
        ob_idx = targets[0]['relationships'][:,1]
        gt_sub = targets[0]['labels'][sub_idx].reshape(-1,1)
        gt_ob = targets[0]['labels'][ob_idx].reshape(-1,1)
        gt_sub_box = targets[0]['boxes'][sub_idx]
        gt_ob_box = targets[0]['boxes'][ob_idx]
        gt_predicate = targets[0]['predicate_labels'].reshape(-1,1)
        gt_traid = torch.cat((gt_sub,gt_predicate,gt_ob),1)


        ## get pre_traid

        results = postprocessors['bbox'](outputs)
        pre_class_mask = results[0]['classification scores'].gt(0.6)
        pre_class = results[0]['classification labels'][pre_class_mask]
        pre_class_idx = torch.arange(0, 100)[pre_class_mask]
        pre_sub_idx = pre_class_idx.repeat(len(pre_class_idx),1).transpose(0,1)
        pre_ob_idx = pre_class_idx.repeat(len(pre_class_idx), 1)


        pre_class_idx_pick = non_max_suppression(pre_class_idx,pre_class,results[0]['boxes'][pre_class_idx], results[0]['classification scores'][pre_class_idx], threshold=0.9)
        pre_class_idx = pre_class_idx[pre_class_idx_pick]
        # get rel_idx
        preboxes = results[0]['boxes'][pre_class_idx]
        rel_idx = prepare_test_pairs(device, preboxes)



        # pre_sub_idx = pre_class_idx.repeat(len(pre_class_idx),1).transpose(0,1)
        # pre_ob_idx = pre_class_idx.repeat(len(pre_class_idx), 1)
        #
        #
        # des = pre_sub_idx - pre_ob_idx
        # mask = des.ne(0)
        #
        # ##Delete duplicates
        # pre_sub_idx = pre_sub_idx[mask]
        # pre_ob_idx = pre_ob_idx[mask]
        pre_sub_idx = pre_class_idx[rel_idx[:,0]]
        pre_ob_idx = pre_class_idx[rel_idx[:,1]]
        pre_predicate_idx = pre_class_idx[rel_idx[:,0]] * 100 + pre_class_idx[rel_idx[:,1]]
        pre_predicate = results[0]['predicate labels'][pre_predicate_idx]
        pre_predicate_scores = results[0]['predicate scores'][pre_predicate_idx].reshape(-1,1)


        pre_sub_scores = results[0]['classification scores'][pre_sub_idx].reshape(-1,1)
        pre_ob_scores = results[0]['classification scores'][pre_ob_idx].reshape(-1,1)
        totalscores = torch.cat((pre_sub_scores,pre_ob_scores,pre_predicate_scores),1)

        pre_predicate_scores, sort_idx = torch.sort(torch.prod(totalscores,1),descending=True)
        pre_predicate = pre_predicate[sort_idx].reshape(-1,1)
        pre_sub_idx = pre_sub_idx[sort_idx].reshape(-1,1)
        pre_ob_idx = pre_ob_idx[sort_idx].reshape(-1,1)
        pre_sub = results[0]['classification labels'][pre_sub_idx]
        pre_ob = results[0]['classification labels'][pre_ob_idx]
        pre_sub_box = results[0]['boxes'][pre_sub_idx].reshape(-1,4)
        pre_ob_box = results[0]['boxes'][pre_ob_idx].reshape(-1,4)

        # pre_traid
        pre_traid = torch.cat((pre_sub,pre_predicate,pre_ob),1)



        # # creat ground truth dict
        # gt_dict = dict()
        # for i, num in enumerate(gt_traid):
        #     key = num[0].item()*1e6 + num[1].item()*1e3 + num[2].item()
        #     if key in gt_dict:
        #         gt_dict[key] = torch.cat((gt_dict[key],torch.cat((gt_sub_box[i],gt_ob_box[i]))))
        #     else:
        #         gt_dict[key] = torch.cat((gt_sub_box[i],gt_ob_box[i]))


        # ##calculate recall 20
        #
        # if len(pre_traid) < 100:
        #     suppl = torch.ones(100-len(pre_traid),3) * 100
        #     suppl_box = torch.ones(100 - len(pre_traid), 4) * 2
        #     pre_traid = torch.cat((pre_traid,suppl),0)
        #     pre_sub_box = torch.cat((pre_sub_box,suppl_box),0)
        #     pre_ob_box = torch.cat((pre_ob_box, suppl_box), 0)
        #
        #
        # # creat pre_traid dict
        # pre_dict = dict()
        # for i, num in enumerate(pre_traid):
        #     key = num[0].item()*1e6 + num[1].item()*1e3 + num[2].item()
        #     if key in pre_dict:
        #         pre_dict[key] = torch.cat((pre_dict[key],torch.cat((pre_sub_box[i],pre_ob_box[i]))))
        #     else:
        #         pre_dict[key] = torch.cat((pre_sub_box[i],pre_ob_box[i]))
        # score = 0
        # for i in range(len(gt_traid)):
        #     key = gt_traid[i][0].item() * 1e6 + gt_traid[i][1].item() * 1e3 + gt_traid[i][2].item()
        #     if key in pre_dict:
        #         for j in range(len(pre_dict[key]) // 8):
        #             src_boxes = pre_dict[key][j*8:j*8+8]
        #             iou1 = torch.diag(box_ops.sgg_box_iou(
        #                 box_ops.box_cxcywh_to_xyxy(src_boxes[:4]).reshape(-1, 4),
        #                 box_ops.box_cxcywh_to_xyxy(gt_sub_box[i]).reshape(-1, 4)))
        #             iou2 = torch.diag(box_ops.sgg_box_iou(
        #                 box_ops.box_cxcywh_to_xyxy(src_boxes[4:]).reshape(-1, 4),
        #                 box_ops.box_cxcywh_to_xyxy(gt_ob_box[i]).reshape(-1, 4)))
        #             if iou1 >= 0.5 and iou2 >= 0.5:
        #                 score += 1
        #                 break
        #
        # recall_100 = score / len(gt_traid)


        k = 20
        recall[0] = recall[0] + calculate_recall_k(k,pre_traid, pre_sub_box, pre_ob_box, gt_traid, gt_sub_box, gt_ob_box)
        calculate_meanrecall_k(k, mean_recall20_dict, pre_traid, pre_sub_box, pre_ob_box, gt_traid, gt_sub_box,
                               gt_ob_box)
        k = 50
        recall[1] = recall[1] + calculate_recall_k(k,pre_traid, pre_sub_box, pre_ob_box, gt_traid, gt_sub_box, gt_ob_box)
        calculate_meanrecall_k(k, mean_recall50_dict, pre_traid, pre_sub_box, pre_ob_box, gt_traid, gt_sub_box,
                               gt_ob_box)
        k = 100
        recall[2] = recall[2] + calculate_recall_k(k,pre_traid,pre_sub_box,pre_ob_box,gt_traid,gt_sub_box,gt_ob_box)
        calculate_meanrecall_k(k, mean_recall100_dict, pre_traid, pre_sub_box, pre_ob_box, gt_traid, gt_sub_box,
                               gt_ob_box)






        # score = 0
        # for i in range(200):
        #     key = pre_traid[i][0].item() * 1e6 + pre_traid[i][1].item() * 1e3 + pre_traid[i][2].item()
        #     if key in gt_dict:
        #         for j in range(len(gt_dict[key]) // 8):
        #             src_boxes = gt_dict[key][j*8:j*8+8]
        #             iou1 = torch.diag(box_ops.sgg_box_iou(
        #                 box_ops.box_cxcywh_to_xyxy(src_boxes[:4]).reshape(-1, 4),
        #                 box_ops.box_cxcywh_to_xyxy(pre_sub_box[i]).reshape(-1, 4)))
        #             iou2 = torch.diag(box_ops.sgg_box_iou(
        #                 box_ops.box_cxcywh_to_xyxy(src_boxes[4:]).reshape(-1, 4),
        #                 box_ops.box_cxcywh_to_xyxy(pre_ob_box[i]).reshape(-1, 4)))
        #             if iou1 >= 0.5 and iou2 >= 0.5:
        #                 score += 1
        #                 break
        #
        # recall_100 = score / len(gt_traid)
        #
        # print(recall_100,len(gt_traid))



        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # optimizer.zero_grad()
        # losses.backward()
        # if max_norm > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # optimizer.step()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        ii += 1
        mean_recall = final_mean_recall(mean_recall20_dict, mean_recall50_dict, mean_recall100_dict)

        print("mean recall_20:", mean_recall[0], "mean recall_50:", mean_recall[1], "mean recall_100:", mean_recall[2])
        print("recall_20:", recall[0] / ii, "recall_50:", recall[1] / ii, "recall_100:", recall[2] / ii)

        # if ii == 20:
        #     break
    # gather the stats from all processes

    mean_recall = final_mean_recall(mean_recall20_dict,mean_recall50_dict,mean_recall100_dict)

    print("mean recall_20:", mean_recall[0], "mean recall_50:", mean_recall[1], "mean recall_100:", mean_recall[2])
    print("recall_20:", recall[0] / ii, "recall_50:", recall[1] / ii, "recall_100:", recall[2] / ii)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def getk_pre_traid(k,pre_traid,pre_sub_box,pre_ob_box):
    if len(pre_traid) < k:
        suppl = torch.ones(k - len(pre_traid), 3) * 100
        suppl_box = torch.ones(k - len(pre_traid), 4) * 2
        pre_traid = torch.cat((pre_traid, suppl), 0)
        pre_sub_box = torch.cat((pre_sub_box, suppl_box), 0)
        pre_ob_box = torch.cat((pre_ob_box, suppl_box), 0)
    else:
        pre_traid = pre_traid[:k]
        pre_sub_box = pre_sub_box[:k]
        pre_ob_box = pre_ob_box[:k]
    return pre_traid,pre_sub_box,pre_ob_box


def calculate_recall_k(k,pre_traid,pre_sub_box,pre_ob_box,gt_traid,gt_sub_box,gt_ob_box):

    # creat pre_traid dict
    pre_traid, pre_sub_box, pre_ob_box = getk_pre_traid(k, pre_traid, pre_sub_box, pre_ob_box)
    pre_dict = dict()
    for i, num in enumerate(pre_traid):
        key = num[0].item() * 1e6 + num[1].item() * 1e3 + num[2].item()
        if key in pre_dict:
            pre_dict[key] = torch.cat((pre_dict[key], torch.cat((pre_sub_box[i], pre_ob_box[i]))))
        else:
            pre_dict[key] = torch.cat((pre_sub_box[i], pre_ob_box[i]))
    score = 0
    for i in range(len(gt_traid)):
        key = gt_traid[i][0].item() * 1e6 + gt_traid[i][1].item() * 1e3 + gt_traid[i][2].item()
        if key in pre_dict:
            for j in range(len(pre_dict[key]) // 8):
                src_boxes = pre_dict[key][j * 8:j * 8 + 8]
                iou1 = torch.diag(box_ops.sgg_box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_boxes[:4]).reshape(-1, 4),
                    box_ops.box_cxcywh_to_xyxy(gt_sub_box[i]).reshape(-1, 4)))
                iou2 = torch.diag(box_ops.sgg_box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_boxes[4:]).reshape(-1, 4),
                    box_ops.box_cxcywh_to_xyxy(gt_ob_box[i]).reshape(-1, 4)))
                if iou1 >= 0.5 and iou2 >= 0.5:
                    score += 1
                    break

    recall__k = score / len(gt_traid)
    return recall__k

def calculate_meanrecall_k(k,mean_recall_dict,pre_traid,pre_sub_box,pre_ob_box,gt_traid,gt_sub_box,gt_ob_box):
    # creat pre_traid dict
    pre_traid, pre_sub_box, pre_ob_box = getk_pre_traid(k, pre_traid, pre_sub_box, pre_ob_box)
    result = 0
    recall = torch.zeros(50)
    for i in range(50):
        mask = gt_traid[:,1].eq(i).repeat(3,1).transpose(0,1)
        mask_box = gt_traid[:, 1].eq(i).repeat(4, 1).transpose(0, 1)
        gt_traid_i = gt_traid[mask].reshape(-1,3)
        if len(gt_traid_i) != 0:
            gt_sub_box_i = gt_sub_box[mask_box].reshape(-1, 4)
            gt_ob_box_i = gt_ob_box[mask_box].reshape(-1, 4)
            recall[i] = calculate_recall_k(len(pre_traid),pre_traid, pre_sub_box, pre_ob_box, gt_traid_i, gt_sub_box_i, gt_ob_box_i)
            result += recall[i]
            if i in mean_recall_dict:
                mean_recall_dict[i] = torch.cat((mean_recall_dict[i], recall[i].reshape(1, -1)))
            else:
                mean_recall_dict[i] = recall[i].reshape(1, -1)
    return


def final_mean_recall(mean_recall20_dict,mean_recall50_dict,mean_recall100_dict):
    results = torch.zeros(3,50)
    for key in mean_recall100_dict:
        results[0, key] = torch.mean(mean_recall20_dict[key])
        results[1, key] = torch.mean(mean_recall50_dict[key])
        results[2, key] = torch.mean(mean_recall100_dict[key])

    return torch.mean(results,1)


def non_max_suppression(pre_class_idx,pre_class,boxes, scores, threshold):
    """执行non-maximum suppression并返回保留的boxes的索引.
    boxes: [N, (y1, x1, y2, x2)].注意(y2, x2)可以会超过box的边界.
    scores: box的分数的一维数组.
    threshold: Float型. 用于过滤IoU的阈值.
    """
    if boxes.shape[0] == 0:
        return pre_class_idx
    # if boxes.dtype.kind != "f":
    #     boxes = boxes.astype(np.float32)
    # 获取根据分数排序的boxes的索引(最高的排在对前面)
    ixs = scores.argsort()
    pick = []
    while len(ixs) > 0:
        # 选择排在最前的box，并将其索引加到列表中
        i = ixs[0]
        pick.append(i)
        # 计算选择的box与剩下的box的IoU
        iou = torch.diag(box_ops.sgg_box_iou(
                    box_ops.box_cxcywh_to_xyxy(boxes[i].repeat(len(ixs)-1,1)),
                    box_ops.box_cxcywh_to_xyxy(boxes[ixs[1:]])))
        # 确定IoU大于阈值的boxes. 这里返回的是ix[1:]之后的索引，
        # 所以为了与ixs保持一致，将结果加1
        remove_ixs = torch.LongTensor(np.where(iou > threshold)[0] + 1)
        mask = remove_ixs.lt(0)

        for j,num in enumerate(remove_ixs):
            if pre_class[ixs[num]] == pre_class[i]:
                mask[j] = True
        remove_ixs = remove_ixs[mask]
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return torch.LongTensor(pick)


def prepare_test_pairs(device, proposals):
    # prepare object pairs for relation prediction
    rel_pair_idxs = []
    n = len(proposals)
    cand_matrix = torch.ones((n, n), device=device) - torch.eye(n, device=device)
    # mode==sgdet and require_overlap
    p_to_xyxy = box_ops.box_cxcywh_to_xyxy(proposals)
    cand_matrix = cand_matrix.byte() & box_ops.sgg_box_iou(p_to_xyxy,p_to_xyxy).gt(0).byte()
    idxs = torch.nonzero(cand_matrix).view(-1,2)
    if len(idxs) > 0:
        rel_pair_idxs.append(idxs)
    else:
        # if there is no candidate pairs, give a placeholder of [[0, 0]]
        idxs = (torch.zeros((1, 2), dtype=torch.int64, device=device))
    return idxs

