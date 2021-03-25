
import numpy as np

import torch
from util import box_ops




def do_sgg_eval(target,result,device,mode,evaluator,result_dict):


    pre_class_mask = result['classification scores'].gt(0.6)
    pre_class = result['classification labels'][pre_class_mask]
    pre_class_idx = torch.arange(0, 100)[pre_class_mask]


    pre_class_idx_pick = non_max_suppression(pre_class_idx, pre_class, result['boxes'][pre_class_idx],
                                             result['classification scores'][pre_class_idx], threshold=0.85)
    pre_class_idx = pre_class_idx[pre_class_idx_pick]
    pre_class = result['classification labels'][pre_class_idx]

    # get rel_idx
    preboxes = result['boxes'][pre_class_idx]
    rel_idx = prepare_test_pairs(device, preboxes)

    pre_sub_idx = pre_class_idx[rel_idx[:, 0]]
    pre_ob_idx = pre_class_idx[rel_idx[:, 1]]
    pre_predicate_idx = pre_sub_idx * 100 + pre_ob_idx
    pre_predicate = result['predicate labels'][pre_predicate_idx]
    pre_predicate_scores = result['predicate scores'][pre_predicate_idx].reshape(-1, 1)


    iou_thres = 0.5
    global_container = {}
    global_container['result_dict'] = result_dict
    global_container['mode'] = mode
    global_container['iou_thres'] = iou_thres

    prediction = {}
    prediction['rel_pair_idxs'] = rel_idx
    prediction['rel_predicated'] = pre_predicate
    prediction['pred_rel_scores'] = pre_predicate_scores
    prediction['boxes'] = box_ops.box_cxcywh_to_xyxy(preboxes)
    prediction['pred_labels'] = pre_class
    prediction['pred_scores'] = result['classification scores'][pre_class_idx]

    local_container = evaluate_relation_of_one_image(target, prediction, global_container, evaluator)

    return

def non_max_suppression(pre_class_idx, pre_class, boxes, scores, threshold):
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
            box_ops.box_cxcywh_to_xyxy(boxes[i].repeat(len(ixs) - 1, 1)),
            box_ops.box_cxcywh_to_xyxy(boxes[ixs[1:]])))
        # 确定IoU大于阈值的boxes. 这里返回的是ix[1:]之后的索引，
        # 所以为了与ixs保持一致，将结果加1
        remove_ixs = torch.LongTensor(np.where(iou > threshold)[0] + 1)
        mask = remove_ixs.lt(0)

        for j, num in enumerate(remove_ixs):
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
    cand_mat = cand_matrix.byte() & box_ops.sgg_box_iou(p_to_xyxy, p_to_xyxy).gt(0).byte()
    idxs = torch.nonzero(cand_mat).view(-1, 2)
    if len(idxs) > 0:
        rel_pair_idxs.append(idxs)
    else:
        # if there is no candidate pairs, give a placeholder of [[0, 0]]
        idxs = torch.nonzero(cand_matrix).view(-1, 2)
    return idxs

def evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator):
    """
    Returns:
        pred_to_gt: Matching from predicate to GT
        pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
        pred_triplet_scores: [cls_0score, relscore, cls1_score]
    """
    # unpack all inputs
    mode = global_container['mode']

    local_container = {}
    local_container['gt_rels'] = groundtruth['relationships'].long().detach().cpu().numpy()
    local_container['gt_rels_predicate'] = groundtruth['predicate_labels'].long().detach().cpu().numpy()

    # if there is no gt relations for current image, then skip it
    if len(local_container['gt_rels']) == 0:
        return

    local_container['gt_boxes'] = box_ops.box_cxcywh_to_xyxy(
        groundtruth['boxes']).detach().cpu().numpy()  # (#gt_objs, 4)
    local_container['gt_classes'] = groundtruth['labels'].long().detach().cpu().numpy()  # (#gt_objs, )

    # about relations
    local_container['pred_rel_inds'] = prediction[
        'rel_pair_idxs'].long().detach().cpu().numpy()  # (#pred_rels, 2)
    local_container['rel_scores'] = prediction[
        'pred_rel_scores'].detach().cpu().numpy()  # (#pred_rels, num_pred_class)
    local_container['rel_predicated'] = prediction[
        'rel_predicated'].long().detach().cpu().numpy()

    # about objects
    local_container['pred_boxes'] = prediction['boxes'].detach().cpu().numpy()  # (#pred_objs, 4)
    local_container['pred_classes'] = prediction[
        'pred_labels'].long().detach().cpu().numpy()  # (#pred_objs, )
    local_container['obj_scores'] = prediction['pred_scores'].detach().cpu().numpy()  # (#pred_objs, )

    # to calculate accuracy, only consider those gt pairs
    if mode == 'predcls':
        local_container['pred_boxes'] = local_container['gt_boxes']
        local_container['pred_classes'] = local_container['gt_classes']
        local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])

    elif mode == 'sgcls':
        if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
            print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
    elif mode == 'sgdet' or mode == 'phrdet':
        pass
    else:
        raise ValueError('invalid mode')

    if local_container['pred_rel_inds'].shape[0] == 0:
        return

    # Traditional Metric with Graph Constraint
    # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
    local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container, mode)
    evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    return local_container
