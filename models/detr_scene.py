# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


class DETRScene(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_predicate_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.predicate_emb = nn.Linear(2 * hidden_dim, num_predicate_classes)
        self.query_embed = nn.Embedding(num_queries + 1, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

    def forward(self, scene=False, samples: NestedTensor=None, hs=None, indices=None, relationships=None):
        if not scene:
            if isinstance(samples, (list, torch.Tensor)):
                samples = nested_tensor_from_tensor_list(samples)
            features, pos = self.backbone(samples)

            src, mask = features[-1].decompose()
            assert mask is not None
            hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

            outputs_class = self.class_embed(hs)
            outputs_coord = self.bbox_embed(hs).sigmoid()

            out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
            ret = out, hs
        else:
            hs_size = hs.size()  # [#decoder_layer x batch_size x #bbox x #hidden]
            num_decoder_layer = hs_size[0]  # 6
            batch_size = hs_size[1]  # 2
            num_queries = hs_size[2]  # 101
            hidden_dim = hs_size[3]  # 256

            # Align indices
            src_indices = [[single_idx[0] for single_idx in idx] for idx in indices]
            lengths = torch.tensor([src_idx.shape[0] for src_idx in src_indices[0]])
            max_match_length = torch.max(lengths)
            complementary_lengths = max_match_length - lengths
            complementary_tensor = [torch.full((length,), num_queries - 1, dtype=torch.int64, device=hs.device) for
                                    length in complementary_lengths]
            complementary_tensor = [complementary_tensor for _ in range(len(indices))]
            src_indices = torch.cat(  # [#decoder_layer x batch_size x #bbox]
                [
                    torch.cat(
                        [torch.cat([i.to(hs.device), j]).unsqueeze(0) for i, j in zip(src_idx, ct)]
                    ).unsqueeze(0) for src_idx, ct in zip(src_indices, complementary_tensor)
                ]
            )

            # reindex hs
            flatten_hs = hs.reshape(-1, hidden_dim)
            flatten_src_indices = src_indices.reshape(-1, max_match_length)
            delta_indices = torch.arange(0, num_decoder_layer * batch_size, device=hs.device).reshape(-1, 1)
            delta_indices = delta_indices.repeat([1, max_match_length]) * (self.num_queries + 1)
            aligned_flatten_hs = flatten_hs[(flatten_src_indices + delta_indices).reshape(
                -1)]  # [(max_box_per_image * batch_size * #decode_layer) x hidden_dim]

            # reindex relationships
            tgt_indices = [[single_idx[1] for single_idx in idx] for idx in indices]
            lengths = torch.tensor([tgt_idx.shape[0] for tgt_idx in tgt_indices[0]])
            max_rel_length = torch.max(lengths)
            complementary_lengths = max_rel_length - lengths
            complementary_tensor = [torch.full((length,), max_rel_length - 1, dtype=torch.int64, device=hs.device) for
                                    length
                                    in complementary_lengths]
            complementary_tensor = [complementary_tensor for _ in range(len(indices))]
            tgt_indices = torch.cat(  # [#decoder_layer x batch_size x #bbox]
                [
                    torch.cat(
                        [torch.cat([i.to(hs.device), j]).unsqueeze(0) for i, j in zip(tgt_idx, ct)]
                    ).unsqueeze(0) for tgt_idx, ct in zip(tgt_indices, complementary_tensor)
                ]
            )
            flatten_tgt_indices = tgt_indices.reshape(-1, max_rel_length)
            lookup_table = torch.zeros_like(flatten_tgt_indices)
            for i in range(flatten_tgt_indices.shape[0]):
                lookup_table[i][flatten_tgt_indices[i]] = torch.arange(max_rel_length, device=hs.device)
            lookup_table = lookup_table.reshape(tgt_indices.size())
            i = [rel[:, 0] for rel in relationships]
            j = [rel[:, 1] for rel in relationships]
            pred_predicate_logits = []
            for img_idx, (ii, jj) in enumerate(zip(i, j)):
                sub_indices = lookup_table[:, img_idx, ii]
                obj_indices = lookup_table[:, img_idx, jj]
                delta_indices = torch.arange(0, num_decoder_layer, device=hs.device).reshape(-1, 1).repeat(
                    [1, ii.shape[0]])
                delta_indices = delta_indices * batch_size * max_match_length + img_idx * max_match_length
                sub_repr = aligned_flatten_hs[sub_indices + delta_indices]
                obj_repr = aligned_flatten_hs[obj_indices + delta_indices]
                pred_predicate_logits.append(self.predicate_emb(torch.cat([sub_repr, obj_repr], dim=-1)))
                ret = {'pred_predicate_logits': pred_predicate_logits}
        return ret

    def postprocess_outputs(self, out):
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out['pred_logits'], out['pred_boxes'], [_[:-1] for _ in out['pred_predicate_logits']])
            # out['aux_outputs'] = self._set_aux_loss(out['pred_logits'], out['pred_boxes'])
        out['pred_logits'], out['pred_boxes'] = out['pred_logits'][-1], out['pred_boxes'][-1]
        out['pred_predicate_logits'] = [_[-1] for _ in out['pred_predicate_logits']]
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outpus_predicate_class):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        aux_outputs = []
        for i in range(outputs_class.shape[0] - 1):
            aux_output = {}
            aux_output['pred_logits'] = outputs_class[i]
            aux_output['pred_boxes'] = outputs_coord[i]
            aux_output['pred_predicate_logits'] = [predicate_class[i] for predicate_class in  outpus_predicate_class]
            aux_outputs.append(aux_output)
        return aux_outputs

    # @torch.jit.unused
    # def _set_aux_loss(self, outputs_class, outputs_coord, outputs_predicate):
    #     # this is a workaround to make torchscript happy, as torchscript
    #     # doesn't support dictionary with non-homogeneous values, such
    #     # as a dict having both a Tensor and a list.
    #     aux_outputs = []
    #     for idx, (a, b) in enumerate(zip(outputs_class[:-1], outputs_coord[:-1])):
    #         c = [_[idx] for _ in outputs_predicate]
    #         aux_outputs.append({'pred_logits': a, 'pred_boxes': b, 'pred_predicate_logits': c})
    #     return aux_outputs


def is_dist_avail_and_initiaSetCriterionlized():
    pass


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, num_predicate_classes, weight_dict, eos_coef, predicate_eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_predicate_classes = num_predicate_classes
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.predicate_eos_coef = predicate_eos_coef
        self.losses = losses
        empty_weight_labels = torch.ones(self.num_classes + 1)
        empty_weight_predicate_labels = torch.ones(self.num_predicate_classes)
        empty_weight_labels[-1] = self.eos_coef
        empty_weight_predicate_labels[-1] = self.predicate_eos_coef
        self.register_buffer('empty_weight_labels', empty_weight_labels)
        self.register_buffer('empty_weight_predicate_labels', empty_weight_predicate_labels)
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight_labels)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_predicate_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_predicate_logits' in outputs
        src_predicate_logits = torch.cat(outputs['pred_predicate_logits'])
        target_classes = torch.cat([t['predicate_labels'] for t in targets])
        loss_predicate_ce = F.cross_entropy(src_predicate_logits, target_classes, self.empty_weight_predicate_labels)
        losses = {'loss_predicate_ce': loss_predicate_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['predicate_class_error'] = 100 - accuracy(src_predicate_logits, target_classes)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'predicate_labels': self.loss_predicate_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, indices):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initiaSetCriterionlized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices[-1], num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices[i], num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # We follow the most scene graph papers and use the setting #object = 150 and #predicates = 50
    num_classes = 150
    num_predicate_classes = 50
    if args.dataset_file[:2] == 'vg':
        num_classes = 150
        num_predicate_classes = 50
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETRScene(
        backbone,
        transformer,
        num_classes=num_classes,
        num_predicate_classes=num_predicate_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    matcher.to(device)
    weight_dict = {'loss_predicate_ce': args.predicate_loss_coef, 'loss_obj_ce': args.obj_loss_coef, 'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef}
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'predicate_labels']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, num_predicate_classes=num_predicate_classes, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, predicate_eos_coef=args.predicate_eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, matcher, postprocessors
