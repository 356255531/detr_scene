# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data

from datasets.vg_detection import VGDetection
from datasets.coco import make_coco_transforms


class VGScene(VGDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(VGScene, self).__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(VGScene, self).__getitem__(idx)
        image_id = self.image_ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = prepare_vg(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
            if len(target['relationships']) == 0:
                return None
        return img, target


def prepare_vg(image, target):
    w, h = image.size

    image_id = target["image_id"]
    image_id = torch.tensor([image_id])

    anno = target["annotations"]

    # guard against no boxes via resizing
    boxes = torch.as_tensor(anno['bbox'], dtype=torch.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)

    object_labels = torch.tensor(anno['object_labels'], dtype=torch.int64)
    predicate_labels = torch.tensor(anno['predicate_labels'], dtype=torch.int64)
    relationships = torch.tensor(anno['relationships'], dtype=torch.int64)
    mask = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

    target = {}
    target["boxes"] = boxes
    target["labels"] = object_labels
    target["predicate_labels"] = predicate_labels
    target["relationships"] = relationships
    target["mask"] = mask
    target['image_id'] = image_id
    target['iscrowd'] = torch.zeros_like(object_labels)
    target["orig_size"] = torch.as_tensor([int(h), int(w)])
    target["size"] = torch.as_tensor([int(h), int(w)])

    return image, target


def build(image_set, args):
    root = Path(args.file_path)
    assert root.exists(), f'provided file path {root} does not exist'
    mode = 'scene_graphs'
    PATHS = {
        "train": (root / "images", root / "annotations" / f'{mode}_train.json'),
        "val": (root / "images", root / "annotations" / f'{mode}_val.json'),
        "test": (root / "images", root / "annotations" / f'{mode}_test.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = VGScene(img_folder, ann_file, transforms=make_coco_transforms(image_set))
    return dataset
