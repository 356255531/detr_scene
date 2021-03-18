# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from random import shuffle
import numpy as np

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    cropped_target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    cropped_target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in cropped_target:
        boxes = cropped_target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        cropped_target["boxes"] = cropped_boxes.reshape(-1, 4)
        cropped_target["area"] = area
        fields.append("boxes")

    if "masks" in cropped_target:
        # FIXME should we update the area here if there are no boxes?
        cropped_target['masks'] = cropped_target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in cropped_target or "masks" in cropped_target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in cropped_target:
            cropped_boxes = cropped_target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = cropped_target['masks'].flatten(1).any(1)

        for field in fields:
            cropped_target[field] = cropped_target[field][keep]

        if "relationships" in cropped_target:
            if (keep == 0).sum().item() > 0:
                relationships = cropped_target['relationships']

                predicate_keep = torch.logical_and(keep[relationships[:, 0]], keep[relationships[:, 1]])
                cropped_target['predicate_labels'] = cropped_target['predicate_labels'][predicate_keep]

                kept_relationships = relationships[predicate_keep]
                lookup_table = torch.zeros_like(keep, dtype=torch.long)
                lookup_table[torch.where(keep)] = torch.arange(keep.sum().item())
                cropped_target['relationships'] = lookup_table[kept_relationships.reshape(-1)].reshape(kept_relationships.size())

    return cropped_image, cropped_target


def reduce_bbox(image, target, num_bbox):
    target_cpy = target.copy()
    fields = ["labels", "iscrowd", "boxes"]
    if "boxes" in target_cpy:
        boxes = target_cpy['boxes']
        if boxes.shape[0] > num_bbox:
            keep = torch.zeros((boxes.shape[0],), dtype=torch.bool)
            keep_idx = np.arange(0, boxes.shape[0])
            shuffle(keep_idx)
            keep[keep_idx[:num_bbox]] = True

            for field in fields:
                target_cpy[field] = target_cpy[field][keep]
            if "area" in target_cpy:
                target_cpy["area"] = target_cpy["area"][keep]
            if "mask" in target_cpy:
                target_cpy["mask"] = target_cpy["mask"][keep]

            if "relationships" in target_cpy:
                relationships = target_cpy['relationships']

                predicate_keep = torch.logical_and(keep[relationships[:, 0]], keep[relationships[:, 1]])
                target_cpy['predicate_labels'] = target_cpy['predicate_labels'][predicate_keep]

                kept_relationships = relationships[predicate_keep]
                lookup_table = torch.zeros_like(keep, dtype=torch.long)
                lookup_table[torch.where(keep)] = torch.arange(keep.sum().item())
                target_cpy['relationships'] = lookup_table[kept_relationships.reshape(-1)].reshape(
                    kept_relationships.size())

    return image, target_cpy


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


def relation_safe_crop(img, size, target):
    region = T.RandomCrop.get_params(img, size)
    cropped_img, cropped_target = crop(img, target, region)
    counter = 0
    while 'relationships' in cropped_target and cropped_target['relationships'].shape[0] == 0:
        if counter > 9:
            return img, target
        region = T.RandomCrop.get_params(img, size)
        cropped_img, cropped_target = crop(img, target, region)
        counter += 1
    return cropped_img, cropped_target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        return relation_safe_crop(img, self.size, target)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        return relation_safe_crop(img, [h, w], target)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class ReduceBox(object):
    def __init__(self, num_bbox=100):
        self.num_bbox = num_bbox

    def __call__(self, img, target):
        return reduce_bbox(img, target, self.num_bbox)


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
