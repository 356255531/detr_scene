# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data

from torchvision.datasets import VisionDataset
from PIL import Image
import json
import time
import os
from collections import defaultdict

from datasets.coco import make_coco_transforms


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class VG:
    def __init__(self, annotation_file=None):
        """
        Constructor of VG helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns, self.imgs = dict(),dict(),dict()
        self.imgToAnns = defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        imgs = {}
        imgToAnns = dict()

        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['image_id']] = img

        print('index created!')

        # create class members
        self.imgToAnns = imgToAnns
        self.imgs = imgs

    def loadAnns(self, img_ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(img_ids):
            return [self.imgToAnns[img_id] for img_id in img_ids]
        elif type(img_ids) == int:
            return self.imgToAnns[img_ids]

    def loadImgs(self, image_ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(image_ids):
            return [self.imgs[id] for id in image_ids]
        elif type(image_ids) == int:
            return [self.imgs[image_ids]]


class VGDetection(VisionDataset):
    """`Adatepd from MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(VGDetection, self).__init__(root, transforms, transform, target_transform)
        self.vg = VG(annFile)
        self.image_ids = list(sorted(self.vg.imgs.keys()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        vg = self.vg
        img_id = self.image_ids[index]
        target = vg.loadAnns(img_id)

        path = vg.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_ids)


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

    classes = torch.tensor(anno['object_labels'], dtype=torch.int64)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    classes = classes[keep]

    target = {}
    target["boxes"] = boxes
    target["labels"] = classes
    target['image_id'] = image_id
    target['iscrowd'] = torch.zeros_like(classes)
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
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = VGScene(img_folder, ann_file, transforms=make_coco_transforms(image_set))
    return dataset
