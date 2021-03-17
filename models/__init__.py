# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build as build_detr_detection
from .detr_scene import build as build_detr_scene


def build_scene_model(args):
    return build_detr_scene(args)


def build_detection_model(args):
    return build_detr_detection(args)
