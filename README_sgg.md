**SGGTR**: End-to-End Scene Graph Generation with Transformers
========
PyTorch training code. We added a new output to the model based on **DETR** (**DE**tection **TR**ansformer) to predict the relationship(predicate) between the two objects.


![DETR](.github/DETR.png)

**What it is**. Unlike traditional computer vision techniques, SGGTR approaches object detection and scene graph generation as a direct set prediction problem. It consists of a set-based global loss, which forces unique predictions via bipartite matching, and a Transformer encoder-decoder architecture. 
Given a fixed small set of learned object queries, SGGTR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. Due to this parallel nature, SGGTR is very fast and efficient.

# Requirements and Installation
* Python 3.8.8
* [PyTorch](http://pytorch.org/) 1.7.1
* [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) 1.7.0
* Install all dependencies run
```
pip3 install -r requirements.txt
```

## Directory structure

The basic skeleton of our source code will look like this :
```bash
├── datasets
│   ├── box_intersections_cpu
│   │   ├── **/*.py
│   │   └── bbox.c
│   ├── __pycache__
│   │   └── **/*.py
│   └── **/*.py
├── models
│   ├── __pycache__
│   │   └── **/*.py
│   └── **/*.py
├── MODEL_WEIGHTS
├── __pycache__
│   └── **/*.py
├── util
│   ├── __pycache__
│   │   └── **/*.py
│   └── **/*.py
├──visualization
│   └── start.jpg
├── **/*.py
├── license.txt
├── README.md
├── requirements.txt
└── Dockerfile
```

## Data preparation

Download and extract VG images with annotations from
[https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip);
[https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip);
[http://visualgenome.org/static/data/dataset/scene_graphs.json.zip](http://visualgenome.org/static/data/dataset/scene_graphs.json.zip).
Unzip the images.zip and images2.zip to the DS/vg/images, unzip the json.zip to the DS/vg/raw. 
We expect the directory structure to be the following:
```
├── DS
    └── vg
        ├── annotations
        │   └── **/*.json
        ├── images
        ├── raw
        │   └── **/*.json
        ├── relationships_preprocessing
        │   ├── **/*.py
        │   └── **/*.txt
        └── scene_graphs_preprocessing
            ├── rel_visual_postprocessed
            ├── **/*.py
            ├── preprocess.sh
            └── **/*.txt
```
Then, open the file DS/vg/scene_graphs_preprocessing, and run the preprocess.sh:
```
bash preprocess.sh
```
## Training
To train SGGTR on a single node for 300 epochs run:
```
python main_scene.py --resume 'MODEL_WEIGHTS/checkpoint_42.pth'
```
We train SGGTR with AdamW setting learning rate in the transformer to 1e-4 and 1e-5 in the backbone.
Horizontal flips, scales and crops are used for augmentation.
Images are rescaled to have min size 800 and max size 1333.
The transformer is trained with dropout of 0.1, and the whole model is trained with grad clip of 0.1.


## Evaluation
To evaluate SGGTR run:
```
python main_scene.py --no_aux_loss --eval --resume 'MODEL_WEIGHTS/checkpoint_42.pth'
```

## Demo
To read the video from the camera and do predictions:
```
python visualize_sgg.py --resume 'MODEL_WEIGHTS/checkpoint_42.pth'
```

# License
DETR is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

