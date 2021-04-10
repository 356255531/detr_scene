import tkinter as tk
import matplotlib.pyplot as plt
import datasets.transforms as T
import argparse
import torch
import numpy as np
import cv2
import networkx as nx
from datasets.do_sgg_eval import non_max_suppression,prepare_test_pairs
from main_scene import get_args_parser
from PIL import Image, ImageTk
from models import build_scene_model
from util import box_ops


def prediction(image):
    (x, y) = image.size

    transform = T.Compose([
        # T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tar = {}
    im = transform(image, tar)
    ima = []
    ima.append(im[0])
    outputs = model(ima)
    results = postprocessors['bbox'](outputs)
    pre_class_mask = results[0]['classification scores'].gt(0.7)
    pre_class = results[0]['classification labels'][pre_class_mask]
    pre_class_idx = torch.arange(0, 100)[pre_class_mask]

    pre_class_idx_pick = non_max_suppression(pre_class_idx, pre_class, results[0]['boxes'][pre_class_idx],
                                             results[0]['classification scores'][pre_class_idx], threshold=0.6)
    pre_class_idx = pre_class_idx[pre_class_idx_pick]
    pre_class = results[0]['classification labels'][pre_class_idx]
    pre_class_score = results[0]['classification scores'][pre_class_idx]

    # get rel_idx
    preboxes = results[0]['boxes'][pre_class_idx]
    rel_idx = prepare_test_pairs(device, preboxes)

    pre_sub_idx = pre_class_idx[rel_idx[:, 0]]
    pre_ob_idx = pre_class_idx[rel_idx[:, 1]]
    pre_predicate_idx = pre_sub_idx * 100 + pre_ob_idx
    pre_predicate = results[0]['predicate labels'][pre_predicate_idx]
    pre_predicate_scores = results[0]['predicate scores'][pre_predicate_idx].reshape(-1, 1)


    triplet_scores = torch.cat((pre_class_score[rel_idx[:, 0]].reshape(-1, 1), pre_class_score[rel_idx[:, 1]].reshape(-1, 1)),1)
    triplet_scores = torch.cat((triplet_scores, pre_predicate_scores),1)
    scores_overall = torch.prod(triplet_scores,1)
    sorted, indices = torch.sort(scores_overall,dim=0,descending=True)

    pre_sub = pre_class[rel_idx[:, 0][indices]]
    pre_obj = pre_class[rel_idx[:, 1][indices]]
    pre_rel = pre_predicate[indices]

    box = box_ops.box_cxcywh_to_xywh(preboxes)

    image_array = np.array(image)
    im_output = Image.fromarray(image_array)
    plt.imshow(im_output)

    ax = plt.gca()

    li = list(OBJECT_NAME_DICT.keys())
    colours = ["#FF0000", "#FF8000", "#FFFF00", "#00FF00", "#00FFFF", "#0000FF", "#007FFF", "#7F00FF", "#FF00FF", "#7F7F7F",
               'peru', 'brown', 'chartreuse', 'lightcoral', 'gold', 'hotpink', 'deepskyblue', 'mediumpurple', 'olive',
               'dimgrey',"#FF8800", "#FF8088", "#FFFF88", "#008800", "#008888", "#000088"]
    for n in range(len(box)):
        ax.add_patch(plt.Rectangle((box[n][0] * x, box[n][1] * y),
                                   box[n][2] * x, box[n][3] * y, fill=False, color=colours[n], linewidth=3))
        text = f'{n,li[pre_class[n]]}'
        ax.text(box[n][0] * x, box[n][1] * y, text, fontsize=15, bbox=dict(facecolor=colours[n], alpha=0.5))
    plt.axis('off')
    plt.savefig('save_obj.jpg')
    plt.close('all')

    if len(pre_rel)<10:
        num = len(pre_rel)
    else:
        num = 10
    predicate_li = list(PREDICATE_DICT.keys())

    ## show prediction tuple
    # image = Image.open('start.jpg').resize((720, 720))
    # image_array = np.array(image)
    # im_output = Image.fromarray(image_array)
    # plt.imshow(im_output)

    # ax = plt.gca()
    # num = 0
    # rel_dict = {}
    # for i in range(n):
    #     key = rel_idx[:, 1][indices][i].item()*30 + rel_idx[:, 0][indices][i].item()
    #     if key not in rel_dict:
    #         key = rel_idx[:, 0][indices][i].item() * 30 + rel_idx[:, 1][indices][i].item()
    #         rel_dict[key] = 1
    #         text = f'{rel_idx[:, 0][indices][i].item(),li[pre_sub[i]]}'
    #         ax.text(0, 10+num*40, text, fontsize=6, bbox=dict(boxstyle='round,pad=0.5',facecolor=colours[rel_idx[:, 0][indices][i].item()], alpha=0.5))
    #         text = f'{predicate_li[pre_rel[i]]}'
    #         ax.text(200, num * 40, text, fontsize=8,color='blue')
    #         ax.arrow(110, 3+num*40, 250, 0, head_width=20, head_length=20, shape="full", fc='gray',ec='gray', alpha=0.5,
    #                  overhang=0.5)
    #         text = f'{rel_idx[:, 1][indices][i].item(),li[pre_obj[i]]}'
    #         ax.text(400, 10+num * 40, text, fontsize=6, bbox=dict(boxstyle='round,pad=0.5',facecolor=colours[rel_idx[:, 1][indices][i].item()], alpha=0.5))
    #         num += 1
    # plt.axis('off')
    # plt.savefig('save_rel.jpg')
    # plt.show()

    # plt.rcParams.update({
    #     'figure.figsize': (7.2, 7.2)
    # })
    rel_dict = {}
    G = nx.DiGraph()

    for i in range(len(box)):
        text = f'{i,li[pre_class[i]]}'
        G.add_node(text)


    for i in range(num):
        key = rel_idx[:, 1][indices][i].item() * 30 + rel_idx[:, 0][indices][i].item()
        if key not in rel_dict:
            key = rel_idx[:, 0][indices][i].item() * 30 + rel_idx[:, 1][indices][i].item()
            rel_dict[key] = 1
            text1 = f'{rel_idx[:, 0][indices][i].item(), li[pre_sub[i]]}'
            text2 = predicate_li[pre_rel[i]]
            text3= f'{rel_idx[:, 1][indices][i].item(), li[pre_obj[i]]}'
            G.add_edge(text1, text3,l=text2)

    nx.draw(G,pos = nx.spiral_layout(G),node_size=1500, with_labels=True, node_color=colours[:len(box)])
    edge_labels = nx.get_edge_attributes(G, "l")
    nx.draw_networkx_edge_labels(G,pos=nx.spiral_layout(G),edge_labels=edge_labels,label_pos=0.5,font_size=15)
    plt.savefig('save_rel.jpg')
    plt.close('all')


def showfunc():
    global imaLabel,relLabel
    ## load in the image file
    # imaPath = filedialog.askopenfilename()
    # image = Image.open(imaPath)
    # prediction(image)
    # newImage = Image.open('save_obj.jpg')
    # newCover = ImageTk.PhotoImage(image=newImage)
    # imaLabel.configure(image=newCover)
    # imaLabel.image = newCover
    # imaLabel.update()
    #
    # newImage = Image.open('save_rel.jpg').resize((720,720))
    # newCover = ImageTk.PhotoImage(image=newImage)
    # relLabel.configure(image=newCover)
    # relLabel.image = newCover
    # relLabel.update()

    # # load in the video file
    # videoPath = filedialog.askopenfilename()
    # videoCapture = cv2.VideoCapture(videoPath)
    # # get a frame
    # sucess, frame = videoCapture.read()
    # while (sucess):
    #     sucess, frame = videoCapture.read()
    #     cv2frame = cv2.resize(frame, (720, 500))  # resize it to (1024,768)
    #     cv2.imwrite('frame.jpg', cv2frame)
    #     image = Image.open('frame.jpg')
    #     prediction(image)
    #     newImage = Image.open('save_obj.jpg')
    #     newCover = ImageTk.PhotoImage(image=newImage)
    #     imaLabel.configure(image=newCover)
    #     imaLabel.image = newCover
    #     imaLabel.update()
    #
    #     newImage = Image.open('save_rel.jpg').resize((720, 720))
    #     newCover = ImageTk.PhotoImage(image=newImage)
    #     relLabel.configure(image=newCover)
    #     relLabel.image = newCover
    #     relLabel.update()

    #load video from camera
    videoCapture = cv2.VideoCapture(0)
    # get a frame
    sucess, frame = videoCapture.read()
    while (sucess):
        sucess, frame = videoCapture.read()
        cv2frame = cv2.resize(frame, (720, 500))
        cv2.imwrite('frame.jpg', cv2frame)
        image = Image.open('frame.jpg')
        prediction(image)
        newImage = Image.open('save_obj.jpg')
        newCover = ImageTk.PhotoImage(image=newImage)
        imaLabel.configure(image=newCover)
        imaLabel.image = newCover
        imaLabel.update()

        newImage = Image.open('save_rel.jpg').resize((720, 720))
        newCover = ImageTk.PhotoImage(image=newImage)
        relLabel.configure(image=newCover)
        relLabel.image = newCover
        relLabel.update()


def show_window():
    global imaLabel,relLabel
    window = tk.Tk()
    window.title('scene graph generation')
    window.geometry('1500x800')

    image = Image.open('start.jpg').resize((720,500))
    cover = ImageTk.PhotoImage(image=image)
    imaLabel = tk.Label(window, width=720, height=500, bd=0, image=cover)
    imaLabel.place(x=0,y=50)
    image1 = Image.open('start.jpg').resize((720,720))
    cover1 = ImageTk.PhotoImage(image=image1)
    relLabel = tk.Label(window, width=720, height=720, bd=0, image=cover1)
    relLabel.place(x=750, y=50)

    pre_label = tk.Label(window, text='prediction relationships:', font=('Arial',15), width=30, height=2)
    pre_label.place(x=750, y=0)
    det_label = tk.Label(window, text='object detection:', font=('Arial',15), width=30, height=2)
    det_label.place(x=0, y=0)

    b = tk.Button(window,text='open the camera', font=('Arial',15),width=15, height=2,
              cursor='hand2', command=showfunc)
    b.place(x=300,y=600)
    window.mainloop()


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
                    'coat': 3639, 'a': 222}
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

parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
device = torch.device(args.device)
model, criterion, postprocessors = build_scene_model(args)
model.to(device)
checkpoint = torch.load(args.resume, map_location='cpu')
model.load_state_dict(checkpoint['model'])
show_window()
