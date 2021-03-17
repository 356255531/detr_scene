import json
import random


def predicate_preprocess(predicate):
    predicate = predicate.lower()
    if predicate not in ['in front of', 'on side of', 'attached to']:
        predicate = predicate.split(' ')[0]
    if predicate == 'wears':
        predicate = 'wearing'
    if predicate == 'holds':
        predicate = 'holding'
    if predicate == 'are on':
        predicate = 'on'
    if predicate == 'are in':
        predicate = 'in'
    if predicate == 'on front of':
        predicate = 'in front of'
    if predicate == 'has':
        predicate = 'have'
    return predicate


OBJECT_NAME_DICT = {'man': 79697, 'window': 54583, 'person': 52876, 'woman': 35742, 'building': 35231, 'shirt': 32121, 'wall': 31638, 'tree': 29918, 'sign': 24021, 'head': 23832, 'ground': 23043, 'table': 23035, 'hand': 21122, 'grass': 20272, 'sky': 19985, 'water': 18989, 'pole': 18665, 'light': 17443, 'leg': 17269, 'car': 17205, 'people': 15635, 'hair': 15543, 'clouds': 14590, 'ear': 14526, 'plate': 13797, 'street': 13470, 'trees': 13083, 'road': 12860, 'shadow': 12611, 'eye': 12524, 'leaves': 12079, 'snow': 11919, 'train': 11692, 'hat': 11680, 'door': 11552, 'boy': 11104, 'pants': 10953, 'wheel': 10730, 'nose': 10629, 'fence': 10334, 'sidewalk': 10233, 'girl': 9826, 'jacket': 9813, 'field': 9698, 'floor': 9549, 'tail': 9532, 'chair': 9308, 'clock': 9144, 'handle': 9083, 'face': 8846, 'boat': 8794, 'line': 8777, 'arm': 8743, 'plane': 8285, 'horse': 8156, 'bus': 8136, 'dog': 8100, 'windows': 7995, 'giraffe': 7950, 'bird': 7892, 'cloud': 7880, 'elephant': 7822, 'helmet': 7748, 'shorts': 7587, 'food': 7277, 'leaf': 7210, 'shoe': 7155, 'zebra': 7031, 'glass': 7021, 'cat': 6990, 'bench': 6757, 'glasses': 6723, 'bag': 6713, 'flower': 6615, 'background': 6539, 'rock': 6213, 'cow': 6190, 'foot': 6165, 'sheep': 6161, 'letter': 6140, 'picture': 6126, 'logo': 6116, 'player': 6065, 'bottle': 6020, 'tire': 6017, 'skateboard': 6017, 'stripe': 6001, 'umbrella': 5979, 'surfboard': 5954, 'shelf': 5944, 'bike': 5868, 'number': 5828, 'part': 5820, 'motorcycle': 5818, 'tracks': 5801, 'mirror': 5747, 'truck': 5610, 'tile': 5602, 'mouth': 5584, 'bowl': 5522, 'pizza': 5521, 'bear': 5389, 'spot': 5328, 'kite': 5307, 'bed': 5295, 'roof': 5256, 'counter': 5252, 'post': 5230, 'dirt': 5204, 'beach': 5102, 'flowers': 5101, 'jeans': 5018, 'top': 5016, 'legs': 4975, 'cap': 4860, 'pillow': 4775, 'box': 4748, 'neck': 4697, 'house': 4629, 'reflection': 4612, 'lights': 4554, 'plant': 4515, 'trunk': 4465, 'sand': 4451, 'cup': 4416, 'child': 4368, 'button': 4334, 'wing': 4325, 'shoes': 4323, 'writing': 4284, 'sink': 4204, 'desk': 4176, 'board': 4168, 'wave': 4147, 'sunglasses': 4129, 'edge': 4119, 'paper': 3994, 'vase': 3983, 'lamp': 3950, 'lines': 3936, 'brick': 3907, 'phone': 3888, 'ceiling': 3860, 'book': 3785, 'airplane': 3695, 'laptop': 3691, 'vehicle': 3686, 'headlight': 3678, 'coat': 3639}
PREDICATE_DICT = {'on': 284572, 'have': 124330, 'in': 89147, 'wearing': 80107, 'of': 72283, 'with': 20517, 'behind': 18353, 'holding': 12781, 'standing': 12307, 'near': 11739, 'sitting': 11569, 'next': 10376, 'walking': 6991, 'riding': 6813, 'are': 6718, 'by': 6517, 'under': 6319, 'in front of': 5539, 'on side of': 5370, 'above': 5224, 'hanging': 4719, 'at': 3536, 'parked': 3308, 'beside': 3210, 'flying': 3032, 'attached to': 2925, 'eating': 2727, 'looking': 2407, 'carrying': 2389, 'laying': 2333, 'over': 2252, 'inside': 2104, 'belonging': 1976, 'covered': 1891, 'growing': 1678, 'covering': 1642, 'driving': 1536, 'lying': 1456, 'around': 1454, 'below': 1408, 'painted': 1386, 'against': 1381, 'along': 1353, 'for': 1272, 'crossing': 1134, 'mounted': 1083, 'playing': 1053, 'outside': 1012, 'watching': 992}

scene_graphs = json.loads(open('../annotations/scene_graphs.json').read())

num_scene_graphs = len(scene_graphs)
for i in range(num_scene_graphs):
    objects = scene_graphs[i]['objects']
    num_objects = len(objects)
    filtered_objs = []
    for j in range(num_objects):
        obj = objects[j]
        common_names = [name.lower() for name in obj['names']]
        common_names = list(set(common_names).intersection(OBJECT_NAME_DICT.keys()))
        if len(common_names) != 0:
            name = random.choice(common_names)
            del obj['names']
            del obj['synsets']
            obj['name'] = name
            filtered_objs.append(objects[j])
    scene_graphs[i]['objects'] = filtered_objs
    filtered_obj_ids = [obj['object_id'] for obj in filtered_objs]

    rels = scene_graphs[i]['relationships']
    num_rels = len(rels)
    filtered_rels = []
    for j in range(num_rels):
        rel = rels[j]
        if predicate_preprocess(rel['predicate']) in PREDICATE_DICT.keys() and \
                rel['subject_id'] in filtered_obj_ids and rel['object_id'] in filtered_obj_ids:
            rel['predicate'] = predicate_preprocess(rel['predicate'])
            del rel['synsets']
            filtered_rels.append(rel)
    scene_graphs[i]['relationships'] = filtered_rels

with open('../annotations/raw/preprocessed_scene_graphs.json', 'w') as outfile:
    json.dump(scene_graphs, outfile)
