import json
import random
import tqdm


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


def get_obj_names(obj):
    obj_name = obj.get('name', None)
    obj_names = obj.get('names', [])
    obj_names += [obj_name] if obj_name is not None else []
    obj_names = [obj_name.lower() for obj_name in obj_names]
    return obj_names


OBJECT_NAME_DICT = {'man': 79704, 'window': 54594, 'person': 52881, 'woman': 35746, 'building': 35262, 'shirt': 32152, 'wall': 31664, 'tree': 29923, 'sign': 24032, 'head': 23846, 'ground': 23061, 'table': 23059, 'hand': 21127, 'grass': 20289, 'sky': 20003, 'water': 19003, 'pole': 18672, 'light': 17452, 'leg': 17271, 'car': 17212, 'people': 15637, 'hair': 15551, 'clouds': 14591, 'ear': 14532, 'plate': 13807, 'street': 13485, 'trees': 13088, 'road': 12877, 'shadow': 12615, 'eye': 12528, 'leaves': 12083, 'snow': 11935, 'train': 11709, 'hat': 11693, 'door': 11561, 'boy': 11104, 'pants': 10956, 'wheel': 10730, 'nose': 10632, 'fence': 10337, 'sidewalk': 10242, 'girl': 9827, 'jacket': 9820, 'field': 9704, 'floor': 9554, 'tail': 9537, 'chair': 9315, 'clock': 9149, 'handle': 9084, 'face': 8850, 'boat': 8795, 'line': 8777, 'arm': 8746, 'plane': 8293, 'horse': 8157, 'bus': 8145, 'dog': 8105, 'windows': 8002, 'giraffe': 7959, 'bird': 7896, 'cloud': 7881, 'elephant': 7830, 'helmet': 7755, 'shorts': 7595, 'food': 7282, 'leaf': 7210, 'shoe': 7157, 'zebra': 7036, 'glass': 7023, 'cat': 6996, 'bench': 6761, 'glasses': 6727, 'bag': 6714, 'flower': 6617, 'background': 6550, 'rock': 6213, 'cow': 6195, 'foot': 6167, 'sheep': 6164, 'letter': 6141, 'picture': 6130, 'logo': 6118, 'player': 6067, 'skateboard': 6024, 'bottle': 6020, 'tire': 6018, 'stripe': 6002, 'umbrella': 5986, 'surfboard': 5964, 'shelf': 5949, 'bike': 5874, 'number': 5831, 'motorcycle': 5828, 'part': 5820, 'tracks': 5804, 'mirror': 5750, 'truck': 5619, 'tile': 5602, 'mouth': 5585, 'pizza': 5535, 'bowl': 5525, 'bear': 5394, 'spot': 5328, 'kite': 5310, 'bed': 5297, 'roof': 5259, 'counter': 5257, 'post': 5231, 'dirt': 5207, 'beach': 5106, 'flowers': 5104, 'top': 5021, 'jeans': 5021, 'legs': 4980, 'cap': 4862, 'pillow': 4777, 'box': 4748, 'neck': 4703, 'house': 4631, 'reflection': 4616, 'lights': 4558, 'plant': 4515, 'trunk': 4468, 'sand': 4455, 'cup': 4419, 'child': 4368, 'button': 4337, 'shoes': 4325, 'wing': 4325, 'writing': 4289, 'sink': 4207, 'desk': 4184, 'board': 4170, 'wave': 4150, 'sunglasses': 4132, 'edge': 4119, 'paper': 3994, 'vase': 3987, 'lamp': 3953, 'lines': 3936, 'brick': 3908, 'phone': 3895, 'ceiling': 3864, 'book': 3790, 'airplane': 3696, 'laptop': 3693, 'vehicle': 3689, 'headlight': 3678, 'coat': 3643, 'ball': 3624}
PREDICATE_DICT = {'on': 285176, 'have': 124491, 'in': 89356, 'wearing': 80121, 'of': 72315, 'with': 20608, 'behind': 18374, 'holding': 13168, 'standing': 12313, 'near': 11789, 'sitting': 11580, 'next': 10407, 'walking': 6993, 'riding': 6813, 'are': 6719, 'by': 6532, 'under': 6330, 'in front of': 5594, 'on side of': 5396, 'above': 5253, 'hanging': 4737, 'at': 3540, 'parked': 3308, 'beside': 3215, 'flying': 3044, 'attached to': 2946, 'eating': 2728, 'looking': 2441, 'carrying': 2395, 'laying': 2344, 'over': 2265, 'inside': 2118, 'belonging': 1976, 'covered': 1891, 'growing': 1678, 'covering': 1642, 'driving': 1536, 'lying': 1492, 'around': 1455, 'below': 1412, 'painted': 1386, 'against': 1383, 'along': 1353, 'for': 1281, 'playing': 1141, 'crossing': 1134, 'mounted': 1084, 'outside': 1012, 'watching': 997, 'to': 988}
filerted_obj_names = OBJECT_NAME_DICT.keys()
filerted_predicates = PREDICATE_DICT.keys()

print('Loading scene_graphs.json...')
scene_graphs = json.loads(open('../raw/scene_graphs.json').read())

num_scene_graphs = len(scene_graphs)
pbar = tqdm.tqdm(range(num_scene_graphs))
for i in pbar:
    pbar.set_description('Filtering according to selected object and predicate names')
    objects = scene_graphs[i]['objects']
    num_objects = len(objects)
    filtered_objs = []
    for j in range(num_objects):
        obj = objects[j]
        obj_names = get_obj_names(obj)
        common_names = list(set(obj_names).intersection(filerted_obj_names))
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
        if predicate_preprocess(rel['predicate']) in filerted_predicates and \
                rel['subject_id'] in filtered_obj_ids and rel['object_id'] in filtered_obj_ids:
            rel['predicate'] = predicate_preprocess(rel['predicate'])
            del rel['synsets']
            filtered_rels.append(rel)
    scene_graphs[i]['relationships'] = filtered_rels
with open('../raw/preprocessed_scene_graphs.json', 'w') as outfile:
    json.dump(scene_graphs, outfile)
