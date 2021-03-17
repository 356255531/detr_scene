import json
import itertools
from collections import Counter
import tqdm


def add_names(dic, kv):
    key, value = kv
    if not isinstance(value, list):
        value = [value]
    value = [v.split('.')[0].lower() for v in value]
    value = list(set(value))
    if key in dic.keys():
        for v in value:
            if v not in dic[key]:
                dic[key].append(v)
    else:
        dic[key] = value


def get_obj_names(obj):
    obj_name = obj.get('name', None)
    obj_names = obj.get('names', [])
    obj_names += [obj_name] if obj_name is not None else []
    obj_names = [obj_name.lower() for obj_name in obj_names]
    return obj_names


def maintain(dic, v):
    threashold = sorted(dic.values(), reverse=True)[v]
    dic = {key: value for key, value in dic.items() if value > threashold}
    dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)}
    return dic


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


print('Loading datasets...')
scene_graphs = json.loads(open('../raw/scene_graphs.json').read())

rel_obj_id2names = {}
rel_obj_id2synsets = {}
pbar = tqdm.tqdm(scene_graphs)
for scene_graph in pbar:
    pbar.set_description('selecting objects')
    rel_obj_ids = [[rel['subject_id'], rel['object_id']] for rel in scene_graph['relationships']]
    rel_obj_ids = list(set(list(itertools.chain.from_iterable(rel_obj_ids))))
    objects = [obj for obj in scene_graph['objects'] if obj['object_id'] in rel_obj_ids]
    for obj in objects:
        add_names(rel_obj_id2names, (obj['object_id'], get_obj_names(obj)))
        add_names(rel_obj_id2synsets, (obj['object_id'], obj['synsets']))
name2occ = dict(Counter(list(itertools.chain.from_iterable(list(rel_obj_id2names.values())))))
synsets2occ = dict(Counter(list(itertools.chain.from_iterable(list(rel_obj_id2synsets.values())))))
filtered_synsets2occ = maintain(synsets2occ, 150)
print('top 150 obj synset: {}'.format(filtered_synsets2occ))
print('counts per obj: {}'.format(sum(filtered_synsets2occ.values()) / 150))
filtered_name2occ = maintain(name2occ, 150)
print('top 150 obj names: {}'.format(filtered_name2occ))
print('counts per obj: {}'.format(sum(filtered_name2occ.values()) / 150))

filtered_names = filtered_name2occ.keys()
filtered_synsets = filtered_synsets2occ.keys()

name_rel_id2predicate = {}
synsets_rel_id2predicate = {}
pbar = tqdm.tqdm(scene_graphs)
for scene_graph in pbar:
    pbar.set_description('selecting relationships')

    name_objects = [
        obj for obj in scene_graph['objects'] \
        if len(set(get_obj_names(obj)).intersection(filtered_names)) != 0
    ]
    name_obj_ids = [obj['object_id'] for obj in name_objects]

    synsets_objects = [
        obj for obj in scene_graph['objects'] \
        if len(set([synset.split('.')[0].lower() for synset in obj['synsets']]).intersection(filtered_synsets)) != 0
    ]
    synsets_obj_ids = [obj['object_id'] for obj in synsets_objects]

    name_rels = [
        rel for rel in scene_graph['relationships'] \
        if rel['subject_id'] in name_obj_ids and rel['object_id'] in name_obj_ids
    ]
    synsets_rels = [
        rel for rel in scene_graph['relationships'] \
        if rel['subject_id'] in synsets_obj_ids and rel['object_id'] in synsets_obj_ids
    ]
    for rel in name_rels:
        add_names(name_rel_id2predicate, (rel['relationship_id'], predicate_preprocess(rel['predicate'])))
    for rel in synsets_rels:
        add_names(synsets_rel_id2predicate, (rel['relationship_id'], predicate_preprocess(rel['predicate'])))
name_predicate2occ = dict(Counter(list(itertools.chain.from_iterable(list(name_rel_id2predicate.values())))))
synset_predicate2occ = dict(Counter(list(itertools.chain.from_iterable(list(synsets_rel_id2predicate.values())))))
filtered_synset_predicate2occ = maintain(synset_predicate2occ, 50)
print('top 50 rels synset: {}'.format(filtered_synset_predicate2occ))
print('counts per rel: {}'.format(sum(filtered_synset_predicate2occ.values()) / 50))
filtered_name_predicate2occ = maintain(name_predicate2occ, 50)
print('top 50 rels names: {}'.format(filtered_name_predicate2occ))
print('counts per rel: {}'.format(sum(filtered_name_predicate2occ.values()) / 50))
