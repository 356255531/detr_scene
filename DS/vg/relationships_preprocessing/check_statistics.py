import json
import itertools
from collections import Counter
import tqdm


def get_obj_names(obj):
    obj_name = obj.get('name', None)
    obj_names = obj.get('names', [])
    obj_names += [obj_name] if obj_name is not None else []
    obj_names = [obj_name.lower() for obj_name in obj_names]
    return obj_names


def objs_in(rel, keys):
    sub_names = get_obj_names(rel['subject'])
    if len(set(sub_names).intersection(keys)) == 0:
        return False

    obj_names = get_obj_names(rel['object'])
    if len(set(obj_names).intersection(keys)) == 0:
        return False
    return True


def maintain(dic, v):
    threashold = sorted(dic.values(), reverse=True)[v-1]
    dic = {key: value for key, value in dic.items() if value > threashold}
    dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)}
    return dic


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
relationships = json.loads(open('../raw/relationships.json').read())


obj_id2names = {}
obj_id2synsets = {}
pbar = tqdm.tqdm(relationships)
for rels in pbar:
    pbar.set_description('Selecting objects')
    for rel in rels['relationships']:
        add_names(obj_id2names, (rel['subject']['object_id'], get_obj_names(rel['subject'])))
        add_names(obj_id2names, (rel['object']['object_id'], get_obj_names(rel['object'])))

        add_names(obj_id2synsets, (rel['subject']['object_id'], rel['subject']['synsets']))
        add_names(obj_id2synsets, (rel['object']['object_id'], rel['object']['synsets']))
name2occ = dict(Counter(list(itertools.chain.from_iterable(list(obj_id2names.values())))))
synsets2occ = dict(Counter(list(itertools.chain.from_iterable(list(obj_id2synsets.values())))))
filtered_synsets2occ = maintain(synsets2occ, 150)
print('top 150 obj synset: {}'.format(filtered_synsets2occ))
print('counts per obj: {}'.format(sum(filtered_synsets2occ.values()) / 150))
filtered_name2occ = maintain(name2occ, 150)
print('top 150 obj names: {}'.format(filtered_name2occ))
print('counts per obj: {}'.format(sum(filtered_name2occ.values()) / 150))

filered_names = filtered_name2occ.keys()
filtered_synsets = filtered_synsets2occ.keys()
name_rel_id2predicates = {}
synsets_rel_id2predicates = {}
pbar = tqdm.tqdm(relationships)
for rels in pbar:
    pbar.set_description('Selecting relationships')
    for rel in rels['relationships']:
        if objs_in(rel, filered_names):
            add_names(name_rel_id2predicates, (rel['relationship_id'], predicate_preprocess(rel['predicate'])))
        if objs_in(rel, filtered_synsets):
            add_names(synsets_rel_id2predicates, (rel['relationship_id'], rel['synsets']))
name_predicate2occ = dict(Counter(list(itertools.chain.from_iterable(list(name_rel_id2predicates.values())))))
synsets_predicate2occ = dict(Counter(list(itertools.chain.from_iterable(list(synsets_rel_id2predicates.values())))))
filtered_synsets_predicate2occ = maintain(synsets_predicate2occ, 50)
print('top 50 rels synset: {}'.format(filtered_synsets_predicate2occ))
print('counts per rel: {}'.format(sum(filtered_synsets_predicate2occ.values()) / 50))
filtered_name_predicate2occ = maintain(name_predicate2occ, 50)
print('top 50 rels names: {}'.format(filtered_name_predicate2occ))
print('counts per rel: {}'.format(sum(filtered_name_predicate2occ.values()) / 50))
