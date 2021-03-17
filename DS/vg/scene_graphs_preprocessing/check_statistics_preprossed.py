import json
import itertools


def increase(dic, key):
    if key in dic.keys():
        dic[key] += 1
    else:
        dic[key] = 1


print('Loading dataset...')
scene_graphs = json.loads(open('../raw/preprocessed_scene_graphs.json').read())

name2occ = {}
predicate2occ = {}
for scene_graph in scene_graphs:
    for obj in scene_graph['objects']:
        increase(name2occ, obj['name'])

    for rel in scene_graph['relationships']:
        rel_name = rel['predicate']
        increase(predicate2occ, rel_name)
name2occ = {k: v for k, v in sorted(name2occ.items(), key=lambda item: item[1], reverse=True)}
predicate2occ = {k: v for k, v in sorted(predicate2occ.items(), key=lambda item: item[1], reverse=True)}
print('top {} obj names: {}'.format(len(name2occ.keys()), name2occ))
print('counts per obj: {}'.format(sum(name2occ.values()) / len(name2occ.keys())))
print('top {} rels names: {}'.format(len(predicate2occ.keys()), predicate2occ))
print('counts per rel: {}'.format(sum(predicate2occ.values()) / len(predicate2occ.keys())))