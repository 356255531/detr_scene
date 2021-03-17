import json
import itertools

scene_graphs = json.loads(open('../annotations/raw/preprocessed_scene_graphs.json').read())
obj_name_dict = {}
rel_name_dict = {}


def increase(dic, key):
    if key in dic.keys():
        dic[key] += 1
    else:
        dic[key] = 1


for scene_graph in scene_graphs:
    for obj in scene_graph['objects']:
        increase(obj_name_dict, obj['name'])
print('top 150 obj names: {}'.format(obj_name_dict))
print('counts per obj: {}'.format(sum(obj_name_dict.values()) / 150))

for scene_graph in scene_graphs:
    for rel in scene_graph['relationships']:
        rel_name = rel['predicate']
        increase(rel_name_dict, rel_name)
print('top 50 rels names: {}'.format(rel_name_dict))
print('counts per rel: {}'.format(sum(rel_name_dict.values()) / 50))