import json

scene_graphs = json.loads(open('../annotations/raw/preprocessed_scene_graphs.json').read())
images = json.loads(open('../annotations/raw/image_data.json').read())

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

    for rel in scene_graph['relationships']:
        rel_name = rel['predicate']
        increase(rel_name_dict, rel_name)
print('top 150 obj names: {}'.format(obj_name_dict))
print('counts per obj: {}'.format(sum(obj_name_dict.values()) / 150))
print('top 50 rels names: {}'.format(rel_name_dict))
print('counts per rel: {}'.format(sum(rel_name_dict.values()) / 50))

sorted_obj_names = list({k: v for k, v in sorted(obj_name_dict.items(), key=lambda item: item[1], reverse=True)}.keys())
sorted_rel_names = list({k: v for k, v in sorted(rel_name_dict.items(), key=lambda item: item[1], reverse=True)})
print('sorted_obj_names: {}'.format(sorted_obj_names))
print('sorted_rel_names: {}'.format(sorted_rel_names))

annotations = []
available_image_ids = []
for scene_graph in scene_graphs:
    bbox = []
    object_labels = []
    obj_id_idx = {}
    for idx, obj in enumerate(scene_graph['objects']):
        bbox.append([obj['x'], obj['y'], obj['w'], obj['h']])
        object_labels.append(sorted_obj_names.index(obj['name']))
        obj_id_idx[obj['object_id']] = idx

    predicate_labels = {}
    for rel in scene_graph['relationships']:
        predicate_label_idx = str((obj_id_idx[rel['subject_id']], obj_id_idx[rel['object_id']]))
        predicate_labels[predicate_label_idx] = sorted_rel_names.index(rel['predicate'])

    if len(object_labels) == 0 or len(predicate_labels) == 0:
        continue

    available_image_ids.append(scene_graph['image_id'])
    annotations.append(
        {
            'image_id': scene_graph['image_id'],
            'bbox': bbox,
            'object_labels': object_labels,
            'predicate_labels': predicate_labels
        }
    )

print('{} available scene graphs in total'.format(len(annotations)))

available_image_ids = set(available_image_ids)
images = [image for image in images if image['image_id'] in available_image_ids]
num_images = len(images)
for i in range(num_images):
    images[i]['id'] = images[i]['image_id']
    del images[i]['image_id']
    images[i]['file_name'] = str(images[i]['id']) + '.jpg'

postprocessed_scene_graphs = {'images': images, 'annotations': annotations}


with open('../annotations/raw/postprocessed_scene_graphs.json', 'w') as outfile:
    json.dump(postprocessed_scene_graphs, outfile)
