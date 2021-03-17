import json
import tqdm


def increase(dic, key):
    if key in dic.keys():
        dic[key] += 1
    else:
        dic[key] = 1


print('Loading datasets...')
scene_graphs = json.loads(open('../raw/preprocessed_scene_graphs.json').read())
images = json.loads(open('../raw/image_data.json').read())

name2occ = {}
predicate2occ = {}
pbar = tqdm.tqdm(scene_graphs)
for scene_graph in pbar:
    pbar.set_description('Getting statistics')
    for obj in scene_graph['objects']:
        increase(name2occ, obj['name'])

    for rel in scene_graph['relationships']:
        rel_name = rel['predicate']
        increase(predicate2occ, rel_name)
name2occ = {k: v for k, v in sorted(name2occ.items(), key=lambda item: item[1], reverse=True)}
predicate2occ = {k: v for k, v in sorted(predicate2occ.items(), key=lambda item: item[1], reverse=True)}
print('Top {} obj names: {}'.format(len(name2occ.keys()), name2occ))
print('Counts per obj: {}'.format(sum(name2occ.values()) / len(name2occ.keys())))
print('Top {} rels names: {}'.format(len(predicate2occ.keys()), predicate2occ))
print('Counts per rel: {}'.format(sum(predicate2occ.values()) / len(predicate2occ.keys())))


filtered_names = list(name2occ.keys())
filtered_predicates = list(predicate2occ.keys())
print('Filtered names: {}'.format(filtered_names))
print('Filtered predicates: {}'.format(filtered_predicates))
name2object_label = {name: label for label, name in enumerate(filtered_names)}
object_label2name = {int(label): name for label, name in enumerate(filtered_names)}
predicate2predicate_label = {predicate: label for label, predicate in enumerate(filtered_predicates)}
predicate_label2predicate = {int(label): predicate for label, predicate in enumerate(filtered_predicates)}

annotations = []
available_image_ids = []
pbar = tqdm.tqdm(scene_graphs)
for scene_graph in pbar:
    pbar.set_description('Converting scene graphs')
    bbox = []
    object_labels = []
    relationships = []
    predicate_labels = []
    obj_id2obj_idx = {}
    for idx, obj in enumerate(scene_graph['objects']):
        bbox.append([obj['x'], obj['y'], obj['w'], obj['h']])
        object_labels.append(filtered_names.index(obj['name']))
        obj_id2obj_idx[obj['object_id']] = idx

    for rel in scene_graph['relationships']:
        relationships.append([obj_id2obj_idx[rel['subject_id']], obj_id2obj_idx[rel['object_id']]])
        predicate_labels.append(filtered_predicates.index(rel['predicate']))

    if len(object_labels) == 0 or len(predicate_labels) == 0:
        continue

    available_image_ids.append(scene_graph['image_id'])
    annotations.append(
        {
            'image_id': scene_graph['image_id'],
            'bbox': bbox,
            'object_labels': object_labels,
            'relationships': relationships,
            'predicate_labels': predicate_labels
        }
    )
print('{} available scene graphs in total'.format(len(annotations)))

available_image_ids = set(available_image_ids)
images = [image for image in images if image['image_id'] in available_image_ids]
num_images = len(images)
pbar = tqdm.tqdm(range(num_images))
for i in pbar:
    pbar.set_description('Adding file name to image...')
    images[i]['file_name'] = str(images[i]['image_id']) + '.jpg'

postprocessed_scene_graphs = {
    'images': images, 'annotations': annotations,
    'name2object_label': name2object_label, 'object_label2name': object_label2name,
    'predicate2predicate_label': predicate2predicate_label, 'predicate_label2predicate': predicate_label2predicate,
}
with open('../raw/postprocessed_scene_graphs.json', 'w') as outfile:
    json.dump(postprocessed_scene_graphs, outfile)
