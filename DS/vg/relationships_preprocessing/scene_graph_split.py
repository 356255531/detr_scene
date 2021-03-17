from random import shuffle
import json

scene_graphs = json.loads(open('../annotations/raw/postprocessed_scene_graphs.json').read())

images = scene_graphs['images']
annotations = scene_graphs['annotations']

ids = [image['id'] for image in images]
ids2image = {image['id']: image for image in images}
ids2annotation = {annotation['image_id']: annotation for annotation in annotations}

shuffle(ids)

train_ids = ids[:int(len(ids) * 0.7)]
test_ids = ids[int(len(ids) * 0.7):]
val_ids = train_ids[-5000:]
train_ids = train_ids[:-5000]

train_ids = sorted(train_ids, reverse=True)
val_ids = sorted(val_ids, reverse=True)
test_ids = sorted(test_ids, reverse=True)

print('train size: {}'.format(len(train_ids)))
print('val size: {}'.format(len(val_ids)))
print('test size: {}'.format(len(test_ids)))

train_images = [ids2image[id] for id in train_ids]
val_images = [ids2image[id] for id in val_ids]
test_images = [ids2image[id] for id in test_ids]

train_annotations = [ids2annotation[id] for id in train_ids]
val_annotations = [ids2annotation[id] for id in val_ids]
test_annotations = [ids2annotation[id] for id in test_ids]

train_scene_graph = {'images': train_images, 'annotations': train_annotations}
val_scene_graph = {'images': val_images, 'annotations': val_annotations}
test_scene_graph = {'images': test_images, 'annotations': test_annotations}

with open('../annotations/scene_graphs_train.json', 'w') as outfile:
    json.dump(train_scene_graph, outfile)
with open('../annotations/scene_graphs_val.json', 'w') as outfile:
    json.dump(val_scene_graph, outfile)
with open('../annotations/scene_graphs_test.json', 'w') as outfile:
    json.dump(test_scene_graph, outfile)
