import random
import json


random.seed(2021)

print('Loading dataset...')
scene_graphs = json.loads(open('../raw/postprocessed_scene_graphs.json').read())

images = scene_graphs['images']
annotations = scene_graphs['annotations']

image_ids = [image['image_id'] for image in images]
image_id2image = {image['image_id']: image for image in images}
image_id2annotation = {annotation['image_id']: annotation for annotation in annotations}

random.shuffle(image_ids)

train_image_ids = image_ids[:int(len(image_ids) * 0.7)]
test_image_ids = image_ids[int(len(image_ids) * 0.7):]
val_image_ids = train_image_ids[-5000:]
train_image_ids = train_image_ids[:-5000]

train_image_ids = sorted(train_image_ids, reverse=False)
val_image_ids = sorted(val_image_ids, reverse=False)
test_image_ids = sorted(test_image_ids, reverse=False)

print('train size: {}'.format(len(train_image_ids)))
print('val size: {}'.format(len(val_image_ids)))
print('test size: {}'.format(len(test_image_ids)))

train_images = [image_id2image[train_image_id] for train_image_id in train_image_ids]
val_images = [image_id2image[val_image_ids] for val_image_ids in val_image_ids]
test_images = [image_id2image[test_image_id] for test_image_id in test_image_ids]

train_annotations = [image_id2annotation[train_image_id] for train_image_id in train_image_ids]
val_annotations = [image_id2annotation[val_image_ids] for val_image_ids in val_image_ids]
test_annotations = [image_id2annotation[test_image_id] for test_image_id in test_image_ids]

train_scene_graph = {'images': train_images, 'annotations': train_annotations}
val_scene_graph = {'images': val_images, 'annotations': val_annotations}
test_scene_graph = {'images': test_images, 'annotations': test_annotations}

result = [0,0,0,0,0]
for id in range(len(train_annotations)):
    if train_annotations[id]['image_id'] == 2353450:
        result[0] = id
    if train_annotations[id]['image_id'] == 2351427:
        result[1] = id
    if train_annotations[id]['image_id'] == 2343148:
        result[2] = id
    if train_annotations[id]['image_id'] == 2343428:
        result[3] = id
    if train_annotations[id]['image_id'] == 2364223:
        result[4] = id
import pdb
pdb.set_trace()
with open('../annotations/scene_graphs_train.json', 'w') as outfile:
    json.dump(train_scene_graph, outfile)
with open('../annotations/scene_graphs_val.json', 'w') as outfile:
    json.dump(val_scene_graph, outfile)
with open('../annotations/scene_graphs_test.json', 'w') as outfile:
    json.dump(test_scene_graph, outfile)
