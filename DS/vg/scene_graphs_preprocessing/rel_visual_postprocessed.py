import json
import random
import itertools
from PIL import Image, ImageDraw

print('Loading datasets...')
scene_graphs = json.loads(open('../raw/postprocessed_scene_graphs.json').read())

annotations = scene_graphs['annotations']
object_label2name = scene_graphs['object_label2name']
predicate_label2predicate = scene_graphs['predicate_label2predicate']

selected_annotations = random.choices(annotations, k=10)
with open('rel_visual_postprocessed/annoataion.txt', 'w') as f:
    for anno_idx, annotation in enumerate(selected_annotations):
        file_path = '../images/' + str(annotation['image_id']) + '.jpg'

        bbox = annotation['bbox']
        names = [object_label2name[str(label)] for label in annotation['object_labels']]
        predicates = [predicate_label2predicate[str(label)] for label in annotation['predicate_labels']]
        for rel_idx, rel in enumerate(annotation['relationships']):
            img = Image.open(file_path).convert('RGB')
            img_draw = ImageDraw.Draw(img)

            i, j = rel
            x, y, w, h = bbox[i]
            img_draw.rectangle([(x, y), (x + w, y + h)], outline="red")
            x, y, w, h = bbox[j]
            img_draw.rectangle([(x, y), (x + w, y + h)], outline="blue")
            img.save('rel_visual_postprocessed/{}{}.jpg'.format(anno_idx, rel_idx))
            f.write("{}{}: {}-{}-{}".format(anno_idx, rel_idx, names[i], predicates[rel_idx], names[j]))

