import json
import random
import itertools
from PIL import Image, ImageDraw

print('Loading datasets...')
scene_graphs = json.loads(open('../raw/postprocessed_scene_graphs.json').read())

annotations = scene_graphs['annotations']
object_label2name = scene_graphs['object_label2name']

selected_annotations = random.choices(annotations, k=10)

with open('obj_visual_postprocessed/annoataion.txt', 'w') as f:
    for anno_idx, annotation in enumerate(selected_annotations):
        file_path = '../images/' + str(annotation['image_id']) + '.jpg'

        bbox = annotation['bbox']
        names = [object_label2name[str(label)] for label in annotation['object_labels']]

        for box_rel, box in enumerate(bbox):
            img = Image.open(file_path).convert('RGB')
            img_draw = ImageDraw.Draw(img)

            x, y, w, h = box
            img_draw.rectangle([(x, y), (x + w, y + h)], outline="red")
            img.save('obj_visual_postprocessed/{}{}.jpg'.format(anno_idx, box_rel))

            f.write("{}{}: {}".format(anno_idx, box_rel, names[box_rel]))

