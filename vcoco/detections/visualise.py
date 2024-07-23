
import os
import sys
import json
import argparse
import numpy as np

import torch
from PIL import ImageDraw
from torchvision.ops import nms
from PIL import Image, ImageDraw, ImageFont

sys.path.append('..')
from vcoco import VCOCO
from vcoco_list import vcoco_values

object_name = ['bg', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def visualize(args):
    image_dir = dict(
        train='images/train2014',
        val='images/train2014',
        trainval='images/train2014',
        test='images/val2014'
    )
    dataset = VCOCO(
        root=os.path.join(args.data_root, image_dir[args.partition]),
        anno_file=os.path.join(
            args.data_root,
            'instances_vcoco_{}.json'.format(args.partition)
        ))
    # image, _ = dataset[args.image_idx]
    image = Image.open(args.image_idx)
    # image_name = dataset.filename(args.image_idx)
    image_name = args.image_idx.split('/')[-1]
    detection_path = os.path.join('./vcoco/detections/trainval_gt_vcoco',
                                  image_name.replace('.jpg', '.json')
                                  )
    with open(detection_path, 'r') as f:
        detections = json.load(f)
    # Remove low-scoring boxes
    box_score_thresh = args.box_score_thresh
    boxes = np.asarray(detections['boxes'])
    scores = np.asarray(detections['scores'])
    labels = np.asarray(detections['labels'])
    hois = np.asarray(detections['hois'])
    keep_idx = np.where(scores >= box_score_thresh)[0]
    boxes = boxes[keep_idx, :]
    scores = scores[keep_idx]
    # Perform NMS
    keep_idx = nms(
        torch.from_numpy(boxes),
        torch.from_numpy(scores),
        args.nms_thresh
    )
    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    labels = labels[keep_idx]
    font_size = 20
    font = ImageFont.truetype("arial.ttf", font_size)
    # Draw boxes
    canvas = ImageDraw.Draw(image)
    for idx in range(boxes.shape[0]):
        coords = boxes[idx, :].tolist()
        canvas.rectangle(coords)
        # canvas.text(coords[:2], str(scores[idx])[:4])
        canvas.text(coords[:2] + [10, 5], object_name[labels[idx]], font=font)
    for idx, i in enumerate(hois):
        hoi_name = vcoco_values[i][0] + ' ' + vcoco_values[i][1]
        canvas.text((10.0, 20 * idx + 10.0), hoi_name, fill=(255, 0, 0), font=font)
    image.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-idx',
                        default='./COCO_train2014_000000000010.jpg')
    parser.add_argument('--data-root', type=str, default='E:/study/boshi_study/hoi/datasets/v-coco')
    parser.add_argument('--partition', type=str, default='trainval')
    parser.add_argument('--box-score-thresh', type=float, default=0.2)
    parser.add_argument('--nms-thresh', type=float, default=0.5)
    args = parser.parse_args()

    visualize(args)
