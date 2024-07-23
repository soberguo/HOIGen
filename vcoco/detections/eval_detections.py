"""
Evaluate generated object detections on V-COCO

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""
import os
import sys
import json
import torch
import argparse
import numpy as np

from tqdm import tqdm
from torchvision.ops.boxes import batched_nms

from pocket.ops import to_tensor
from pocket.utils import BoxAssociation, DetectionAPMeter

sys.path.append('..')
from vcoco import VCOCO

def compute_map(
        dataset, detection_dir,
        h_thresh, o_thresh, nms_thresh,
        max_human, max_object,
        human_idx=1, min_iou=0.5
    ):
    num_pairs_object = torch.zeros(81)
    associate = BoxAssociation(min_iou=min_iou)
    meter = DetectionAPMeter(
        81, algorithm='INT', nproc=10
    )
    # Skip images without valid human-object pairs
    valid_idx = dataset._keep

    for i in tqdm(valid_idx):
        # Load annotation
        annotation = dataset.annotations[i]
        image_name = annotation.pop('file_name')
        target = to_tensor(annotation, input_format='dict')
        # Load detection
        detection_path = os.path.join(
            detection_dir,
            image_name.replace('jpg', 'json')
        )
        with open(detection_path, 'r') as f:
            detection = to_tensor(json.load(f), input_format='dict')

        boxes = detection['boxes']
        labels = detection['labels']
        scores = detection['scores']        
        
        # Filter out low scoring human boxes
        idx = torch.nonzero(labels == human_idx).squeeze(1)
        keep_idx = idx[torch.nonzero(scores[idx] >= h_thresh).squeeze(1)]

        # Filter out low scoring object boxes
        idx = torch.nonzero(labels != human_idx).squeeze(1)
        keep_idx = torch.cat([
            keep_idx,
            idx[torch.nonzero(scores[idx] >= o_thresh).squeeze(1)]
        ])

        boxes = boxes[keep_idx].view(-1, 4)
        scores = scores[keep_idx].view(-1)
        labels = labels[keep_idx].view(-1)

        # Class-wise non-maximum suppression
        keep_idx = batched_nms(
            boxes, scores, labels, nms_thresh
        )
        boxes = boxes[keep_idx].view(-1, 4)
        scores = scores[keep_idx].view(-1)
        labels = labels[keep_idx].view(-1)

        sorted_idx = torch.argsort(scores, descending=True)
        boxes = boxes[sorted_idx]
        scores = scores[sorted_idx]
        labels = labels[sorted_idx]

        h_idx = torch.nonzero(labels == human_idx).squeeze(1)
        o_idx = torch.nonzero(labels != human_idx).squeeze(1)
        if len(h_idx) > max_human:
            h_idx = h_idx[:max_human]
        if len(o_idx) > max_object:
            o_idx = o_idx[:max_object]
        keep_idx = torch.cat([h_idx, o_idx])

        boxes = boxes[keep_idx].view(-1, 4)
        scores = scores[keep_idx].view(-1)
        labels = labels[keep_idx].view(-1)

        # Format ground truth boxes
        gt_boxes = torch.cat([
            target['boxes_h'],
            target['boxes_o']
        ])
        gt_classes = torch.cat([
            human_idx * torch.ones_like(target['objects']),
            target['objects']
        ])
        # Remove duplicates
        _, keep = np.unique(gt_boxes, return_index=True, axis=0)
        keep = torch.from_numpy(keep)
        gt_boxes = gt_boxes[keep]
        gt_classes = gt_classes[keep]
        # Update number of ground truth annotations
        for c in gt_classes:
            num_pairs_object[c] += 1

        # Associate detections with ground truth
        binary_labels = torch.zeros_like(scores)
        unqiue_obj = labels.unique()
        for obj_idx in unqiue_obj:
            det_idx = torch.nonzero(labels == obj_idx).squeeze(1)
            gt_idx = torch.nonzero(gt_classes == obj_idx).squeeze(1)
            if len(gt_idx) == 0:
                continue
            binary_labels[det_idx] = associate(
                gt_boxes[gt_idx].view(-1, 4),
                boxes[det_idx].view(-1, 4),
                scores[det_idx].view(-1)
            )

        meter.append(scores, labels, binary_labels)

    meter.num_gt = num_pairs_object.tolist()
    ap = meter.eval()
    object_keep = dataset.present_objects
    ap_present = ap[object_keep]
    rec_present = meter.max_rec[object_keep]
    print(
        "Mean average precision: {:.4f} |".format(ap_present.mean().item()),
        "Mean maximum recall: {:.4f}".format(rec_present.mean().item())
    )

def main(args):
    
    image_dir = dict(
        train='mscoco2014/train2014',
        val='mscoco2014/train2014',
        trainval='mscoco2014/train2014',
        test='mscoco2014/val2014'
    )
    dataset = VCOCO(
        root=os.path.join(args.data_root, image_dir[args.partition]),
        anno_file=os.path.join(
            args.data_root,
            'instances_vcoco_{}.json'.format(args.partition)
    ))

    h_score_thresh = args.human_thresh
    o_score_thresh = args.object_thresh
    nms_thresh = args.nms_thresh
    max_human = args.max_human
    max_object = args.max_object

    compute_map(
        dataset, args.detection_root,
        h_score_thresh, o_score_thresh, nms_thresh,
        max_human, max_object
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection-root', required=True, type=str)
    parser.add_argument('--partition', type=str, default='test')
    parser.add_argument('--data-root', type=str, default='../')
    parser.add_argument('--human-thresh', default=0.05, type=float,
                        help="Threshold used to filter low scoring human detections")
    parser.add_argument('--max-human', default=50, type=int,
                        help="Maximum number of human instances to keep in an image")
    parser.add_argument('--object-thresh', default=0.05, type=float,
                        help="Threshold used to filter low scoring object detections")
    parser.add_argument('--max-object', default=50, type=int,
                        help="Maximum number of (pure) object instances to keep in an image")
    parser.add_argument('--nms-thresh', default=0.5, type=float,
                        help="Threshold for non-maximum suppression")
    args = parser.parse_args()

    print(args)
    main(args)
