"""
Visualise ground truth human-object pairs
and save as .png images

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import sys
import argparse
import numpy as np

from tqdm import tqdm
from PIL import ImageDraw

import pocket

sys.path.append('..')
from vcoco import VCOCO

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", required=True, type=str,
                        help="Choose amongst train, val, trainval and test")
    args = parser.parse_args()

    image_root = dict(
        train='../mscoco2014/train2014',
        val='../mscoco2014/train2014',
        trainval='../mscoco2014/train2014',
        test='../mscoco2014/val2014'
    )
    dataset = VCOCO(
        root=image_root[args.partition],
        anno_file='../instances_vcoco_{}.json'.format(args.partition)
    )

    root_cache = os.path.join("./visualisations", args.partition)

    for idx, (image, target) in enumerate(tqdm(dataset)):
        classes = np.asarray(target["actions"])
        unique_cls = np.unique(classes)
        # Visualise by class
        for cls_idx in unique_cls:
            sample_idx = np.where(classes == cls_idx)[0]
            image_ = image.copy()
            canvas = ImageDraw.Draw(image_)
            for i in sample_idx:
                b1 = np.asarray(target["boxes_h"][i])
                b2 = np.asarray(target["boxes_o"][i])

                canvas.rectangle(b1.tolist(), outline='#007CFF', width=5)
                canvas.rectangle(b2.tolist(), outline='#46FF00', width=5)
                b_h_centre = (b1[:2]+b1[2:])/2
                b_o_centre = (b2[:2]+b2[2:])/2
                canvas.line(
                    b_h_centre.tolist() + b_o_centre.tolist(),
                    fill='#FF4444', width=5
                )
                canvas.ellipse(
                    (b_h_centre - 5).tolist() + (b_h_centre + 5).tolist(),
                    fill='#FF4444'
                )
                canvas.ellipse(
                    (b_o_centre - 5).tolist() + (b_o_centre + 5).tolist(),
                    fill='#FF4444'
                )
            cache_dir = os.path.join(root_cache, "class_{:03d}".format(cls_idx))
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            image_.save(os.path.join(
                cache_dir, "{}.png".format(idx)
            ))
