

[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]


# HOIGen
Official code of ACM MM2024 paper- Unseen No More: Unlocking the Potential of CLIP for Generative Zero-shot HOI Detection.
![产品截图][product-screenshot]

## Dataset 
Follow the process of [UPT](https://github.com/fredzzhang/upt).

The downloaded files should be placed as follows. Otherwise, please replace the default path to your custom locations.
```
|- ADA-CM
|   |- hicodet
|   |   |- hico_20160224_det
|   |       |- annotations
|   |       |- images
|   |- vcoco
|   |   |- mscoco2014
|   |       |- train2014
|   |       |-val2014
:   :      
```

## Dependencies
1. Follow the environment setup in [UPT](https://github.com/fredzzhang/upt).

2. Our code is built upon [CLIP](https://github.com/openai/CLIP). Install the local package of CLIP:
```
cd CLIP && python setup.py develop && cd ..
```

3. Download the CLIP weights to `checkpoints/pretrained_clip`.
```
|- ADA-CM
|   |- checkpoints
|   |   |- pretrained_clip
|   |       |- ViT-B-16.pt
|   |       |- ViT-L-14-336px.pt
:   :      
```

4. Download the weights of DETR and put them in `checkpoints/`.


| Dataset | DETR weights |
| --- | --- |
| HICO-DET | [weights](https://drive.google.com/file/d/1BQ-0tbSH7UC6QMIMMgdbNpRw2NcO8yAD/view?usp=sharing)  |
| V-COCO | [weights](https://drive.google.com/file/d/1AIqc2LBkucBAAb_ebK9RjyNS5WmnA4HV/view?usp=sharing) |


```
|- ADA-CM
|   |- checkpoints
|   |   |- detr-r50-hicodet.pth
|   |   |- detr-r50-vcoco.pth
:   :   :
```

## Pre-extracted Features
Download the pre-extracted features from [HERE](https://drive.google.com/file/d/1lUnUQD3XcWyQdwDHMi74oXBcivibGIWN/view?usp=sharing) and the pre-extracted bboxes from [HERE](https://drive.google.com/file/d/19Mo1d4J6xX9jDNvDJHEWDpaiPKxQHQsT/view?usp=sharing). The downloaded files have to be placed as follows.

```
|- ADA-CM
|   |- hicodet_pkl_files
|   |   |- union_embeddings_cachemodel_crop_padding_zeros_vitb16.p
|   |   |- hicodet_union_embeddings_cachemodel_crop_padding_zeros_vit336.p
|   |   |- hicodet_train_bbox_R50.p
|   |   |- hicodet_test_bbox_R50.p
|   |- vcoco_pkl_files
|   |   |- vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit16.p
|   |   |- vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit336.p
|   |   |- vcoco_train_bbox_R50.p
|   |   |- vcoco_test_bbox_R50.p
:   :      
```

## FineTuning Mode
### HICO-DET
#### Train on HICO-DET:
```
python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico --use_insadapter --num_classes 117 --use_multi_hot --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt 
```

#### Test on HICO-DET:
```
python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico --use_insadapter --num_classes 117 --use_multi_hot --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --eval --resume CKPT_PATH
```


### V-COCO
#### Training on V-COCO
```
python main_tip_finetune.py --world-size 1 --dataset vcoco --data-root vcoco/ --partitions trainval test --pretrained checkpoints/detr-r50-vcoco.pth --output-dir checkpoints/vcoco-injector-r50 --use_insadapter --num_classes 24 --use_multi_hot --file1 vcoco_pkl_files/vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit16.p  --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt
```

#### Cache detection results for evaluation on V-COCO
```
python main_tip_finetune.py --world-size 1 --dataset vcoco --data-root vcoco/ --partitions trainval test --pretrained checkpoints/detr-r50-vcoco.pth --output-dir checkpoints/vcoco-injector-r50 --use_insadapter --num_classes 24 --use_multi_hot --file1 vcoco_pkl_files/vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit16.p  --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --cache --resume CKPT_PATH
```






<!-- MARKDOWN 链接 & 图片 -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[forks-shield]: https://img.shields.io/github/forks/soberguo/HOIGen.svg?style=for-the-badge
[forks-url]: https://github.com/soberguo/HOIGen/network/members
[stars-shield]: https://img.shields.io/github/stars/soberguo/HOIGen.svg?style=for-the-badge
[stars-url]: https://github.com/soberguo/HOIGen/stargazers
[issues-shield]: https://img.shields.io/github/issues/soberguo/HOIGen.svg?style=for-the-badge
[issues-url]: https://github.com/soberguo/HOIGen/issues
[product-screenshot]: images/fig1.png


