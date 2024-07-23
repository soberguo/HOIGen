

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
|- HOIGen
|   |- hicodet
|   |   |- hico_20160224_det
|   |       |- annotations
|   |       |- images
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
|- HOIGen
|   |- checkpoints
|   |   |- pretrained_clip
|   |       |- ViT-B-16.pt
:   :      
```

4. Download the weights of DETR and put them in `checkpoints/`.


| Dataset | DETR weights |
| --- | --- |
| HICO-DET | [weights](https://drive.google.com/file/d/1BQ-0tbSH7UC6QMIMMgdbNpRw2NcO8yAD/view?usp=sharing)  |



```
|- HOIGen
|   |- checkpoints
|   |   |- detr-r50-hicodet.pth
:   :   :
```

## Pre-extracted Features
Download the pre-extracted features from [HERE](https://drive.google.com/file/d/1lUnUQD3XcWyQdwDHMi74oXBcivibGIWN/view?usp=sharing). The downloaded files have to be placed as follows.

```
|- HOIGen
|   |- hicodet_pkl_files
|   |   |- union_embeddings_cachemodel_crop_padding_zeros_vitb16.p
:   :      
```

## Training and Testing
### HICO-DET
#### Fully-supervised:
```
python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico --use_insadapter --num_classes 117 --use_multi_hot --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt 
```
```
python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico --use_insadapter --num_classes 117 --use_multi_hot --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --eval --resume CKPT_PATH
```

#### UC:
```
python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico --use_insadapter --num_classes 117 --use_multi_hot --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --zs --zs_type uc0/uc1/uc2/uc3/uc4 --eval --resume CKPT_PATH
```
```
python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico --use_insadapter --num_classes 117 --use_multi_hot --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --zs --zs_type uc0/uc1/uc2/uc3/uc4 --eval --resume CKPT_PATH
```
#### RF-UC:
```
python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico --use_insadapter --num_classes 117 --use_multi_hot --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --zs --zs_type rare_first --eval --resume CKPT_PATH
```
```
python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico --use_insadapter --num_classes 117 --use_multi_hot --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --zs --zs_type rare_first --eval --resume CKPT_PATH
```
#### NF-UC:
```
python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico --use_insadapter --num_classes 117 --use_multi_hot --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --zs --zs_type non_rare_first --eval --resume CKPT_PATH
```
```
python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico --use_insadapter --num_classes 117 --use_multi_hot --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --zs --zs_type non_rare_first --eval --resume CKPT_PATH
```
#### UV:
```
python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico --use_insadapter --num_classes 117 --use_multi_hot --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --zs --zs_type unseen_verb --eval --resume CKPT_PATH
```
```
python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico --use_insadapter --num_classes 117 --use_multi_hot --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --zs --zs_type unseen_verb --eval --resume CKPT_PATH
```
#### UO:
```
python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico --use_insadapter --num_classes 117 --use_multi_hot --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --zs --zs_type unseen_object --eval --resume CKPT_PATH
```
```
python main_tip_finetune.py --world-size 1 --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/hico --use_insadapter --num_classes 117 --use_multi_hot --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --zs --zs_type unseen_object --eval --resume CKPT_PATH
```
## Model Zoo
| Setting | Full | Seen | Unseen | Weights |
| --- | --- |--- |--- |--- |
| UC | 33.44  | 34.23 | 30.26 | [weights](https://drive.google.com/file/d/1UA9rzFFxNkuhUqvTGGrCJ5xpRYw-H-Ei/view?usp=sharing)|
| RF-UC | 33.86  | 34.57 | 31.01 |[weights](https://drive.google.com/file/d/1UA9rzFFxNkuhUqvTGGrCJ5xpRYw-H-Ei/view?usp=sharing)|
| NF-UC | 33.08  | 32.86 | 33.98 |[weights](https://drive.google.com/file/d/1UA9rzFFxNkuhUqvTGGrCJ5xpRYw-H-Ei/view?usp=sharing)|
| UO | 33.48  | 32.90 | 36.35 |[weights](https://drive.google.com/file/d/1UA9rzFFxNkuhUqvTGGrCJ5xpRYw-H-Ei/view?usp=sharing)|
| UV | 32.34  | 34.31 | 20.27 |[weights](https://drive.google.com/file/d/1UA9rzFFxNkuhUqvTGGrCJ5xpRYw-H-Ei/view?usp=sharing)|

## Citation
If you find our paper and/or code helpful, please consider citing:


## Acknowledgement
We gratefully thank the authors from [UPT](https://github.com/fredzzhang/upt), [ADA-CM](https://github.com/ltttpku/ADA-CM/tree/main), [SHIP](https://github.com/mrflogs/SHIP) and [CaFo](https://github.com/OpenGVLab/CaFo) for open-sourcing their code.









<!-- MARKDOWN 链接 & 图片 -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[forks-shield]: https://img.shields.io/github/forks/soberguo/HOIGen.svg?style=for-the-badge
[forks-url]: https://github.com/soberguo/HOIGen/network/members
[stars-shield]: https://img.shields.io/github/stars/soberguo/HOIGen.svg?style=for-the-badge
[stars-url]: https://github.com/soberguo/HOIGen/stargazers
[issues-shield]: https://img.shields.io/github/issues/soberguo/HOIGen.svg?style=for-the-badge
[issues-url]: https://github.com/soberguo/HOIGen/issues
[product-screenshot]: images/fig1.png


