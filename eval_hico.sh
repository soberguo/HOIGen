CUDA_VISIBLE_DEVICES=1 python main_tip_finetune.py \
    --world-size 1 \
    --pretrained checkpoints/detr-r50-hicodet.pth \
    --output-dir checkpoints/new_best \
    --use_insadapter \
    --num_classes 117 \
    --use_multi_hot \
    --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p \
    --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt \
    --batch-size 4 \
    --data-root /home/guoyixin/datasets/ \
    --epochs 15 \
    --zs \
    --zs_type non_rare_first \
    --eval \
    --resume 'checkpoints/new_best/non_rare_first.pt'

CUDA_VISIBLE_DEVICES=1 python main_tip_finetune.py \
    --world-size 1 \
    --pretrained checkpoints/detr-r50-hicodet.pth \
    --output-dir checkpoints/new_best \
    --use_insadapter \
    --num_classes 117 \
    --use_multi_hot \
    --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p \
    --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt \
    --batch-size 4 \
    --data-root /home/guoyixin/datasets/ \
    --epochs 15 \
    --zs \
    --zs_type rare_first \
    --eval \
    --resume 'checkpoints/new_best/rare_first.pt'



CUDA_VISIBLE_DEVICES=1 python main_tip_finetune.py \
    --world-size 1 \
    --pretrained checkpoints/detr-r50-hicodet.pth \
    --output-dir checkpoints/new_best \
    --use_insadapter \
    --num_classes 117 \
    --use_multi_hot \
    --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p \
    --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt \
    --batch-size 4 \
    --data-root /home/guoyixin/datasets/ \
    --epochs 15 \
    --zs \
    --zs_type uc4 \
    --eval \
    --resume 'checkpoints/new_best/uc4.pt'

CUDA_VISIBLE_DEVICES=1 python main_tip_finetune.py \
    --world-size 1 \
    --pretrained checkpoints/detr-r50-hicodet.pth \
    --output-dir checkpoints/new_best \
    --use_insadapter \
    --num_classes 117 \
    --use_multi_hot \
    --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p \
    --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt \
    --batch-size 4 \
    --data-root /home/guoyixin/datasets/ \
    --epochs 15 \
    --zs \
    --zs_type uc1 \
    --eval \
    --resume 'checkpoints/new_best/uc1.pt'

CUDA_VISIBLE_DEVICES=1 python main_tip_finetune.py \
    --world-size 1 \
    --pretrained checkpoints/detr-r50-hicodet.pth \
    --output-dir checkpoints/new_best \
    --use_insadapter \
    --num_classes 117 \
    --use_multi_hot \
    --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p \
    --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt \
    --batch-size 4 \
    --data-root /home/guoyixin/datasets/ \
    --epochs 15 \
    --zs \
    --zs_type uc2 \
    --eval \
    --resume 'checkpoints/new_best/uc2.pt'


CUDA_VISIBLE_DEVICES=1 python main_tip_finetune.py \
    --world-size 1 \
    --pretrained checkpoints/detr-r50-hicodet.pth \
    --output-dir checkpoints/new_best \
    --use_insadapter \
    --num_classes 117 \
    --use_multi_hot \
    --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p \
    --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt \
    --batch-size 4 \
    --data-root /home/guoyixin/datasets/ \
    --epochs 15 \
    --zs \
    --zs_type uc3 \
    --eval \
    --resume 'checkpoints/new_best/uc3.pt'


CUDA_VISIBLE_DEVICES=1 python main_tip_finetune.py \
    --world-size 1 \
    --pretrained checkpoints/detr-r50-hicodet.pth \
    --output-dir checkpoints/new_best \
    --use_insadapter \
    --num_classes 117 \
    --use_multi_hot \
    --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p \
    --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt \
    --batch-size 4 \
    --data-root /home/guoyixin/datasets/ \
    --epochs 15 \
    --zs \
    --zs_type uc0 \
    --eval \
    --resume 'checkpoints/new_best/uc0.pt'



CUDA_VISIBLE_DEVICES=1 python main_tip_finetune.py \
    --world-size 1 \
    --pretrained checkpoints/detr-r50-hicodet.pth \
    --output-dir checkpoints/new_best \
    --use_insadapter \
    --num_classes 117 \
    --use_multi_hot \
    --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p \
    --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt \
    --batch-size 4 \
    --data-root /home/guoyixin/datasets/ \
    --epochs 15 \
    --zs \
    --zs_type unseen_object \
    --eval \
    --resume 'checkpoints/new_best/object.pt'

CUDA_VISIBLE_DEVICES=1 python main_tip_finetune.py \
    --world-size 1 \
    --pretrained checkpoints/detr-r50-hicodet.pth \
    --output-dir checkpoints/new_best \
    --use_insadapter \
    --num_classes 117 \
    --use_multi_hot \
    --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p \
    --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt \
    --batch-size 4 \
    --data-root /home/guoyixin/datasets/ \
    --epochs 15 \
    --zs \
    --zs_type unseen_verb \
    --eval \
    --resume 'checkpoints/new_best/verb.pt'

CUDA_VISIBLE_DEVICES=1 python main_tip_finetune.py \
    --world-size 1 \
    --pretrained checkpoints/detr-r50-hicodet.pth \
    --output-dir checkpoints/new_best \
    --use_insadapter \
    --num_classes 117 \
    --use_multi_hot \
    --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p \
    --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt \
    --batch-size 4 \
    --data-root /home/guoyixin/datasets/ \
    --epochs 15 \
    --eval \
    --resume 'checkpoints/new_best/no_unseen.pt'
