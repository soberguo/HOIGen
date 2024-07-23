#-------------non rare first ----------------------------
CUDA_VISIBLE_DEVICES=0 python main_tip_finetune.py \
    --batch-size 4 \
    --world-size 1 \
    --pretrained checkpoints/detr-r50-hicodet.pth \
    --output-dir checkpoints/non_rare_first1 \
    --use_insadapter \
    --num_classes 117 \
    --use_multi_hot \
    --file1 hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p \
    --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt \
    --epoch 15 \
    --num_shot 1 \
    --zs \
    --zs_type non_rare_first \


