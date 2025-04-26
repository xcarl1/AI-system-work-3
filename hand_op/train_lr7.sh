export CUDA_VISIBLE_DEVICES=0
python /root/siton-data-412581749c3f4cfea0d7c972b8742057/proj/work_3_xzp/train_ds.py \
--num_classes 37 \
--patch_size 16 \
--epochs 100 \
--num_layers 4 \
--learning_rate 0.7 \
--exp_name "vit_lr_7" \
--save_dir "./checkpoints/vit_lr_7" \