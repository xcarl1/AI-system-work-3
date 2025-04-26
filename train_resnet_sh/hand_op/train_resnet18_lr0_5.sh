export CUDA_VISIBLE_DEVICES=1
python /root/siton-data-412581749c3f4cfea0d7c972b8742057/proj/work_3_xzp/train_ds.py \
--num_classes 37 \
--learning_rate 0.0005 \
--exp_name "resnet18_lr_0_5" \
--save_dir "./checkpoints/resnet18_lr_0_5" \