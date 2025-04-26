export CUDA_VISIBLE_DEVICES=1
python /root/siton-data-412581749c3f4cfea0d7c972b8742057/proj/work_3_xzp/train_ds.py \
--num_classes 37 \
--learning_rate 0.0003 \
--exp_name "resnet34_lr_0_3" \
--save_dir "./checkpoints/resnet34_lr_0_3" \
--model_type "resnet34" \