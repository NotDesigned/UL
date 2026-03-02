# 单卡训练（使用 small 预设，适合 AFHQ 128px）
# output_dir 默认为 ./runs/stage{N}，自动创建 run_0001, run_0002, ...
python train.py --stage 1 --preset small --data_root ~/datasets/afhq \
  --flat_data \
  --wandb --wandb_project ul \
  --save_every 1000 \
  --viz_every 500 --log_every 100 \
  --grad_ckpt

# 恢复训练（自动使用检查点所在目录）
# python train.py --stage 1 --preset small --data_root ~/datasets/afhq \
#   --flat_data --grad_ckpt \
#   --resume ./runs/stage1/run_0001/ckpt_0010000.pt

# 多卡训练（使用 small 预设，可覆盖 batch_size）
# torchrun --nproc_per_node=4 train.py --stage 1 --preset small 1\
#   --data_root ~/datasets/afhq \
#   --batch_size 16 \
#   --grad_accum 4 \
#   --flat_data \
#   --wandb --wandb_project ul \
#   --save_every 1000 \
#   --viz_every 100 --log_every 100
