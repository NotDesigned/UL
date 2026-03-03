export OMP_NUM_THREADS=2
NPROC_PER_NODE=8
# 单卡训练（使用 small 预设）
# output_dir 默认为 ./runs/stage{N}，自动创建 run_0001, run_0002, ...
torchrun --nproc_per_node=$NPROC_PER_NODE train.py --stage 1 --preset large --data_root /data/datasets/mini-imagenet \
  --wandb --wandb_project ul \
  --effective_batch_size 512 \
  --save_every 1000 \
  --viz_every 1000 --log_every 100 \
#  --grad_ckpt

# 恢复训练（自动使用检查点所在目录）
# torchrun --nproc_per_node=$NPROC_PER_NODE --preset small --data_root ~/data/mini-imagenet \
#   --grad_ckpt \
#   --resume ./runs/stage1/run_0001/ckpt_0010000.pt

# 阶段二训练

# torchrun --nproc_per_node=$NPROC_PER_NODE train.py --stage 2 --preset small --data_root /data/datasets/mini-imagenet \
#   --wandb --wandb_project ul \
#   --effective_batch_size 512
#   --save_every 1000 \
#   --viz_every 500 --log_every 100 \
#   --grad_ckpt \
#   --stage1_ckpt ./runs/stage1/run_0001/ckpt_final.pt