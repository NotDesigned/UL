# 单卡训练
python train.py --stage 1 --data_root ~/datasets/afhq \
  --output_dir ./runs/stage1 --resolution 128 \
  --latent_size 8 \
  --latent_channels 32 \
  --enc_channels 64,128,256,512 \
  --dec_channels 64,128,256 \
  --embed_dim 512 \
  --vit_blocks 6 \
  --vit_heads 8 \
  --batch_size 32 \
  --lr 1e-4 \
  --mixed_precision \
  --flat_data \
  --wandb --wandb_project ul \
  --save_every 1000 \
  --viz_every 100 --log_every 100 \
  --resume runs/stage1/ckpt_0035000.pt

# 多卡训练（带梯度累积和 WandB）
# torchrun --nproc_per_node=4 train.py --stage 1 --data_root ~/datasets/afhq \
#   --output_dir ./runs/stage1 --resolution 128 \
#   --latent_size 8 \
#   --latent_channels 32 \
#   --enc_channels 64,128,256,512 \
#   --dec_channels 64,128,256 \
#   --embed_dim 512 \
#   --vit_blocks 6 \
#   --vit_heads 8 \
#   --batch_size 16 \
#   --grad_accum 4 \
#   --lr 1e-4 \
#   --mixed_precision \
#   --flat_data \
#   --wandb --wandb_project ul \
#   --save_every 1000 \
#   --viz_every 100 --log_every 100
