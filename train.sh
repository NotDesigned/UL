export OMP_NUM_THREADS=2
NPROC_PER_NODE=1
DATA_ROOT=~/datasets/mini-imagenet
PRESET=small

# === TRAIN ====

LOG_EVERY=100
SAVE_EVERY=1000
VIZ_EVERY=1000
TOTAL_STEPS_1=1000
TOTAL_STEPS_2=1000

# === EVAL ====

NUM_REAL=50000
NUM_GEN=50000
BATCH_SIZE=64
LATENT_STEPS=100
DECODE_STEPS=100
SAMPLER="ddim"
OUTPUT_DIR=./eval_output

# ===== 阶段一：编码器 + 先验 + 解码器联合训练 =====
# output_dir 默认为 ./runs/stage1，自动创建 run_0001, run_0002, ...
torchrun --nproc_per_node=$NPROC_PER_NODE train.py --stage 1 --preset $PRESET --data_root $DATA_ROOT \
  --wandb --wandb_project ul \
  --effective_batch_size 64 \
  --save_every $SAVE_EVERY \
  --viz_every $VIZ_EVERY --log_every $LOG_EVERY \
  --total_steps $TOTAL_STEPS_1 \
  --grad_ckpt

# ===== 阶段二：冻结编码器/解码器，训练 BaseModel =====
# 自动查找阶段一最新的 run 目录
STAGE1_RUN=$(ls -d ./runs/stage1/run_* 2>/dev/null | sort | tail -1)
STAGE1_CKPT="${STAGE1_RUN}/ckpt_final.pt"

torchrun --nproc_per_node=$NPROC_PER_NODE train.py --stage 2 --preset $PRESET --data_root $DATA_ROOT \
  --wandb --wandb_project ul \
  --effective_batch_size 64 \
  --save_every $SAVE_EVERY \
  --viz_every $VIZ_EVERY --log_every $LOG_EVERY \
  --stage1_ckpt "$STAGE1_CKPT" \
  --total_steps $TOTAL_STEPS_2 \
  --grad_ckpt

# ===== 评测：gFID / rFID / PSNR =====
STAGE2_RUN=$(ls -d ./runs/stage2/run_* 2>/dev/null | sort | tail -1)
STAGE2_CKPT="${STAGE2_RUN}/ckpt_final.pt"

python eval.py \
  --stage1_ckpt "$STAGE1_CKPT" \
  --stage2_ckpt "$STAGE2_CKPT" \
  --data_root   $DATA_ROOT \
  --output_dir  $OUTPUT_DIR \
  --n_real      $NUM_REAL \
  --n_gen       $NUM_GEN \
  --batch_size  $BATCH_SIZE \
  --latent_steps $LATENT_STEPS \
  --decode_steps $DECODE_STEPS \
  --sampler $SAMPLER