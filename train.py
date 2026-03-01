"""
Unified Latents (UL) - Training
阶段一：编码器 + 先验 + 解码器联合训练
阶段二：冻结编码器/解码器，训练 BaseModel

用法：
  # 阶段一（单卡）
  python train.py --stage 1 --data_root /path/to/imagenet --output_dir ./runs/stage1

  # 阶段一（多卡）
  torchrun --nproc_per_node=4 train.py --stage 1 --data_root /path/to/imagenet --output_dir ./runs/stage1

  # 阶段二（需要先完成阶段一）
  python train.py --stage 2 --data_root /path/to/imagenet \
                  --stage1_ckpt ./runs/stage1/ckpt_final.pt \
                  --output_dir ./runs/stage2
"""

import os
import math
import argparse
import contextlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from dataset import get_dataloader
from models import Encoder, PriorModel, DiffusionDecoder, BaseModel
from sample import make_sample_grid, reconstruct
import torchvision
from torchvision.utils import save_image

from utils import (
    get_latent_schedule, get_image_schedule,
    add_latent_noise, sample_timesteps,
    diffusion_loss, kl_standard_normal,
    loss_weight_unweighted, loss_weight_sigmoid,
)

try:
    import wandb
except ImportError:
    wandb = None


# ============================================================
# 配置
# ============================================================

def get_args():
    p = argparse.ArgumentParser()

    # 基本设置
    p.add_argument('--stage',        type=int,   default=1)
    p.add_argument('--data_root',    type=str,   required=True)
    p.add_argument('--output_dir',   type=str,   default='./runs')
    p.add_argument('--stage1_ckpt',  type=str,   default=None,
                   help='阶段二需要提供阶段一的检查点路径')

    # 数据
    p.add_argument('--resolution',   type=int,   default=512)
    p.add_argument('--batch_size',   type=int,   default=16)
    p.add_argument('--num_workers',  type=int,   default=8)
    p.add_argument('--flat_data',    action='store_true',
                   help='使用 FlatImageDataset（无类别子目录）')

    # 模型
    p.add_argument('--latent_channels', type=int, default=32)
    p.add_argument('--latent_size',     type=int, default=32)
    p.add_argument('--enc_channels',    type=str, default='128,256,512,512',
                   help='Encoder 的各阶段通道数配置')
    p.add_argument('--dec_channels',    type=str, default='128,256,512',
                   help='Decoder 下采样与上采样的卷积通道配置')
    p.add_argument('--base_dims',       type=str, default='512,1024',
                   help='BaseModel 两阶段 ViT 的特征维度')
    p.add_argument('--embed_dim',       type=int, default=1024,
                   help='全局 ViT 的核心注意力维度')
    p.add_argument('--vit_blocks',      type=int, default=8,
                   help='PriorModel 与 Decoder 的 ViT 块数量')
    p.add_argument('--base_blocks',     type=str, default='6,16',
                   help='BaseModel 两阶段的 ViT 块数量')
    p.add_argument('--vit_heads',       type=int, default=16,
                   help='PriorModel 与 Decoder 中 ViT 的多头注意力头数')
    p.add_argument('--base_heads',      type=str, default='8,16',
                   help='BaseModel 两阶段 ViT 的多头注意力头数')

    # 训练超参
    p.add_argument('--total_steps',  type=int,   default=500_000)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--warmup_steps', type=int,   default=5_000)
    p.add_argument('--grad_clip',    type=float, default=1.0)
    p.add_argument('--ema_decay',    type=float, default=0.9999)
    p.add_argument('--grad_accum',   type=int,   default=1,
                   help='梯度累积步数（有效 batch = batch_size × grad_accum × world_size）')

    # UL 超参（控制 latent bitrate）
    p.add_argument('--loss_factor',   type=float, default=1.5,
                   help='解码器损失放大系数，对抗 posterior collapse，论文用 1.3-1.7')
    p.add_argument('--sigmoid_shift', type=float, default=0.0,
                   help='解码器 sigmoid 权重的偏移量 b')

    # 可视化
    p.add_argument('--viz_every',    type=int,   default=5_000,
                   help='每隔多少步生成一次样本图（0=关闭）')
    p.add_argument('--viz_n_samples',type=int,   default=16)
    p.add_argument('--viz_steps',    type=int,   default=50,
                   help='可视化采样步数，少一些以加快速度')

    # 日志与检查点
    p.add_argument('--log_every',    type=int,   default=100)
    p.add_argument('--save_every',   type=int,   default=10_000)
    p.add_argument('--resume',       type=str,   default=None,
                   help='从检查点恢复训练')

    # WandB
    p.add_argument('--wandb',        action='store_true',
                   help='启用 WandB 日志')
    p.add_argument('--wandb_project', type=str, default='ul',
                   help='WandB 项目名称')
    p.add_argument('--wandb_run',    type=str, default=None,
                   help='WandB 运行名称（默认自动生成）')

    # 其他
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--mixed_precision', action='store_true',
                   help='使用 bf16 混合精度训练')
    p.add_argument('--sampler',        type=str, default='ddim',
                   choices=['ddpm', 'ddim'],
                   help='可视化采样器类型')

    return p.parse_args()


# ============================================================
# DDP 辅助
# ============================================================

def setup_distributed():
    """检测并初始化分布式训练环境（torchrun 启动时自动设置环境变量）。"""
    if 'RANK' not in os.environ:
        return 0, 1, 0

    rank       = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def unwrap(model):
    """获取 DDP 包装下的原始模型。"""
    return model.module if hasattr(model, 'module') else model


def _ddp_no_sync(model):
    """DDP 模型跳过梯度同步（用于非累积边界的 micro-step）。"""
    if hasattr(model, 'no_sync'):
        return model.no_sync()
    return contextlib.nullcontext()


# ============================================================
# EMA（指数移动平均）
# ============================================================

class EMA:
    """
    维护模型参数的指数移动平均，用于推理时获得更稳定的结果。
    EMA 参数不参与梯度更新，只在 log/save 时使用。
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        # 深拷贝一份参数作为 EMA 参数
        self.shadow = {k: v.clone().detach()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        """将 EMA 权重加载到模型，备份原始权重供 restore 恢复。"""
        self._backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model: nn.Module):
        """恢复 apply 前的原始训练权重。"""
        model.load_state_dict(self._backup)
        del self._backup

    def state_dict(self):
        return self.shadow


# ============================================================
# 学习率 Schedule：线性 warmup + cosine decay
# ============================================================

def get_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ============================================================
# 检查点
# ============================================================

def save_checkpoint(path: str, step: int, models: dict,
                    optimizers: dict, emas: dict, args,
                    keep_last: int = 3):
    # 保存时使用 unwrapped 模型
    torch.save({
        'step':       step,
        'args':       vars(args),
        'models':     {k: unwrap(v).state_dict() for k, v in models.items()},
        'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
        'emas':       {k: v.state_dict() for k, v in emas.items()},
    }, path)
    print(f"[ckpt] saved → {path}")

    # 自动清理旧检查点，保留最近 keep_last 个（不删除 ckpt_final）
    if keep_last > 0:
        ckpt_dir = os.path.dirname(path)
        existing = sorted(
            p for p in Path(ckpt_dir).glob('ckpt_*.pt')
            if p.name != 'ckpt_final.pt'
        )
        for old in existing[:-keep_last]:
            old.unlink()
            print(f"[ckpt] removed old → {old}")


def load_checkpoint(path: str, models: dict, optimizers: dict = None,
                    emas: dict = None, device: torch.device = None):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    for k, v in models.items():
        unwrap(v).load_state_dict(ckpt['models'][k])
    if optimizers:
        for k, v in optimizers.items():
            if k in ckpt['optimizers']:
                v.load_state_dict(ckpt['optimizers'][k])
    if emas:
        for k, v in emas.items():
            if k in ckpt['emas']:
                v.shadow = {name: t.to(device) for name, t in ckpt['emas'][k].items()}
    return ckpt.get('step', 0)


# ============================================================
# 阶段一：联合训练
# ============================================================

def train_stage1(args, device: torch.device, rank: int, world_size: int, local_rank: int):
    """
    联合训练编码器、先验、解码器。

    每个 step 做两件事：
      1. 先验损失（unweighted ELBO）：
           对 z_clean 加不同程度的噪声，让先验模型学会对潜变量分布建模。
           梯度会流回编码器，使编码器产生"对先验友好"的潜变量。

      2. 解码器损失（sigmoid reweighted ELBO × loss_factor）：
           对 z_clean 加 t=0 的固定小噪声得到 z_0，
           再以 z_0 为条件对图像加噪，让解码器学会重建图像。
           loss_factor 放大解码器损失，防止 posterior collapse。

    两个损失相加，统一反传。
    """
    is_main = (rank == 0)

    if is_main:
        print("=== 阶段一：联合训练编码器 + 先验 + 解码器 ===")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ----- Noise Schedules -----
    latent_schedule = get_latent_schedule()   # lambda_0=5（固定小噪声）
    image_schedule  = get_image_schedule()    # lambda_0=10

    enc_ch      = tuple(map(int, args.enc_channels.split(',')))
    dec_ch      = tuple(map(int, args.dec_channels.split(',')))

    # ----- 模型 -----
    encoder = Encoder(
        in_channels=3,
        latent_channels=args.latent_channels,
        channel_mults=enc_ch
    ).to(device)
    prior = PriorModel(
        latent_channels=args.latent_channels,
        latent_size=args.latent_size,
        embed_dim=args.embed_dim,
        n_blocks=args.vit_blocks,
        n_heads=args.vit_heads
    ).to(device)
    decoder = DiffusionDecoder(
        in_channels=3, out_channels=3,
        latent_channels=args.latent_channels,
        resolution=args.resolution,
        latent_size=args.latent_size,
        conv_channels=dec_ch,
        embed_dim=args.embed_dim,
        n_blocks=args.vit_blocks,
        n_heads=args.vit_heads
    ).to(device)

    # ----- DDP 包装 -----
    if world_size > 1:
        encoder = DDP(encoder, device_ids=[local_rank])
        prior   = DDP(prior,   device_ids=[local_rank])
        decoder = DDP(decoder, device_ids=[local_rank])

    # ----- EMA（基于 unwrapped 模型）-----
    ema_encoder = EMA(unwrap(encoder), args.ema_decay)
    ema_prior   = EMA(unwrap(prior),   args.ema_decay)
    ema_decoder = EMA(unwrap(decoder), args.ema_decay)

    # ----- 优化器 -----
    all_params = (list(encoder.parameters()) +
                  list(prior.parameters()) +
                  list(decoder.parameters()))
    optimizer = torch.optim.AdamW(all_params, lr=args.lr,
                                  betas=(0.9, 0.99), weight_decay=1e-4)

    # ----- 混合精度 -----
    dtype  = torch.bfloat16 if args.mixed_precision else torch.float32
    scaler = GradScaler(enabled=(args.mixed_precision and dtype == torch.float16))

    # ----- 数据 -----
    loader, sampler = get_dataloader(
        root=args.data_root, split='train',
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        flat=args.flat_data,
        distributed=(world_size > 1),
        rank=rank,
        world_size=world_size,
    )
    loader_iter = _infinite_loader(loader, sampler)

    # ----- 恢复检查点 -----
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            args.resume,
            models={'encoder': encoder, 'prior': prior, 'decoder': decoder},
            optimizers={'optimizer': optimizer},
            emas={'encoder': ema_encoder, 'prior': ema_prior,
                  'decoder': ema_decoder},
            device=device,
        )
        if is_main:
            print(f"[resume] 从 step {start_step} 继续训练")

    # ----- WandB -----
    if is_main and args.wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run,
                   config=vars(args))

    # ----- 训练循环 -----
    encoder.train(); prior.train(); decoder.train()
    log_losses = {'prior': 0.0, 'decoder': 0.0, 'kl': 0.0, 'total': 0.0}
    log_count  = 0

    optimizer.zero_grad()

    for step in range(start_step, args.total_steps):

        # 学习率更新
        lr = get_lr(step, args.total_steps, args.warmup_steps, args.lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x = next(loader_iter).to(device)
        B = x.shape[0]

        with autocast(device_type=device.type, dtype=dtype):

            # ---- 编码 ----
            z_clean = encoder(x)                # [B, C, h, w]，确定性

            # ==========================================
            # 先验损失（unweighted ELBO）
            # ==========================================
            t_prior   = sample_timesteps(B, device)
            z_t, _    = latent_schedule.forward_noise(z_clean, t_prior)
            z_hat     = prior(z_t, t_prior)

            loss_prior = diffusion_loss(
                z_clean, z_hat, t_prior,
                schedule=latent_schedule,
                weight_fn=loss_weight_unweighted,
                loss_factor=1.0,
            )

            # KL[p(z_1|x) || N(0,I)]
            loss_kl = kl_standard_normal(z_clean, latent_schedule)

            # ==========================================
            # 解码器损失（sigmoid reweighted ELBO）
            # ==========================================
            z_0       = add_latent_noise(z_clean, latent_schedule)

            t_dec     = sample_timesteps(B, device)
            x_t, _    = image_schedule.forward_noise(x, t_dec)
            x_hat     = decoder(x_t, z_0, t_dec)

            loss_dec = diffusion_loss(
                x, x_hat, t_dec,
                schedule=image_schedule,
                weight_fn=lambda lam: loss_weight_sigmoid(lam, args.sigmoid_shift),
                loss_factor=args.loss_factor,
            )

            # ==========================================
            # 总损失
            # ==========================================
            loss = loss_prior + loss_kl + loss_dec

        # 反传（梯度累积）
        is_accum_step = (step + 1) % args.grad_accum == 0
        sync_context = contextlib.nullcontext if (is_accum_step or world_size == 1) else _ddp_no_sync
        with sync_context(encoder), sync_context(prior), sync_context(decoder):
            scaler.scale(loss / args.grad_accum).backward()

        if is_accum_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(all_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # EMA 更新（基于 unwrapped 模型）
            ema_encoder.update(unwrap(encoder))
            ema_prior.update(unwrap(prior))
            ema_decoder.update(unwrap(decoder))

        # 记录
        log_losses['prior']   += loss_prior.item()
        log_losses['decoder'] += loss_dec.item()
        log_losses['kl']      += loss_kl.item()
        log_losses['total']   += loss.item()
        log_count += 1

        if (step + 1) % args.log_every == 0 and is_main:
            avg = {k: v / log_count for k, v in log_losses.items()}
            print(
                f"[step {step+1:>7d}] "
                f"lr={lr:.2e} | "
                f"total={avg['total']:.4f} | "
                f"prior={avg['prior']:.4f} | "
                f"dec={avg['decoder']:.4f} | "
                f"kl={avg['kl']:.10f}"
            )
            if args.wandb:
                wandb.log({
                    'loss/total': avg['total'],
                    'loss/prior': avg['prior'],
                    'loss/decoder': avg['decoder'],
                    'loss/kl': avg['kl'],
                    'lr': lr,
                }, step=step + 1)
            log_losses = {k: 0.0 for k in log_losses}
            log_count  = 0

        # 训练中可视化（仅重建对比，阶段一的先验采样质量较差故省略）
        if args.viz_every > 0 and (step + 1) % args.viz_every == 0 and is_main:
            viz_dir = os.path.join(args.output_dir, "viz")
            os.makedirs(viz_dir, exist_ok=True)

            enc_raw = unwrap(encoder)
            dec_raw = unwrap(decoder)

            ema_encoder.apply(enc_raw)
            ema_decoder.apply(dec_raw)
            enc_raw.eval(); dec_raw.eval()

            viz_x = x[:args.viz_n_samples]

            with torch.no_grad():
                imgs = reconstruct(
                    encoder=enc_raw,
                    decoder=dec_raw,
                    images=viz_x,
                    latent_schedule=latent_schedule,
                    image_schedule=image_schedule,
                    n_steps=args.viz_steps,
                    sampler=args.sampler,
                    device=device,
                    dtype=dtype,
                )

            ema_encoder.restore(enc_raw)
            ema_decoder.restore(dec_raw)
            enc_raw.train(); dec_raw.train()

            comparison = torch.cat([viz_x, imgs], dim=0)
            recon_grid = torchvision.utils.make_grid(
                comparison.float().cpu() * 0.5 + 0.5,
                nrow=viz_x.shape[0],
                padding=2,
            )
            save_image(recon_grid, os.path.join(viz_dir, f"step_{step+1:07d}_recon.png"))
            print(f"  [viz] 已保存重建对比网格 → {viz_dir}/step_{step+1:07d}_recon.png")

            if args.wandb:
                wandb.log({
                    'reconstruction': wandb.Image(recon_grid),
                }, step=step + 1)

        if (step + 1) % args.save_every == 0 and is_main:
            save_checkpoint(
                path=os.path.join(args.output_dir, f'ckpt_{step+1:07d}.pt'),
                step=step + 1,
                models={'encoder': encoder, 'prior': prior, 'decoder': decoder},
                optimizers={'optimizer': optimizer},
                emas={'encoder': ema_encoder, 'prior': ema_prior,
                      'decoder': ema_decoder},
                args=args,
            )

    # 最终检查点
    if is_main:
        save_checkpoint(
            path=os.path.join(args.output_dir, 'ckpt_final.pt'),
            step=args.total_steps,
            models={'encoder': encoder, 'prior': prior, 'decoder': decoder},
            optimizers={'optimizer': optimizer},
            emas={'encoder': ema_encoder, 'prior': ema_prior, 'decoder': ema_decoder},
            args=args,
        )
        print("阶段一训练完成。")


# ============================================================
# 阶段二：训练 BaseModel
# ============================================================

def train_stage2(args, device: torch.device, rank: int, world_size: int, local_rank: int):
    """
    冻结编码器和解码器，只训练 BaseModel。
    """
    is_main = (rank == 0)

    if args.stage1_ckpt is None:
        raise ValueError("阶段二需要通过 --stage1_ckpt 指定阶段一的检查点路径。")

    if is_main:
        print("=== 阶段二：训练 BaseModel ===")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    latent_schedule = get_latent_schedule()

    enc_ch      = tuple(map(int, args.enc_channels.split(',')))
    dec_ch      = tuple(map(int, args.dec_channels.split(',')))
    base_dims   = tuple(map(int, args.base_dims.split(',')))
    base_blocks = tuple(map(int, args.base_blocks.split(',')))
    base_heads  = tuple(map(int, args.base_heads.split(',')))

    # ----- 加载阶段一的编码器和解码器（冻结）-----
    encoder = Encoder(
        in_channels=3,
        latent_channels=args.latent_channels,
        channel_mults=enc_ch
    ).to(device)

    decoder = DiffusionDecoder(
        in_channels=3, out_channels=3,
        latent_channels=args.latent_channels,
        resolution=args.resolution,
        latent_size=args.latent_size,
        conv_channels=dec_ch,
        embed_dim=args.embed_dim,
        n_blocks=args.vit_blocks,
        n_heads=args.vit_heads
    ).to(device)

    load_checkpoint(args.stage1_ckpt,
                    models={'encoder': encoder, 'decoder': decoder},
                    device=device)
    for m in (encoder, decoder):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    if is_main:
        print(f"[stage2] 编码器和解码器已从 {args.stage1_ckpt} 加载并冻结。")

    image_schedule = get_image_schedule()

    # ----- BaseModel -----
    base_model = BaseModel(
        latent_channels=args.latent_channels,
        latent_size=args.latent_size,
        stage_dims=base_dims,
        stage_blocks=base_blocks,
        n_heads=base_heads
    ).to(device)

    # ----- DDP 包装 -----
    if world_size > 1:
        base_model = DDP(base_model, device_ids=[local_rank])

    ema_base   = EMA(unwrap(base_model), args.ema_decay)

    optimizer = torch.optim.AdamW(
        base_model.parameters(), lr=args.lr,
        betas=(0.9, 0.99), weight_decay=1e-4,
    )

    dtype  = torch.bfloat16 if args.mixed_precision else torch.float32
    scaler = GradScaler(enabled=(args.mixed_precision and dtype == torch.float16))

    loader, sampler = get_dataloader(
        root=args.data_root, split='train',
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        flat=args.flat_data,
        distributed=(world_size > 1),
        rank=rank,
        world_size=world_size,
    )
    loader_iter = _infinite_loader(loader, sampler)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            args.resume,
            models={'base_model': base_model},
            optimizers={'optimizer': optimizer},
            emas={'base_model': ema_base},
            device=device,
        )

    # ----- WandB -----
    if is_main and args.wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run,
                   config=vars(args))

    base_model.train()
    log_loss  = 0.0
    log_count = 0

    optimizer.zero_grad()

    for step in range(start_step, args.total_steps):

        lr = get_lr(step, args.total_steps, args.warmup_steps, args.lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x = next(loader_iter).to(device)
        B = x.shape[0]

        with autocast(device_type=device.type, dtype=dtype):

            with torch.no_grad():
                z_clean = encoder(x)
                z_0 = add_latent_noise(z_clean, latent_schedule)

            t      = sample_timesteps(B, device)
            z_t, _ = latent_schedule.forward_noise(z_0, t)
            z_hat  = base_model(z_t, t)

            loss = diffusion_loss(
                z_0, z_hat, t,
                schedule=latent_schedule,
                weight_fn=lambda lam: loss_weight_sigmoid(lam, args.sigmoid_shift),
                loss_factor=1.0,
            )

        # 反传（梯度累积）
        is_accum_step = (step + 1) % args.grad_accum == 0
        sync_context = contextlib.nullcontext if (is_accum_step or world_size == 1) else _ddp_no_sync
        with sync_context(base_model):
            scaler.scale(loss / args.grad_accum).backward()

        if is_accum_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema_base.update(unwrap(base_model))

        log_loss  += loss.item()
        log_count += 1

        if (step + 1) % args.log_every == 0 and is_main:
            avg_loss = log_loss / log_count
            print(
                f"[step {step+1:>7d}] "
                f"lr={lr:.2e} | "
                f"loss={avg_loss:.4f}"
            )
            if args.wandb:
                wandb.log({
                    'loss': avg_loss,
                    'lr': lr,
                }, step=step + 1)
            log_loss  = 0.0
            log_count = 0

        # 训练中可视化
        if args.viz_every > 0 and (step + 1) % args.viz_every == 0 and is_main:
            viz_dir = os.path.join(args.output_dir, "viz")
            os.makedirs(viz_dir, exist_ok=True)
            base_raw = unwrap(base_model)
            ema_base.apply(base_raw)
            base_raw.eval()
            grid = make_sample_grid(
                base_model=base_raw,
                decoder=decoder,
                latent_schedule=latent_schedule,
                image_schedule=image_schedule,
                n_samples=args.viz_n_samples,
                latent_channels=args.latent_channels,
                latent_size=args.latent_size,
                resolution=args.resolution,
                n_steps=args.viz_steps,
                sampler=args.sampler,
                device=device,
                dtype=dtype,
            )
            ema_base.restore(base_raw)
            base_raw.train()
            save_image(grid, os.path.join(viz_dir, f"step_{step+1:07d}.png"))
            print(f"  [viz] 已保存样本网格 → {viz_dir}/step_{step+1:07d}.png")

            if args.wandb:
                wandb.log({
                    'samples': wandb.Image(grid.float().cpu() * 0.5 + 0.5),
                }, step=step + 1)

        if (step + 1) % args.save_every == 0 and is_main:
            save_checkpoint(
                path=os.path.join(args.output_dir, f'ckpt_{step+1:07d}.pt'),
                step=step + 1,
                models={'base_model': base_model},
                optimizers={'optimizer': optimizer},
                emas={'base_model': ema_base},
                args=args,
            )

    if is_main:
        save_checkpoint(
            path=os.path.join(args.output_dir, 'ckpt_final.pt'),
            step=args.total_steps,
            models={'base_model': base_model},
            optimizers={'optimizer': optimizer},
            emas={'base_model': ema_base},
            args=args,
        )
        print("阶段二训练完成。")


# ============================================================
# 工具：无限循环 DataLoader
# ============================================================

def _infinite_loader(loader: DataLoader, sampler=None):
    """将有限的 DataLoader 包装成无限迭代器，训练时不需要管 epoch 边界。"""
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


# ============================================================
# 入口
# ============================================================

def main():
    args = get_args()
    if args.latent_size != args.resolution // 16:
        print("latent 不等于分辨率/16， 已经修改")
        args.latent_size = args.resolution // 16

    rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)

    torch.manual_seed(args.seed + rank)

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if is_main:
        print(f"使用设备：{device}（world_size={world_size}）")

    if args.stage == 1:
        train_stage1(args, device, rank, world_size, local_rank)
    elif args.stage == 2:
        train_stage2(args, device, rank, world_size, local_rank)
    else:
        raise ValueError(f"--stage 只支持 1 或 2，收到 {args.stage}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
