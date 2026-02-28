"""
Unified Latents (UL) - Training
阶段一：编码器 + 先验 + 解码器联合训练
阶段二：冻结编码器/解码器，训练 BaseModel

用法：
  # 阶段一
  python train.py --stage 1 --data_root /path/to/imagenet --output_dir ./runs/stage1

  # 阶段二（需要先完成阶段一）
  python train.py --stage 2 --data_root /path/to/imagenet \
                  --stage1_ckpt ./runs/stage1/ckpt_final.pt \
                  --output_dir ./runs/stage2
"""

import os
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
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

    # 其他
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--mixed_precision', action='store_true',
                   help='使用 bf16 混合精度训练')
    p.add_argument('--sampler',        type=str, default='ddim',
                   choices=['ddpm', 'ddim'],
                   help='可视化采样器类型')

    return p.parse_args()

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
    torch.save({
        'step':       step,
        'args':       vars(args),
        'models':     {k: v.state_dict() for k, v in models.items()},
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
        v.load_state_dict(ckpt['models'][k])
    if optimizers:
        for k, v in optimizers.items():
            if k in ckpt['optimizers']:
                v.load_state_dict(ckpt['optimizers'][k])
    if emas:
        for k, v in emas.items():
            if k in ckpt['emas']:
                v.shadow = ckpt['emas'][k]
    return ckpt.get('step', 0)


# ============================================================
# 阶段一：联合训练
# ============================================================

def train_stage1(args, device: torch.device):
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
    print("=== 阶段一：联合训练编码器 + 先验 + 解码器 ===")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ----- Noise Schedules -----
    latent_schedule = get_latent_schedule()   # lambda_min=5（固定小噪声）
    image_schedule  = get_image_schedule()    # lambda_min=10

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
        n_heads=args.vit_heads      # 必须确保 args.embed_dim % args.vit_heads == 0
    ).to(device)
    
    # 提示：在此前的修改中，Decoder 也需要同步接收头数参数
    decoder = DiffusionDecoder(
        in_channels=3, out_channels=3,
        latent_channels=args.latent_channels,
        resolution=args.resolution,
        latent_size=args.latent_size,
        conv_channels=dec_ch,
        embed_dim=args.embed_dim,
        n_blocks=args.vit_blocks,
        n_heads=args.vit_heads      # 同步传递给解码器中的 ViT 中间层
    ).to(device)

    # ----- EMA -----
    ema_encoder = EMA(encoder, args.ema_decay)
    ema_prior   = EMA(prior,   args.ema_decay)
    ema_decoder = EMA(decoder, args.ema_decay)

    # ----- 优化器 -----
    # 三个模型用同一个优化器，梯度统一管理
    all_params = (list(encoder.parameters()) +
                  list(prior.parameters()) +
                  list(decoder.parameters()))
    optimizer = torch.optim.AdamW(all_params, lr=args.lr,
                                  betas=(0.9, 0.99), weight_decay=1e-4)

    # ----- 混合精度 -----
    # bf16 比 fp16 更稳定（无需 loss scaling），推荐在 Ampere+ 卡上使用
    dtype  = torch.bfloat16 if args.mixed_precision else torch.float32
    scaler = GradScaler(enabled=(args.mixed_precision and dtype == torch.float16))

    # ----- 数据 -----
    loader = get_dataloader(
        root=args.data_root, split='train',
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        flat=args.flat_data,
    )
    loader_iter = _infinite_loader(loader)

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
        print(f"[resume] 从 step {start_step} 继续训练")

    # ----- 训练循环 -----
    encoder.train(); prior.train(); decoder.train()
    log_losses = {'prior': 0.0, 'decoder': 0.0, 'kl': 0.0, 'total': 0.0}
    log_count  = 0

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
            # ------------------------------------------
            # 对 z_clean 采样不同噪声级别的 z_t，
            # 让先验模型学会从任意噪声级别预测干净潜变量。
            # 梯度流回编码器：编码器被迫产生"结构化"的潜变量，
            # 使先验更容易建模。
            # ==========================================
            t_prior   = sample_timesteps(B, device)
            z_t, _    = latent_schedule.forward_noise(z_clean, t_prior)
            z_hat     = prior(z_t, t_prior)

            loss_prior = diffusion_loss(
                z_clean, z_hat, t_prior,
                schedule=latent_schedule,
                weight_fn=loss_weight_unweighted,   # 必须 unweighted
                loss_factor=1.0,
            )

            # KL[p(z_1|x) || N(0,I)]：z_1 几乎是纯噪声，这项接近 0
            # 保留它是为了完整的 ELBO，实践中权重很小
            loss_kl = kl_standard_normal(z_clean, latent_schedule)

            # ==========================================
            # 解码器损失（sigmoid reweighted ELBO）
            # ------------------------------------------
            # z_0：对 z_clean 加 t=0 处的固定小噪声（sigma ≈ 0.08）
            # 这是 UL 的关键设计：将编码器的精度与先验的最大精度对齐
            # ==========================================
            z_0       = add_latent_noise(z_clean, latent_schedule)  # 固定 t=0 噪声

            t_dec     = sample_timesteps(B, device)
            x_t, _    = image_schedule.forward_noise(x, t_dec)
            x_hat     = decoder(x_t, z_0, t_dec)

            loss_dec = diffusion_loss(
                x, x_hat, t_dec,
                schedule=image_schedule,
                weight_fn=lambda lam: loss_weight_sigmoid(lam, args.sigmoid_shift),
                loss_factor=args.loss_factor,       # 放大解码器损失，防 posterior collapse
            )

            # ==========================================
            # 总损失
            # ==========================================
            loss = loss_prior + loss_kl + loss_dec

        # 反传
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(all_params, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # EMA 更新
        ema_encoder.update(encoder)
        ema_prior.update(prior)
        ema_decoder.update(decoder)

        # 记录
        log_losses['prior']   += loss_prior.item()
        log_losses['decoder'] += loss_dec.item()
        log_losses['kl']      += loss_kl.item()
        log_losses['total']   += loss.item()
        log_count += 1

        if (step + 1) % args.log_every == 0:
            avg = {k: v / log_count for k, v in log_losses.items()}
            print(
                f"[step {step+1:>7d}] "
                f"lr={lr:.2e} | "
                f"total={avg['total']:.4f} | "
                f"prior={avg['prior']:.4f} | "
                f"dec={avg['decoder']:.4f} | "
                f"kl={avg['kl']:.10f}"
            )
            log_losses = {k: 0.0 for k in log_losses}
            log_count  = 0

        # 训练中可视化
        if args.viz_every > 0 and (step + 1) % args.viz_every == 0:
            viz_dir = os.path.join(args.output_dir, "viz")
            os.makedirs(viz_dir, exist_ok=True)
            # 阶段一还没有 BaseModel，用先验模型临时替代做 latent 采样
            # 注意：阶段一的先验采样质量较差，仅用于观察解码器是否在学习
            
            grid = make_sample_grid(
                base_model=prior,
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
            save_image(grid, os.path.join(viz_dir, f"step_{step+1:07d}.png"))
            print(f"  [viz] 已保存样本网格 → {viz_dir}/step_{step+1:07d}.png")
            
            # 阶段一切换为重建验证模式
            was_training_enc = encoder.training
            was_training_dec = decoder.training
            encoder.eval()
            decoder.eval()

            # 从当前训练批次中截取部分真实图像作为基准
            viz_x = x[:args.viz_n_samples]

            with torch.no_grad():
                # 直接测试从原图 -> 编码 -> 解码的完整信号通路
                imgs = reconstruct(
                    encoder=encoder,
                    decoder=decoder,
                    images=viz_x,
                    latent_schedule=latent_schedule,
                    image_schedule=image_schedule,
                    n_steps=args.viz_steps,
                    sampler=args.sampler,
                    device=device,
                    dtype=dtype,
                )

            encoder.train(was_training_enc)
            decoder.train(was_training_dec)

            # 拼接：上方为真实原图，下方为解码器重建图，直观对比特征保真度
            comparison = torch.cat([viz_x, imgs], dim=0)
            grid = torchvision.utils.make_grid(
                comparison.float().cpu() * 0.5 + 0.5,
                nrow=viz_x.shape[0],  # 每行显示同一批次的图
                padding=2,
            )
            save_image(grid, os.path.join(viz_dir, f"step_{step+1:07d}_recon.png"))
            print(f"  [viz] 已保存重建对比网格 → {viz_dir}/step_{step+1:07d}_recon.png")

        if (step + 1) % args.save_every == 0:
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

def train_stage2(args, device: torch.device):
    """
    冻结编码器和解码器，只训练 BaseModel。

    阶段二的训练几乎等同于标准 LDM 的训练：
      - 用冻结的编码器把图像批量编码为 z_clean
      - 对 z_clean 加 t=0 固定噪声得到 z_0
      - 对 z_0 采样不同噪声级别，让 BaseModel 学会预测 z_clean
      - 损失用 sigmoid weighting（与阶段一的先验不同）

    唯一区别：logsnr 的上界固定为 lambda_min=5（与阶段一的 latent schedule 一致），
    而不是标准 LDM 里的自由超参。
    """
    if args.stage1_ckpt is None:
        raise ValueError("阶段二需要通过 --stage1_ckpt 指定阶段一的检查点路径。")

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
    ema_base   = EMA(base_model, args.ema_decay)

    optimizer = torch.optim.AdamW(
        base_model.parameters(), lr=args.lr,
        betas=(0.9, 0.99), weight_decay=1e-4,
    )

    dtype  = torch.bfloat16 if args.mixed_precision else torch.float32
    scaler = GradScaler(enabled=(args.mixed_precision and dtype == torch.float16))

    loader = get_dataloader(
        root=args.data_root, split='train',
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        flat=args.flat_data,
    )
    loader_iter = _infinite_loader(loader)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            args.resume,
            models={'base_model': base_model},
            optimizers={'optimizer': optimizer},
            emas={'base_model': ema_base},
            device=device,
        )

    base_model.train()
    log_loss  = 0.0
    log_count = 0

    for step in range(start_step, args.total_steps):

        lr = get_lr(step, args.total_steps, args.warmup_steps, args.lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x = next(loader_iter).to(device)
        B = x.shape[0]

        with autocast(device_type=device.type, dtype=dtype):

            # 编码器不参与梯度计算
            with torch.no_grad():
                z_clean = encoder(x)
                # z_0：加固定 t=0 噪声，这是 BaseModel 的训练目标
                z_0 = add_latent_noise(z_clean, latent_schedule)

            # 对 z_0 采样不同噪声级别，BaseModel 学会从任意 z_t 预测 z_0
            # 注意：BaseModel 预测的是 z_0（略带噪声的潜变量），而非 z_clean
            # 这与阶段一的先验预测 z_clean 不同，与论文 Sec.3.3 一致
            t      = sample_timesteps(B, device)
            z_t, _ = latent_schedule.forward_noise(z_0, t)
            z_hat  = base_model(z_t, t)

            loss = diffusion_loss(
                z_0, z_hat, t,
                schedule=latent_schedule,
                weight_fn=lambda lam: loss_weight_sigmoid(lam, args.sigmoid_shift),
                loss_factor=1.0,    # 阶段二不需要 loss_factor
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        ema_base.update(base_model)

        log_loss  += loss.item()
        log_count += 1

        if (step + 1) % args.log_every == 0:
            print(
                f"[step {step+1:>7d}] "
                f"lr={lr:.2e} | "
                f"loss={log_loss / log_count:.4f}"
            )
            log_loss  = 0.0
            log_count = 0

        # 训练中可视化
        if args.viz_every > 0 and (step + 1) % args.viz_every == 0:
            viz_dir = os.path.join(args.output_dir, "viz")
            os.makedirs(viz_dir, exist_ok=True)
            grid = make_sample_grid(
                base_model=base_model,
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
            save_image(grid, os.path.join(viz_dir, f"step_{step+1:07d}.png"))
            print(f"  [viz] 已保存样本网格 → {viz_dir}/step_{step+1:07d}.png")

        if (step + 1) % args.save_every == 0:
            save_checkpoint(
                path=os.path.join(args.output_dir, f'ckpt_{step+1:07d}.pt'),
                step=step + 1,
                models={'base_model': base_model},
                optimizers={'optimizer': optimizer},
                emas={'base_model': ema_base},
                args=args,
            )

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

def _infinite_loader(loader: DataLoader):
    """将有限的 DataLoader 包装成无限迭代器，训练时不需要管 epoch 边界。"""
    while True:
        for batch in loader:
            yield batch


# ============================================================
# 入口
# ============================================================

def main():
    args = get_args()
    if args.latent_size != args.resolution // 16:
        print("latent 不等于分辨率/16， 已经修改")
        args.latent_size = args.resolution // 16
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")

    if args.stage == 1:
        train_stage1(args, device)
    elif args.stage == 2:
        train_stage2(args, device)
    else:
        raise ValueError(f"--stage 只支持 1 或 2，收到 {args.stage}")


if __name__ == '__main__':
    main()