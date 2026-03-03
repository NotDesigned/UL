"""
Unified Latents (UL) - Evaluation
计算论文中的三个指标：
  - gFID:  生成图像与真实图像的 FID（衡量生成质量）
  - rFID:  重建图像与真实图像的 FID（衡量重建质量）
  - PSNR:  重建图像与原图的峰值信噪比（衡量像素级重建精度）

依赖：
  pip install pytorch-fid

用法：
  python eval.py \
    --stage1_ckpt ./runs/stage1/ckpt_final.pt \
    --stage2_ckpt ./runs/stage2/ckpt_final.pt \
    --data_root   /path/to/imagenet \
    --n_real      50000 \
    --n_gen       50000
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from dataset import get_dataloader
from utils import get_latent_schedule, get_image_schedule
from sample import (
    sample_latents, sample_images, reconstruct,
    build_models_from_ckpt,
)


# ============================================================
# PSNR
# ============================================================

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    计算批量图像的平均 PSNR（dB）。

    pred / target 值域 [-1, 1]，内部转换到 [0, 1] 再计算。
    PSNR = 10 * log10(MAX^2 / MSE)，MAX=1。
    """
    pred   = (pred.clamp(-1, 1)   * 0.5 + 0.5).float()
    target = (target.clamp(-1, 1) * 0.5 + 0.5).float()

    mse = F.mse_loss(pred, target, reduction='none').flatten(1).mean(1)  # [B]
    # 避免 MSE=0 时 log 溢出
    psnr_per_image = 10 * torch.log10(1.0 / mse.clamp(min=1e-10))
    return psnr_per_image.mean().item()


# ============================================================
# FID（借助 pytorch-fid）
# ============================================================

def compute_fid(real_dir: str, gen_dir: str) -> float:
    """
    调用 pytorch-fid 计算两个目录之间的 FID。
    real_dir 和 gen_dir 下需要有 PNG/JPEG 图像。
    """
    try:
        from pytorch_fid import fid_score
    except ImportError:
        raise ImportError(
            "计算 FID 需要安装 pytorch-fid：pip install pytorch-fid"
        )

    fid = fid_score.calculate_fid_given_paths(
        [real_dir, gen_dir],
        batch_size=64,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dims=2048,
    )
    return fid


# ============================================================
# 保存图像到目录（FID 计算用）
# ============================================================

def save_images_to_dir(images: torch.Tensor, out_dir: str, start_idx: int = 0):
    """
    将 [N, 3, H, W]（值域 [-1,1]）保存为 PNG 文件。
    文件名从 start_idx 开始编号，方便多批次累积写入。
    """
    os.makedirs(out_dir, exist_ok=True)
    images = images.float().cpu() * 0.5 + 0.5  # → [0, 1]
    for i, img in enumerate(images):
        save_image(img, os.path.join(out_dir, f'{start_idx + i:06d}.png'))


# ============================================================
# 主评测流程
# ============================================================

def evaluate(args, device: torch.device):
    latent_schedule = get_latent_schedule()
    image_schedule  = get_image_schedule()

    # ----- 加载模型（从 checkpoint 恢复完整结构参数）-----
    encoder, decoder, base, info = build_models_from_ckpt(
        args.stage1_ckpt, args.stage2_ckpt, device,
    )
    encoder.eval(); decoder.eval(); base.eval()
    latent_channels = info['latent_channels']
    latent_size     = info['latent_size']
    resolution      = info['resolution']

    # ----- 目录准备 -----
    real_dir  = os.path.join(args.output_dir, 'real')
    gen_dir   = os.path.join(args.output_dir, 'generated')
    recon_dir = os.path.join(args.output_dir, 'reconstructed')
    for d in [real_dir, gen_dir, recon_dir]:
        os.makedirs(d, exist_ok=True)

    # ----- 数据加载器（用于 rFID 和 PSNR）-----
    val_loader, _ = get_dataloader(
        root=args.data_root, split='validation',
        resolution=resolution,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ==========================================
    # 1. rFID + PSNR：遍历验证集，编码再解码
    # ==========================================
    print("\n[1/3] 计算 rFID 和 PSNR（重建质量）...")
    psnr_total = 0.0
    psnr_count = 0
    n_real_saved = 0

    for batch_idx, real_images in enumerate(val_loader):
        if n_real_saved >= args.n_real:
            break

        real_images = real_images.to(device)
        bs          = real_images.shape[0]

        # 重建
        recon_images = reconstruct(
            encoder, decoder, real_images,
            latent_schedule, image_schedule,
            n_steps=args.decode_steps,
            sampler=args.sampler,
            device=device,
        )

        # PSNR
        psnr_total += compute_psnr(recon_images, real_images) * bs
        psnr_count += bs

        # 保存真实图和重建图（用于 FID 计算）
        save_images_to_dir(real_images,  real_dir,  start_idx=n_real_saved)
        save_images_to_dir(recon_images, recon_dir, start_idx=n_real_saved)
        n_real_saved += bs

        if (batch_idx + 1) % 10 == 0:
            print(f"  重建进度: {n_real_saved}/{args.n_real} "
                  f"| 当前 PSNR: {psnr_total/psnr_count:.2f} dB")

    psnr = psnr_total / max(psnr_count, 1)
    print(f"  PSNR = {psnr:.2f} dB")

    # ==========================================
    # 2. gFID：采样生成图像
    # ==========================================
    print(f"\n[2/3] 生成 {args.n_gen} 张图像（gFID 用）...")
    n_gen_saved = 0

    while n_gen_saved < args.n_gen:
        bs = min(args.batch_size, args.n_gen - n_gen_saved)

        z_0 = sample_latents(
            base, latent_schedule,
            n_samples=bs,
            latent_channels=latent_channels,
            latent_size=latent_size,
            n_steps=args.latent_steps,
            sampler=args.sampler,
            device=device,
        )
        gen_images = sample_images(
            decoder, z_0, image_schedule,
            n_steps=args.decode_steps,
            sampler=args.sampler,
            resolution=resolution,
            device=device,
        )
        save_images_to_dir(gen_images, gen_dir, start_idx=n_gen_saved)
        n_gen_saved += bs

        if n_gen_saved % 1000 == 0 or n_gen_saved >= args.n_gen:
            print(f"  生成进度: {n_gen_saved}/{args.n_gen}")

    # ==========================================
    # 3. 计算 FID
    # ==========================================
    print("\n[3/3] 计算 FID...")
    rfid = compute_fid(real_dir, recon_dir)
    gfid = compute_fid(real_dir, gen_dir)

    # ==========================================
    # 汇总输出
    # ==========================================
    print("\n" + "=" * 40)
    print(f"  gFID  = {gfid:.2f}")
    print(f"  rFID  = {rfid:.2f}")
    print(f"  PSNR  = {psnr:.2f} dB")
    print("=" * 40)

    # 保存结果
    result_path = os.path.join(args.output_dir, 'results.txt')
    with open(result_path, 'w') as f:
        f.write(f"gFID  = {gfid:.4f}\n")
        f.write(f"rFID  = {rfid:.4f}\n")
        f.write(f"PSNR  = {psnr:.4f} dB\n")
        f.write(f"\nstage1_ckpt = {args.stage1_ckpt}\n")
        f.write(f"stage2_ckpt = {args.stage2_ckpt}\n")
    print(f"结果已保存到 {result_path}")

    return {'gFID': gfid, 'rFID': rfid, 'PSNR': psnr}


# ============================================================
# 命令行入口
# ============================================================

def get_args():
    p = argparse.ArgumentParser()

    p.add_argument('--stage1_ckpt',     type=str, required=True)
    p.add_argument('--stage2_ckpt',     type=str, required=True)
    p.add_argument('--data_root',       type=str, required=True)
    p.add_argument('--output_dir',      type=str, default='./eval_output')

    p.add_argument('--n_real',          type=int, default=50_000,
                   help='用于 rFID 的真实图像数量')
    p.add_argument('--n_gen',           type=int, default=50_000,
                   help='用于 gFID 的生成图像数量')
    p.add_argument('--batch_size',      type=int, default=8)
    p.add_argument('--num_workers',     type=int, default=4)

    p.add_argument('--latent_steps',    type=int, default=100,
                   help='BaseModel 采样步数')
    p.add_argument('--decode_steps',    type=int, default=100,
                   help='DiffusionDecoder 采样步数')
    p.add_argument('--sampler',         type=str, default='ddim',
                   choices=['ddpm', 'ddim'])

    p.add_argument('--seed',            type=int, default=0)
    return p.parse_args()


def main():
    args   = get_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"设备：{device} | 输出目录：{args.output_dir}")
    evaluate(args, device)


if __name__ == '__main__':
    main()