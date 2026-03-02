"""
Unified Latents (UL) - Sampling

核心采样函数（sample_latents / sample_images / reconstruct）只接受模型对象，
不涉及任何文件 IO，可以直接在 train.py 里调用做训练中可视化。

命令行入口负责加载检查点、保存图像等 IO 操作。

用法：
  python sample.py \
    --stage1_ckpt ./runs/stage1/ckpt_final.pt \
    --stage2_ckpt ./runs/stage2/ckpt_final.pt \
    --output_dir  ./samples \
    --n_samples   16 --sampler ddim \
    --latent_steps 100 --image_steps 100
"""

import os
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image, make_grid

from models import Encoder, DiffusionDecoder, BaseModel, PriorModel
from utils import NoiseSchedule, get_latent_schedule, get_image_schedule


# ============================================================
# 单步去噪（DDPM / DDIM）
# ============================================================

@torch.no_grad()
def denoise_step(
    model_out: torch.Tensor,
    x_t:       torch.Tensor,
    t_now:     float,
    t_next:    float,
    schedule:  NoiseSchedule,
    eta:       float = 0.0,     # 0=DDIM（确定性），1=DDPM（随机）
) -> torch.Tensor:
    """
    从 x_t（t=t_now）推进到 x_{t_next}（t=t_next）。
    eta=0 时为 DDIM，eta=1 时为完整 DDPM。
    """
    device   = x_t.device
    dtype    = x_t.dtype
    t_now_t  = torch.tensor([t_now],  device=device, dtype=dtype)
    t_next_t = torch.tensor([t_next], device=device, dtype=dtype)

    alpha_now,  sigma_now  = schedule.alpha_sigma(t_now_t)
    alpha_next, sigma_next = schedule.alpha_sigma(t_next_t)

    eps_pred = (x_t - alpha_now * model_out) / sigma_now.clamp(min=1e-8)

    if eta > 0 and t_next > 0:
        # 标准 DDIM（Song et al. 2020）：
        #   σ²_η = η² · (σ²_{next} / σ²_{now}) · (1 - α²_{now} / α²_{next})
        #   x_{next} = α_{next} · x̂_0 + √(σ²_{next} - σ²_η) · ε_pred + σ_η · z
        sigma_eta_sq = (eta ** 2) * (sigma_next ** 2 / sigma_now.clamp(min=1e-8) ** 2) * \
                       (1 - (alpha_now / alpha_next.clamp(min=1e-8)) ** 2)
        sigma_eta_sq = sigma_eta_sq.clamp(min=0)
        sigma_eta    = sigma_eta_sq.sqrt()
        coeff_eps    = (sigma_next ** 2 - sigma_eta_sq).clamp(min=0).sqrt()
        x_next = alpha_next * model_out + coeff_eps * eps_pred + sigma_eta * torch.randn_like(x_t)
    else:
        # 纯 DDIM（确定性）
        x_next = alpha_next * model_out + sigma_next * eps_pred

    return x_next


def _run_diffusion_loop(
    model_fn,               # callable: (x_t, t_batch) -> x_hat
    shape:    tuple,
    schedule: NoiseSchedule,
    n_steps:  int,
    eta:      float,
    device:   torch.device,
) -> torch.Tensor:
    """
    通用扩散采样循环。
    model_fn 封装了模型调用（含条件），外部传入，内部不感知模型类型。
    """
    B         = shape[0]
    x         = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    timesteps = torch.linspace(1.0, 0.0, n_steps + 1).tolist()

    for i in range(n_steps):
        t_now  = timesteps[i]
        t_next = timesteps[i + 1]
        t_b    = torch.full((B,), t_now, device=device, dtype=torch.bfloat16)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            x_hat  = model_fn(x, t_b)
        x      = denoise_step(x_hat, x, t_now, t_next, schedule, eta)

    return x


# ============================================================
# 对外接口（只接受模型对象，无 IO）
# ============================================================

@torch.no_grad()
def sample_latents(
    base_model:      BaseModel,
    latent_schedule: NoiseSchedule,
    n_samples:       int,
    latent_channels: int,
    latent_size:     int,
    n_steps:         int   = 100,
    sampler:         str   = 'ddim',
    device:          torch.device = None,
) -> torch.Tensor:
    """
    BaseModel 从纯噪声采样 z_0。
    可在训练循环中直接调用（传入正在训练的 base_model 即可）。
    """
    if device is None:
        device = next(base_model.parameters()).device

    eta = 0.0 if sampler == 'ddim' else 1.0

    return _run_diffusion_loop(
        model_fn  = base_model,
        shape     = (n_samples, latent_channels, latent_size, latent_size),
        schedule  = latent_schedule,
        n_steps   = n_steps,
        eta       = eta,
        device    = device,
    )


@torch.no_grad()
def sample_images(
    decoder:        DiffusionDecoder,
    z_0:            torch.Tensor,
    image_schedule: NoiseSchedule,
    n_steps:        int  = 100,
    sampler:        str  = 'ddim',
    resolution:     int  = 512,
    device:         torch.device = None,
) -> torch.Tensor:
    """
    DiffusionDecoder 以 z_0 为条件采样图像。
    可在训练循环中直接调用。

    Returns:
        images [B, 3, H, W]，值域 [-1, 1]
    """
    if device is None:
        device = next(decoder.parameters()).device

    B   = z_0.shape[0]
    eta = 0.0 if sampler == 'ddim' else 1.0
    z_0 = z_0.to(device=device, dtype=torch.bfloat16)

    images = _run_diffusion_loop(
        model_fn  = lambda x, t: decoder(x, z_0, t),
        shape     = (B, 3, resolution, resolution),
        schedule  = image_schedule,
        n_steps   = n_steps,
        eta       = eta,
        device    = device,
    )
    return images.clamp(-1, 1)


@torch.no_grad()
def reconstruct(
    encoder:         Encoder,
    decoder:         DiffusionDecoder,
    images:          torch.Tensor,
    latent_schedule: NoiseSchedule,
    image_schedule:  NoiseSchedule,
    n_steps:         int  = 50,
    sampler:         str  = 'ddim',
    device:          torch.device = None,
) -> torch.Tensor:
    """
    编码 → z_0 → 解码，用于重建质量评测（PSNR / rFID）。
    z_0 来自编码器而非 BaseModel，与生成路径完全独立。
    """
    if device is None:
        device = next(encoder.parameters()).device

    images = images.to(device=device, dtype=torch.bfloat16)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        z_clean = encoder(images)

    t_zero = torch.zeros(images.shape[0], device=device, dtype=torch.bfloat16)
    z_0, _ = latent_schedule.forward_noise(z_clean, t_zero)

    return sample_images(
        decoder, z_0, image_schedule,
        n_steps=n_steps, sampler=sampler,
        resolution=images.shape[-1],
        device=device,
    )


# ============================================================
# 便利函数：生成网格图（训练可视化用）
# ============================================================

@torch.no_grad()
def make_sample_grid(
    base_model:      BaseModel | PriorModel,
    decoder:         DiffusionDecoder,
    latent_schedule: NoiseSchedule,
    image_schedule:  NoiseSchedule,
    n_samples:       int  = 16,
    latent_channels: int  = 32,
    latent_size:     int  = 32,
    resolution:      int  = 512,
    n_steps:         int  = 50,     # 可视化时用更少步数以加快速度
    sampler:         str  = 'ddim',
    device:          torch.device = None,
) -> torch.Tensor:
    """
    一步生成 n_samples 张图像并拼成网格，供训练日志使用。

    Returns:
        grid [3, H, W]，值域 [0, 1]，可直接传给 save_image 或 wandb
    """
    was_training_base    = base_model.training
    was_training_decoder = decoder.training
    base_model.eval()
    decoder.eval()

    z_0 = sample_latents(
        base_model, latent_schedule,
        n_samples=n_samples,
        latent_channels=latent_channels,
        latent_size=latent_size,
        n_steps=n_steps, sampler=sampler,
        device=device,
    )
    imgs = sample_images(
        decoder, z_0, image_schedule,
        n_steps=n_steps, sampler=sampler,
        resolution=resolution,
        device=device,
    )

    # 恢复训练模式
    base_model.train(was_training_base)
    decoder.train(was_training_decoder)

    grid = make_grid(
        imgs.float().cpu() * 0.5 + 0.5,    # [-1,1] → [0,1]
        nrow=int(n_samples ** 0.5),
        padding=2,
    )
    return grid


# ============================================================
# IO 工具（仅命令行入口使用）
# ============================================================

def build_models_from_ckpt(
    stage1_ckpt: str,
    stage2_ckpt: str,
    device: torch.device,
):
    """
    从 checkpoint 保存的 args 中恢复模型结构参数并构建完整模型。
    所有超参从 checkpoint 读取，避免手动指定导致不一致。

    Returns:
        (encoder, decoder, base_model, info_dict)
    """
    ckpt1 = torch.load(stage1_ckpt, map_location=device, weights_only=True)
    ckpt2 = torch.load(stage2_ckpt, map_location=device, weights_only=True)
    ta = ckpt1.get('args', {})

    latent_channels = ta['latent_channels']
    resolution      = ta['resolution']
    enc_ch          = tuple(ta['enc_channels'])
    dec_ch          = tuple(ta['dec_channels'])
    embed_dim       = ta['embed_dim']
    vit_blocks      = ta['vit_blocks']
    vit_heads       = ta['vit_heads']
    latent_size     = resolution // (2 ** len(enc_ch))

    ta2 = ckpt2.get('args', {})
    base_dims   = tuple(ta2['base_dims'])
    base_blocks = tuple(ta2['base_blocks'])
    base_heads  = tuple(ta2['base_heads'])

    encoder = Encoder(
        in_channels=3, latent_channels=latent_channels, channel_mults=enc_ch,
    ).to(device)
    decoder = DiffusionDecoder(
        in_channels=3, out_channels=3, latent_channels=latent_channels,
        resolution=resolution, latent_size=latent_size,
        conv_channels=dec_ch, embed_dim=embed_dim,
        n_blocks=vit_blocks, n_heads=vit_heads,
    ).to(device)
    base_model = BaseModel(
        latent_channels=latent_channels, latent_size=latent_size,
        stage_dims=base_dims, stage_blocks=base_blocks, n_heads=base_heads,
    ).to(device)

    # 加载权重（优先 EMA）
    for key, model, ckpt in [
        ('encoder', encoder, ckpt1), ('decoder', decoder, ckpt1),
        ('base_model', base_model, ckpt2),
    ]:
        if 'emas' in ckpt and key in ckpt['emas']:
            model.load_state_dict(ckpt['emas'][key])
            print(f"  [load] {key} ← EMA")
        elif 'models' in ckpt and key in ckpt['models']:
            model.load_state_dict(ckpt['models'][key])
            print(f"  [load] {key} ← model")
        else:
            raise KeyError(f"检查点中找不到 key='{key}'")

    info = dict(latent_channels=latent_channels, latent_size=latent_size,
                resolution=resolution)
    return encoder, decoder, base_model, info


# ============================================================
# 命令行入口
# ============================================================

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage1_ckpt',     type=str, required=True)
    p.add_argument('--stage2_ckpt',     type=str, required=True)
    p.add_argument('--output_dir',      type=str, default='./samples')
    p.add_argument('--n_samples',       type=int, default=16)
    p.add_argument('--batch_size',      type=int, default=4)
    p.add_argument('--sampler',         type=str, default='ddim',
                   choices=['ddpm', 'ddim'])
    p.add_argument('--latent_steps',    type=int, default=100)
    p.add_argument('--image_steps',     type=int, default=100)
    p.add_argument('--seed',            type=int, default=0)
    return p.parse_args()


def main():
    args   = get_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    latent_schedule = get_latent_schedule()
    image_schedule  = get_image_schedule()

    print("加载权重...")
    encoder, decoder, base, info = build_models_from_ckpt(
        args.stage1_ckpt, args.stage2_ckpt, device,
    )
    encoder.eval(); decoder.eval(); base.eval()

    latent_channels = info['latent_channels']
    latent_size     = info['latent_size']
    resolution      = info['resolution']

    all_images = []
    remaining  = args.n_samples
    while remaining > 0:
        bs = min(args.batch_size, remaining)
        print(f"  采样 {bs} 张（剩余 {remaining}）...")

        z_0  = sample_latents(base, latent_schedule,
                               n_samples=bs,
                               latent_channels=latent_channels,
                               latent_size=latent_size,
                               n_steps=args.latent_steps,
                               sampler=args.sampler,
                               device=device)
        imgs = sample_images(decoder, z_0, image_schedule,
                              n_steps=args.image_steps,
                              sampler=args.sampler,
                              resolution=resolution,
                              device=device)
        all_images.append(imgs.float().cpu())
        remaining -= bs

    all_images = torch.cat(all_images, dim=0)

    for i, img in enumerate(all_images):
        save_image(img * 0.5 + 0.5,
                   os.path.join(args.output_dir, f'sample_{i:04d}.png'))

    grid = make_grid(
        all_images * 0.5 + 0.5,
        nrow=int(args.n_samples ** 0.5), padding=2,
    )
    save_image(grid, os.path.join(args.output_dir, 'grid.png'))
    print(f"完成，已保存到 {args.output_dir}/")


if __name__ == '__main__':
    main()