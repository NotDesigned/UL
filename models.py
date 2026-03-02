"""
Unified Latents (UL) - Model Architectures
包含：
  - Encoder:          ResNet 编码器（确定性）
  - PriorModel:       ViT 先验扩散模型（阶段一）
  - BaseModel:        两阶段 ViT（阶段二）
  - DiffusionDecoder: UViT 扩散解码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import math


# ============================================================
# 通用基础模块
# ============================================================

def _num_groups(channels: int, target: int = 32) -> int:
    """
    为 GroupNorm 选择合适的 num_groups。
    优先用 target=32，若不整除则取最大公约数方向往下找最近的因子。
    """
    g = min(target, channels)
    while channels % g != 0:
        g -= 1
    return g

class ResBlock(nn.Module):
    """
    标准残差块：GroupNorm + SiLU + Conv。
    时间步 embedding 通过加法注入（可选）。
    forward 签名始终是 (x, t_emb=None)，
    方便在 ModuleList 里统一调用而不需要 isinstance 判断。
    """
    def __init__(self, in_channels: int, out_channels: int,
                 time_emb_dim: int = None):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_channels),  in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act   = nn.SiLU()
        self.skip  = (nn.Conv2d(in_channels, out_channels, 1)
                      if in_channels != out_channels else nn.Identity())
        self.time_proj = (nn.Linear(time_emb_dim, out_channels)
                          if time_emb_dim is not None else None)

    def forward(self, x: torch.Tensor,
                t_emb: torch.Tensor = None) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        if self.time_proj is not None and t_emb is not None:
            h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    """stride=2 卷积下采样。"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """nearest 插值 + 卷积上采样。"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


def run_layers(layers: nn.ModuleList, x: torch.Tensor,
               t_emb: torch.Tensor = None) -> torch.Tensor:
    """
    统一遍历 ModuleList，每层都以 (x, t_emb) 调用。
    ResBlock / Downsample / Upsample 的 forward 都接受 t_emb=None，
    所以这里无需 isinstance 判断。
    """
    for layer in layers:
        x = layer(x, t_emb)
    return x


def _run_block(block, tokens, t_emb, use_ckpt: bool):
    """统一处理 ViT block 调用，支持 gradient checkpointing。"""
    if use_ckpt:
        return grad_checkpoint(block, tokens, t_emb, use_reentrant=False)
    return block(tokens, t_emb)


def make_sinusoidal_freqs(dim: int) -> torch.Tensor:
    """预计算 sinusoidal embedding 的频率向量，用于 register_buffer 缓存。"""
    assert dim % 2 == 0
    half = dim // 2
    return torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)


def sinusoidal_embedding(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """t ∈ [0,1] → sinusoidal embedding [B, dim]，使用预计算的 freqs buffer。"""
    args = (t * 1000)[:, None] * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)

def interpolate_pos_embed(pos_embed: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    将一维形式的二维网格位置编码自适应插值到目标尺寸 (h, w)。
    
    参数:
        pos_embed: 形状为 [1, N, D] 的张量，其中 N = H_orig * W_orig
        h: 目标特征网格的高度
        w: 目标特征网格的宽度
        
    返回:
        形状为 [1, h*w, D] 的插值后位置编码张量
    """
    N = pos_embed.shape[1]
    # 若当前网格尺寸与位置编码初始化尺寸一致，直接返回
    if N == h * w:
        return pos_embed

    D = pos_embed.shape[2]
    # 假定原始位置编码对应一个正方形特征网格，推导原始物理尺寸
    orig_size = int(math.sqrt(N))
    assert orig_size * orig_size == N, "位置编码序列长度不是完全平方数，无法推导原始二维网格维度。"

    # [1, N, D] -> [1, D, N] -> [1, D, H_orig, W_orig]
    pos_embed_grid = pos_embed.transpose(1, 2).reshape(1, D, orig_size, orig_size)

    # 执行二维双三次插值对齐新分辨率
    pos_embed_grid_resized = F.interpolate(
        pos_embed_grid,
        size=(h, w),
        mode='bicubic',
        align_corners=False
    )

    # [1, D, h, w] -> [1, D, h*w] -> [1, h*w, D]
    pos_embed_resized = pos_embed_grid_resized.flatten(2).transpose(1, 2)
    
    return pos_embed_resized


# ============================================================
# 1. Encoder（ResNet，确定性）
# ============================================================

class Encoder(nn.Module):
    """
    确定性 ResNet 编码器。
    下采样 2^N×：patch_embed(2×) + (N-1) 个 Downsample(2×)，N = len(channel_mults)。
    前 N-1 个阶段各 2 个 ResBlock + Downsample，最后一个阶段 3 个 ResBlock 不下采样。
    输出 z_clean [B, latent_channels, H/(2^N), W/(2^N)]。
    """
    def __init__(self, in_channels: int = 3, latent_channels: int = 32,
                 channel_mults: tuple = (128, 256, 512, 512)):
        super().__init__()

        self.patch_embed = nn.Conv2d(in_channels, channel_mults[0],
                                     kernel_size=2, stride=2)

        layers: list[nn.Module] = []
        in_ch = channel_mults[0]
        for i, out_ch in enumerate(channel_mults):
            n_blocks = 3 if i == len(channel_mults) - 1 else 2
            for _ in range(n_blocks):
                layers.append(ResBlock(in_ch, out_ch))   # time_emb_dim=None
                in_ch = out_ch
            if i < len(channel_mults) - 1:
                layers.append(Downsample(out_ch))

        self.layers   = nn.ModuleList(layers)
        self.norm_out = nn.GroupNorm(_num_groups(channel_mults[-1]), channel_mults[-1])
        self.proj_out = nn.Conv2d(channel_mults[-1], latent_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.patch_embed(x)
        h = run_layers(self.layers, h)          # t_emb=None，编码器不需要
        h = F.silu(self.norm_out(h))
        return self.proj_out(h)


# ============================================================
# 2. ViT 基础模块（先验 & base model 共用）
# ============================================================

class PatchEmbed2D(nn.Module):
    """特征图 → token 序列。"""
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        """返回 (tokens [B,N,D], (h, w))"""
        h, w = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        return tokens, (h, w)


class SelfAttention(nn.Module):
    """
    多头自注意力，使用 F.scaled_dot_product_attention。
    PyTorch 2.0+ 会自动分派到 FlashAttention（CUDA fp16/bf16）
    或 memory-efficient attention，无需额外安装。
    """
    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.qkv  = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)                            # [B,N,heads,head_dim]
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # [B,heads,N,head_dim]

        # scale、softmax、dropout 由 PyTorch 内部处理；bf16/fp16 时自动用 FlashAttn
        out = F.scaled_dot_product_attention(q, k, v)      # [B,heads,N,head_dim]
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class ViTBlock(nn.Module):
    """
    ViT block with adaLN 时间调制。
    adaLN：用 t_emb 预测 (scale1, shift1, scale2, shift2)，
    分别作用于 Attention 前和 FFN 前的 LayerNorm。
    """
    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: float = 4.0,
                 time_emb_dim: int = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = SelfAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim)
        )
        if time_emb_dim is not None:
            ada_linear = nn.Linear(time_emb_dim, dim * 4)
            nn.init.zeros_(ada_linear.weight)
            nn.init.zeros_(ada_linear.bias)
            self.ada_ln = nn.Sequential(nn.SiLU(), ada_linear)
        else:
            self.ada_ln = None

    def forward(self, x: torch.Tensor,
                t_emb: torch.Tensor = None) -> torch.Tensor:
        if self.ada_ln is not None and t_emb is not None:
            s1, b1, s2, b2 = self.ada_ln(t_emb).chunk(4, dim=-1)  # 各 [B,dim]
            x = x + self.attn(self.norm1(x) * (1 + s1[:, None]) + b1[:, None])
            x = x + self.mlp( self.norm2(x) * (1 + s2[:, None]) + b2[:, None])
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp( self.norm2(x))
        return x


# ============================================================
# 3. PriorModel（单层 ViT，阶段一）
# ============================================================

class PriorModel(nn.Module):
    """
    潜在空间扩散先验（阶段一，unweighted ELBO）。
    单层 ViT：8 blocks，1024 channels。
    输入 z_t [B,C,h,w] + t [B] → 输出 z_hat [B,C,h,w]。
    """
    def __init__(self, latent_channels: int = 32, latent_size: int = 32,
                 embed_dim: int = 1024, n_blocks: int = 8,
                 n_heads: int = 16, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        time_dim        = embed_dim * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.register_buffer('_sin_freqs', make_sinusoidal_freqs(embed_dim), persistent=False)
        self.patch_embed = PatchEmbed2D(latent_channels, embed_dim, patch_size)
        n_patches        = (latent_size // patch_size) ** 2
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.blocks   = nn.ModuleList([
            ViTBlock(embed_dim, n_heads, time_emb_dim=time_dim)
            for _ in range(n_blocks)
        ])
        self.norm     = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Linear(embed_dim, latent_channels * patch_size ** 2)
        self.gradient_checkpointing = False

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z_t.shape
        t_emb = self.time_mlp(sinusoidal_embedding(t, self._sin_freqs))

        tokens, (h, w) = self.patch_embed(z_t)
        tokens = tokens + interpolate_pos_embed(self.pos_embed, h, w)
        for block in self.blocks:
            tokens = _run_block(block, tokens, t_emb, self.gradient_checkpointing and self.training)
        tokens = self.norm(tokens)

        p      = self.patch_size
        tokens = self.proj_out(tokens)                        # [B,N,C*p*p]
        tokens = tokens.reshape(B, h, w, C, p, p)
        residual = tokens.permute(0, 3, 1, 4, 2, 5).reshape(B, C, h*p, w*p)
        return z_t + residual


# ============================================================
# 4. DiffusionDecoder（UViT）
# ============================================================

class DiffusionDecoder(nn.Module):
    """
    UViT 扩散解码器，以 z_0 为条件重建图像。

    结构：卷积下采样 → [z_0 prefix tokens | img tokens] → ViT → 卷积上采样。
    z_0 以 prefix token 方式拼接，让自注意力自然跨图像和潜变量交互。

    论文配置：卷积通道 [128, 256, 512]，ViT 8 blocks / 1024 dim。
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 latent_channels: int = 32,
                 conv_channels: tuple = (128, 256, 512),
                 embed_dim: int = 1024, n_blocks: int = 8,
                 n_heads: int = 16, patch_size: int = 2,
                 dropout: float = 0.1,
                 resolution: int = 512, latent_size: int = 32):
        super().__init__()
        self.patch_size    = patch_size
        self.conv_channels = conv_channels
        time_dim           = embed_dim * 4

        # 动态计算 pos_embed 大小
        # 图像经过 len(conv_channels) 次 2× 下采样后进入 ViT
        feat_size     = resolution // (2 ** len(conv_channels))
        n_img_patches = (feat_size   // patch_size) ** 2
        n_z_patches   = (latent_size // patch_size) ** 2

        # ----- 时间步 embedding -----
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.register_buffer('_sin_freqs', make_sinusoidal_freqs(embed_dim), persistent=False)

        self.conv_in = nn.Conv2d(in_channels, conv_channels[0], 3, padding=1)

        # ----- 下采样路径 -----
        # 每个阶段：ResBlock × 2，然后 Downsample
        self.down_layers = nn.ModuleList()
        in_ch = conv_channels[0]
        for out_ch in conv_channels:
            self.down_layers.append(ResBlock(in_ch, out_ch, time_emb_dim=time_dim))
            self.down_layers.append(ResBlock(out_ch, out_ch, time_emb_dim=time_dim))
            self.down_layers.append(Downsample(out_ch))
            in_ch = out_ch

        # ----- ViT 中间层 -----
        self.img_patch_embed = PatchEmbed2D(conv_channels[-1], embed_dim, patch_size)
        self.img_pos_embed   = nn.Parameter(torch.zeros(1, n_img_patches, embed_dim))
        nn.init.normal_(self.img_pos_embed, std=0.02)

        self.z_patch_embed = PatchEmbed2D(latent_channels, embed_dim, patch_size)
        self.z_pos_embed   = nn.Parameter(torch.zeros(1, n_z_patches, embed_dim))
        nn.init.normal_(self.z_pos_embed, std=0.02)

        self.vit_blocks = nn.ModuleList([
            ViTBlock(embed_dim, n_heads, time_emb_dim=time_dim)
            for _ in range(n_blocks)
        ])
        self.vit_norm = nn.LayerNorm(embed_dim)
        self.dropout  = nn.Dropout(dropout)
        self.proj_out = nn.Linear(embed_dim, conv_channels[-1] * patch_size ** 2)
        self.gradient_checkpointing = False

        # ----- 上采样路径 -----
        # 每个阶段：Upsample，然后 ResBlock × 2（接收 skip，通道翻倍）
        self.up_layers = nn.ModuleList()
        in_ch_up = conv_channels[-1]
        for out_ch in reversed(conv_channels):
            self.up_layers.append(Upsample(in_ch_up))
            self.up_layers.append(ResBlock(in_ch_up + out_ch, out_ch, time_emb_dim=time_dim))
            self.up_layers.append(ResBlock(out_ch, out_ch, time_emb_dim=time_dim))
            in_ch_up = out_ch

        self.norm_out = nn.GroupNorm(_num_groups(conv_channels[0]), conv_channels[0])
        self.conv_out = nn.Conv2d(conv_channels[0], out_channels, 3, padding=1)

    def forward(self, x_t: torch.Tensor, z_0: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        B = x_t.shape[0]
        t_emb = self.time_mlp(sinusoidal_embedding(t, self._sin_freqs))

        # ----- 下采样，记录 skip -----
        skips: list[torch.Tensor] = []
        # h = x_t
        h = self.conv_in(x_t)
        stage_size = 3          # ResBlock, ResBlock, Downsample
        n_stages   = len(self.conv_channels)
        for s in range(n_stages):
            base = s * stage_size
            h = self.down_layers[base    ](h, t_emb)   # ResBlock
            h = self.down_layers[base + 1](h, t_emb)   # ResBlock
            skips.append(h)
            h = self.down_layers[base + 2](h)           # Downsample（无 t_emb）

        # ----- ViT -----
        img_tokens, (fh, fw) = self.img_patch_embed(h)
        img_tokens = img_tokens + interpolate_pos_embed(self.img_pos_embed, fh, fw)

        z_tokens, (zh, zw) = self.z_patch_embed(z_0)
        z_tokens    = z_tokens + interpolate_pos_embed(self.z_pos_embed, zh, zw)

        tokens = self.dropout(torch.cat([z_tokens, img_tokens], dim=1))
        for block in self.vit_blocks:
            tokens = _run_block(block, tokens, t_emb, self.gradient_checkpointing and self.training)
        tokens = self.vit_norm(tokens)

        # 丢弃 z prefix，还原图像特征图
        n_z        = z_tokens.shape[1]
        img_tokens = tokens[:, n_z:]
        p          = self.patch_size
        out_ch     = self.conv_channels[-1]
        img_tokens = self.proj_out(img_tokens)           # [B,N,out_ch*p*p]
        h = (img_tokens
             .reshape(B, fh, fw, out_ch, p, p)
             .permute(0, 3, 1, 4, 2, 5)
             .reshape(B, out_ch, fh * p, fw * p))

        # ----- 上采样，接 skip -----
        for s in range(n_stages):
            base = s * stage_size
            skip = skips[n_stages - 1 - s]
            h    = self.up_layers[base    ](h)                   # Upsample
            h    = torch.cat([h, skip], dim=1)
            h    = self.up_layers[base + 1](h, t_emb)            # ResBlock
            h    = self.up_layers[base + 2](h, t_emb)            # ResBlock

        return x_t + self.conv_out(F.silu(self.norm_out(h)))


# ============================================================
# 5. BaseModel（两阶段 ViT，阶段二）
# ============================================================

class BaseModel(nn.Module):
    """
    两阶段 ViT base model（阶段二，sigmoid weighting）。
    Stage1: 512 dim × 6 blocks → Stage2: 1024 dim × 16 blocks。
    输入 z_t [B,C,h,w] + t [B] → 输出 z_hat [B,C,h,w]。
    """
    def __init__(self, latent_channels: int = 32, latent_size: int = 32,
                 stage_dims: tuple = (512, 1024),
                 stage_blocks: tuple = (6, 16),
                 n_heads: tuple = (8, 16),
                 patch_size: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        n_patches       = (latent_size // patch_size) ** 2
        dim1, dim2      = stage_dims
        time_dim        = dim2 * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(dim2, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.register_buffer('_sin_freqs', make_sinusoidal_freqs(dim2), persistent=False)

        # Stage 1
        self.patch_embed1 = PatchEmbed2D(latent_channels, dim1, patch_size)
        self.pos_embed1   = nn.Parameter(torch.zeros(1, n_patches, dim1))
        nn.init.normal_(self.pos_embed1, std=0.02)
        self.blocks1  = nn.ModuleList([
            ViTBlock(dim1, n_heads[0], time_emb_dim=time_dim)
            for _ in range(stage_blocks[0])
        ])
        self.norm1    = nn.LayerNorm(dim1)
        self.dropout1 = nn.Dropout(dropout)

        # Stage 1 → Stage 2
        self.bridge     = nn.Linear(dim1, dim2)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, n_patches, dim2))
        nn.init.normal_(self.pos_embed2, std=0.02)

        # Stage 2
        self.blocks2  = nn.ModuleList([
            ViTBlock(dim2, n_heads[1], time_emb_dim=time_dim)
            for _ in range(stage_blocks[1])
        ])
        self.norm2    = nn.LayerNorm(dim2)
        self.dropout2 = nn.Dropout(dropout)
        self.proj_out = nn.Linear(dim2, latent_channels * patch_size ** 2)
        self.gradient_checkpointing = False

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z_t.shape
        t_emb = self.time_mlp(sinusoidal_embedding(t, self._sin_freqs))

        tokens, (h, w) = self.patch_embed1(z_t)
        tokens = self.dropout1(tokens + interpolate_pos_embed(self.pos_embed1, h, w))
        for block in self.blocks1:
            tokens = _run_block(block, tokens, t_emb, self.gradient_checkpointing and self.training)
        tokens = self.norm1(tokens)

        tokens = self.dropout2(self.bridge(tokens) + interpolate_pos_embed(self.pos_embed2, h, w))
        for block in self.blocks2:
            tokens = _run_block(block, tokens, t_emb, self.gradient_checkpointing and self.training)
        tokens = self.norm2(tokens)

        p      = self.patch_size
        tokens = self.proj_out(tokens)                    # [B,N,C*p*p]
        tokens = tokens.reshape(B, h, w, C, p, p)
        residual = tokens.permute(0, 3, 1, 4, 2, 5).reshape(B, C, h*p, w*p)
        return z_t + residual

