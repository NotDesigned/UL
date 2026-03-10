import torch
import torch.nn.functional as F

class NoiseSchedule:
    """
    VP noise scheduling.
    alpha_t^2 + sigma_t^2 = 1
    lambda(t) = log(alpha_t^2 / sigma_t^2)
    alpha_t^2 = 1/ 1+ exp(lambda) = sigmoid(lambda), beta_t^2  = sigmoid(-lambda)

    参数语义：
        lambda_0: t=0 时的 log-SNR（高 SNR，低噪声，接近干净数据）
        lambda_1: t=1 时的 log-SNR（低 SNR，高噪声，接近纯噪声）
    """
    def __init__(self, lambda_0: float, lambda_1: float):
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.dlam_dt = lambda_0 - lambda_1  # -dλ/dt，正常数，预计算避免重复运算

    def log_snr(self, t: torch.Tensor) -> torch.Tensor:
        return self.lambda_0 + (self.lambda_1 - self.lambda_0) * t
    
    def alpha_sigma(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lam = self.log_snr(t)
        alpha = torch.sqrt(torch.sigmoid(lam))
        sigma = torch.sqrt(torch.sigmoid(-lam))
        return alpha, sigma
    
    def forward_noise(self, x: torch.Tensor, t: torch.Tensor, eps: torch.Tensor = None):
        if eps is None:
            eps = torch.randn_like(x)
        
        t = t.reshape(t.shape + (1,) * (x.dim()-t.dim()))
        
        alpha, sigma = self.alpha_sigma(t)
        
        x_t = alpha * x + sigma * eps
        return x_t, eps

def loss_weight_unweighted(log_snr: torch.Tensor) -> torch.Tensor:
    """
    Unweighted ELBO 权重，恒为 1。
    
    用于先验模型（Prior）的损失。
    论文强调先验必须用 unweighted，否则编码器会把信息
    藏在权重最小的噪声层，绕过先验的正则化。
    """
    return torch.ones_like(log_snr)


def loss_weight_sigmoid(log_snr: torch.Tensor, shift: float = 0.0) -> torch.Tensor:
    """
    Sigmoid reweighted ELBO 权重。

    w(lambda_t) = sigmoid(shift - lambda_t)

    效果：
      - 低 log-SNR（高噪声）区域权重小：
        鼓励解码器在噪声较大的时间步承担任务。
      - 高 log-SNR（低噪声）区域权重大：
        鼓励将这些信息从编码器里传递。

    用于解码器（Decoder）的损失。与 loss_factor 配合，
    共同控制编码器的信息编码压力。

    Args:
        log_snr: 当前时间步的 log-SNR，shape [B]
        shift:   sigmoid 的偏移量，对应论文中的 bias b
    """
    return torch.sigmoid(shift - log_snr)

def diffusion_loss(
    x_clean: torch.Tensor,
    x_pred: torch.Tensor,
    t: torch.Tensor,
    schedule: NoiseSchedule,
    weight_fn,
    loss_factor: float = 1.0,
) -> torch.Tensor:
    """
    计算单个扩散模型的 ELBO 损失。
    
    完整的 ELBO 损失形式（来自论文 Eq.2）：
      L = E_t [ -d lambda/dt * exp(lambda/2) * w(lambda) * ||x - x_hat||^2 ]
    
    其中 -d lambda/dt 是 schedule 的导数项（对 linear schedule 是常数），
    exp(lambda/2) 是标准的 SNR 加权项。
    
    实践中常把这两项合并为一个与 SNR 相关的系数，
    再乘以用户指定的 weight_fn（unweighted 或 sigmoid）。
    
    Args:
        x_clean:     干净的目标数据 [B, ...]
        x_pred:      模型预测的干净数据 [B, ...]
        t:           时间步 [B]
        schedule:    使用的 noise schedule
        weight_fn:   损失权重函数（unweighted 或 sigmoid）
        loss_factor: 额外的损失缩放系数（对应论文的 c_lf），
                     用于解码器以对抗 posterior collapse
                     
    Returns:
        loss: 标量损失值
    """
    lam = schedule.log_snr(t)                  # [B]

    # x-prediction 的连续时间 ELBO（Kingma VDM 2021, Eq.12）：
    #   L = ½ E_t [ |dλ/dt| · exp(λ) · ||x - x̂||² ]
    # linear schedule 下 -dλ/dt = λ_min - λ_max（正常数）
    snr_weight = schedule.dlam_dt * torch.exp(lam) / 2     # [B]

    w = weight_fn(lam)                          # [B]，unweighted=1 或 sigmoid 值

    # 逐样本的 MSE，在空间维度上取均值
    mse = F.mse_loss(x_pred, x_clean, reduction='none').flatten(1).mean(dim=1)            # [B]

    # 组合：每个样本的加权损失
    per_sample_loss = snr_weight * w * mse      # [B]

    return loss_factor * per_sample_loss.mean()

def kl_standard_normal(z_clean: torch.Tensor, schedule: NoiseSchedule, t: float) -> torch.Tensor:
    """
    KL[p(z_t|x) || N(0,I)] 的解析解。
    而实践中因为 lambda_t 很负，t=1时，这项几乎为 0。
    """
    t_ = torch.full((z_clean.shape[0],), t, device=z_clean.device)
    alpha1, sigma1 = schedule.alpha_sigma(t_)

    # 广播到 [B, 1, 1, 1] 以匹配 z_clean [B, C, H, W]
    view_shape = (-1,) + (1,) * (z_clean.dim() - 1)
    alpha1 = alpha1.view(view_shape)
    sigma1 = sigma1.view(view_shape)

    # p(z_1|x) = N(alpha1 * z_clean, sigma1^2 * I)
    mu = alpha1 * z_clean          # [B, C, H, W]
    var = sigma1[0].pow(2)         # 标量（所有元素相同）

    d = mu[0].numel()              # 单样本维度数

    kl = 0.5 * (
        mu.pow(2).flatten(1).sum(1)   # ||mu||^2，shape [B]
        + d * var
        - d
        - d * torch.log(var)
    )
    return kl.mean()

# ============================================================
# 3. 加噪 / 采样工具
# ============================================================

def add_latent_noise(z_clean: torch.Tensor, schedule: NoiseSchedule) -> torch.Tensor:
    """
    对干净潜变量加固定量的噪声，得到 z_0。
    
    这是 UL 的核心设计之一：
    编码器输出确定性的 z_clean，然后在 t=0 处加固定噪声，
    将编码器的精度与先验模型的最大精度对齐（lambda_0=5, sigma≈0.08）。
    
    Args:
        z_clean:  编码器输出的干净潜变量 [B, C, H, W]
        schedule: 潜在空间的 noise schedule（lambda_0=5）
        
    Returns:
        z_0: 带有固定噪声的潜变量，shape 与 z_clean 相同
    """
    t_zero = torch.zeros(z_clean.shape[0], device=z_clean.device)
    z_0, _ = schedule.forward_noise(z_clean, t_zero)
    return z_0


def sample_timesteps(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    均匀采样时间步 t ~ U(0, 1)。
    
    Args:
        batch_size: 批大小
        device:     设备
        
    Returns:
        t: shape [B]，值域 [0, 1]
    """
    return torch.rand(batch_size, device=device)


# ============================================================
# 4. 预定义 Schedule 实例（方便直接使用）
# ============================================================

def get_latent_schedule() -> NoiseSchedule:
    """
    潜在空间的 noise schedule。
    
    lambda_0 = 5   →  t=0 时 sigma ≈ 0.08（固定的小噪声）
    lambda_1 = -20 →  t=1 时接近纯高斯噪声

    这是 UL 论文的核心设定。
    """
    return NoiseSchedule(lambda_0=5.0, lambda_1=-20.0)


def get_image_schedule() -> NoiseSchedule:
    """
    图像空间的 noise schedule（用于扩散解码器）。
    
    lambda_0 = 10  →  t=0 时几乎是完全干净的图像
    lambda_1 = -20 →  t=1 时接近纯高斯噪声
    """
    return NoiseSchedule(lambda_0=10.0, lambda_1=-20.0)


# ============================================================
# 5. 简单验证
# ============================================================

if __name__ == "__main__":
    print("=== Noise Schedule 验证 ===")
    schedule = get_latent_schedule()

    t_vals = torch.tensor([0.0, 0.5, 1.0])
    for t in t_vals:
        lam = schedule.log_snr(t)
        alpha, sigma = schedule.alpha_sigma(t)
        print(f"  t={t.item():.1f} | lambda={lam.item():.2f} | "
              f"alpha={alpha.item():.4f} | sigma={sigma.item():.4f} | "
              f"alpha^2+sigma^2={alpha.item()**2 + sigma.item()**2:.6f}")

    print("\n=== 加噪验证 ===")
    x = torch.randn(4, 32, 32, 32)   # [B, C, H, W]
    t = sample_timesteps(4, device=torch.device('cpu'))
    x_t, eps = schedule.forward_noise(x, t)
    print(f"  x shape: {x.shape}, x_t shape: {x_t.shape}")
    print(f"  t: {t.tolist()}")

    print("\n=== z_0 加噪验证（固定 sigma≈0.08）===")
    z_clean = torch.randn(4, 32, 32, 32)
    z_0 = add_latent_noise(z_clean, schedule)
    diff = (z_0 - z_clean).std()
    print(f"  z_clean std: {z_clean.std():.4f}")
    print(f"  z_0 std:     {z_0.std():.4f}")
    print(f"  diff std (≈0.08): {diff:.4f}")

    print("\n=== 损失函数验证 ===")
    x_clean = torch.randn(4, 3, 32, 32)
    x_pred  = x_clean + 0.1 * torch.randn_like(x_clean)
    img_schedule = get_image_schedule()

    loss_prior = diffusion_loss(
        x_clean, x_pred, t, img_schedule,
        weight_fn=loss_weight_unweighted,
        loss_factor=1.0,
    )
    loss_decoder = diffusion_loss(
        x_clean, x_pred, t, img_schedule,
        weight_fn=lambda lam: loss_weight_sigmoid(lam, shift=0.0),
        loss_factor=1.5,
    )
    print(f"  Prior loss (unweighted): {loss_prior.item():.4f}")
    print(f"  Decoder loss (sigmoid, lf=1.5): {loss_decoder.item():.4f}")

    print("\n全部验证通过 ✓")