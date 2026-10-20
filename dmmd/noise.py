import torch
from torch import nn


class DDPMNoise(nn.Module):
    """Linear-beta DDPM schedule with interpolation between tabulated times.

    A clean endpoint is prepended. t=0 is exactly clean; t=1 is approximately
    standard Gaussian for the usual 1000-step beta schedule.
    """

    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, **unused):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        alpha_bar = torch.cat((torch.ones(1, dtype=betas.dtype),
                               torch.cumprod(1 - betas, dim=0)))
        self.register_buffer("alpha_bar", alpha_bar.float())

    def coefficients(self, t):
        t = torch.as_tensor(t, device=self.alpha_bar.device, dtype=torch.float32)
        if torch.any((t < 0) | (t > 1)):
            raise ValueError("Noise time must lie in [0,1]")
        location = t * (len(self.alpha_bar) - 1)
        lo = location.floor().long()
        hi = (lo + 1).clamp_max(len(self.alpha_bar) - 1)
        value = torch.lerp(self.alpha_bar[lo], self.alpha_bar[hi], location - lo)
        return value.sqrt(), (1 - value).clamp_min(0).sqrt()

    def forward(self, clean, t, eps=None):
        eps = torch.randn_like(clean) if eps is None else eps
        alpha, sigma = self.coefficients(t)
        if alpha.numel() != 1:
            alpha, sigma = alpha[:, None, None, None], sigma[:, None, None, None]
        return alpha * clean + sigma * eps
