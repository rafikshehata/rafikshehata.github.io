"""Training loss (DESIGN.md §4): multi-resolution STFT + SI-SDR + howl penalty."""
from __future__ import annotations

import torch
import torch.nn.functional as F

_WIN = {}


def _hann(n, device):
    key = (n, device)
    if key not in _WIN:
        _WIN[key] = torch.hann_window(n, device=device)
    return _WIN[key]


def stft(x, n_fft, hop):
    return torch.stft(x, n_fft, hop, window=_hann(n_fft, x.device), return_complex=True, center=True)


def mrstft(y, s, resolutions):
    total = 0.0
    for n_fft, hop in resolutions:
        S, Y = stft(s, n_fft, hop).abs(), stft(y, n_fft, hop).abs()
        sc = torch.linalg.vector_norm(S - Y, dim=(1, 2)) / (torch.linalg.vector_norm(S, dim=(1, 2)) + 1e-9)
        delta = 1e-3 * S.amax(dim=(1, 2), keepdim=True) + 1e-12
        lm = (torch.log(S + delta) - torch.log(Y + delta)).abs().mean(dim=(1, 2))
        total = total + (sc + lm).mean()
    return total / len(resolutions)


def sisdr(y, s, eps=1e-8):
    """Scale-invariant SDR in dB per item, (B,)."""
    y = y - y.mean(-1, keepdim=True)
    s = s - s.mean(-1, keepdim=True)
    alpha = (y * s).sum(-1, keepdim=True) / ((s * s).sum(-1, keepdim=True) + eps)
    t = alpha * s
    return 10 * torch.log10((t * t).sum(-1) / ((y - t) ** 2).sum(-1).clamp_min(eps) + eps)


def howl_ratio(y, s, n_fft=2048, hop=512, peak_db=6.0, nbhd=31):
    """Energy of narrowband output peaks absent from the target, relative to target energy, (B,)."""
    PY, PS = stft(y, n_fft, hop).abs() ** 2, stft(s, n_fft, hop).abs() ** 2       # (B, F, T)
    eps = 1e-12 + 1e-6 * PS.mean(dim=(1, 2), keepdim=True)
    Bt, Fq, Tt = PY.shape
    AY = F.avg_pool1d(PY.transpose(1, 2).reshape(Bt * Tt, 1, Fq), nbhd, 1, nbhd // 2,
                      count_include_pad=False).reshape(Bt, Tt, Fq).transpose(1, 2)
    w = torch.sigmoid((10 * torch.log10((PY + eps) / (AY + eps)) - peak_db) / 1.5)
    E = F.relu(PY - 2 * PS)
    return (w * E).sum(dim=(1, 2)) / (PS.sum(dim=(1, 2)) + eps.squeeze())


def total_loss(y, s, lc: dict):
    l_mr = mrstft(y, s, lc["mrstft"])
    si = sisdr(y, s)
    l_howl = torch.log1p(howl_ratio(y, s, lc["howl_nfft"], lc["howl_hop"], lc["howl_peak_db"], lc["howl_nbhd"])).mean()
    loss = lc["w_mrstft"] * l_mr + lc["w_sisdr"] * (-si.mean()) + lc["w_howl"] * l_howl
    return loss, {"l_mr": float(l_mr.detach()), "sisdr": float(si.detach().mean()), "l_howl": float(l_howl.detach())}
