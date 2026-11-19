"""Sampling for ELF with per-sample reproducible randomness and an optional per-step hook.

Math is identical to ELF/src/utils/generation_utils._generate_samples_single_batch and
utils/sampling_utils._{ode,sde}_step (the network call itself IS the official
_forward_sample), with two generalizations:
  1. t can differ per sample (each sample draws its own logit-normal schedule), which makes
     results independent of batch size.
  2. all randomness (initial noise, schedule, SDE noise) is drawn per sample index from
     generators seeded by (seed, index), using sha256 (Python's hash() is salted per process).

The last step is always an ODE step (as in the official sampler). Decoding uses the official
_dlm_decode_batch on the final latent z, then mask_after_eos, then tokenizer.decode.
"""
import hashlib
import time

import numpy as np
import torch
import torch.nn as nn

from . import ensure_elf_on_path

ensure_elf_on_path()
from utils.data_utils import get_pad_token_id  # noqa: E402
from utils.generation_utils import _dlm_decode_batch, mask_after_eos  # noqa: E402
from utils.sampling_utils import _forward_sample  # noqa: E402

# Global counter of T5 encoder forward calls; hooks that re-encode must call count_enc(k).
n_enc_fwd = 0


def count_enc(k=1):
    global n_enc_fwd
    n_enc_fwd += int(k)


def seed_int(*parts):
    """Stable 63-bit seed from arbitrary parts (process-independent, unlike hash())."""
    h = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


class CounterWrapper(nn.Module):
    """Counts batched forward calls of the ELF network."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.count = 0

    def forward(self, *args, **kwargs):
        self.count += 1
        return self.model(*args, **kwargs)


def unwrap(model):
    seen = set()
    while id(model) not in seen:
        seen.add(id(model))
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        elif isinstance(model, CounterWrapper):
            model = model.model
        else:
            break
    return model


def _device_of(model):
    return next(unwrap(model).parameters()).device


def initial_noise(indices, seed, L, D, noise_scale, device, dtype=torch.float32):
    """z_j = randn(L, D) * noise_scale from Generator(seed*100003 + j)."""
    zs = []
    for j in indices:
        g = torch.Generator(device=device).manual_seed(int(seed) * 100003 + int(j))
        zs.append(torch.randn((L, D), generator=g, device=device, dtype=dtype))
    return torch.stack(zs) * noise_scale


def sample_schedules(indices, seed, n_steps, time_schedule, p_mean, p_std, device, dtype=torch.float32):
    """Per-sample version of utils.sampling_utils.get_sampling_steps -> (B, n_steps+1)."""
    if time_schedule == "uniform":
        t = torch.linspace(0.0, 1.0, n_steps + 1, dtype=dtype)
        return t.unsqueeze(0).repeat(len(indices), 1).to(device)
    if time_schedule not in ("logit_normal", "logit_normal_oversampled"):
        raise ValueError(f"Unsupported time_schedule {time_schedule!r}")
    if time_schedule == "logit_normal_oversampled":
        raise NotImplementedError("logit_normal_oversampled is a training schedule; not used for sampling")
    rows = []
    for j in indices:
        g = torch.Generator(device="cpu").manual_seed(seed_int(seed, int(j), "sched"))
        s = torch.sigmoid(torch.randn((n_steps - 1,), generator=g, dtype=dtype) * p_std + p_mean)
        s = torch.sort(s).values
        rows.append(torch.cat([torch.zeros(1, dtype=dtype), s, torch.ones(1, dtype=dtype)]))
    return torch.stack(rows).to(device)


class PerSampleNoise:
    """SDE noise: eps_j drawn from a per-sample generator seeded by (seed, j, 'sde')."""

    def __init__(self, indices, seed, shape_LD, device, dtype=torch.float32):
        self.gens = [torch.Generator(device=device).manual_seed(seed_int(seed, int(j), "sde")) for j in indices]
        self.shape, self.device, self.dtype = shape_LD, device, dtype

    def __call__(self, step_idx):
        return torch.stack([torch.randn(self.shape, generator=g, device=self.device, dtype=self.dtype)
                            for g in self.gens])


@torch.no_grad()
def rollout(model, z, t_steps, config, method, gamma, sc_cfg, eps_fn=None, hook=None):
    """Euler / SDE rollout. t_steps: (B, n+1) or (n+1,). Returns the final latent z.

    hook(i, t, t_next, t_eff, z, x_pred) -> (z, x_pred) is called after every NON-final step.
    t, t_next, t_eff are (B,) tensors; t_eff = t for ODE and t_back for SDE steps. The returned
    x_pred becomes the self-conditioning input of the next step.
    """
    B, L, D = z.shape
    shared_t = t_steps.dim() == 1
    if shared_t:
        t_1d = t_steps.to(device=z.device, dtype=z.dtype)
        t_steps = t_1d.unsqueeze(0).expand(B, -1)
    else:
        t_steps = t_steps.to(device=z.device, dtype=z.dtype)
    if method not in ("ode", "sde"):
        raise ValueError(f"Invalid sampling method: {method}")
    if method == "sde" and eps_fn is None:
        raise ValueError("SDE sampling needs eps_fn")
    cond_seq = torch.zeros_like(z)
    cond_seq_mask = torch.zeros((B, L), dtype=z.dtype, device=z.device)
    x_pred = torch.zeros_like(z)  # == restore_cond(zeros, zeros, zeros) in the official sampler
    n = t_steps.shape[1]
    use_bf16 = bool(getattr(config, "use_bf16", True)) and z.is_cuda
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        for i in range(n - 1):
            t, t_next = t_steps[:, i], t_steps[:, i + 1]
            last = i == n - 2
            if shared_t:
                ti, tni = t_1d[i].item(), t_1d[i + 1].item()
                if method == "sde" and not last:
                    h = float(tni - ti)
                    alpha = max(0.0, min(1.0, 1.0 - float(gamma) * h))
                    t_back = alpha * float(ti)
                    eps = eps_fn(i) * config.denoiser_noise_scale
                    z_in = alpha * z + (1.0 - alpha) * eps
                    t_eff = torch.full((B,), t_back, dtype=z.dtype, device=z.device)
                    dt = tni - t_back
                else:
                    z_in = z
                    t_eff = torch.full((B,), float(ti), dtype=z.dtype, device=z.device)
                    dt = tni - ti
            elif method == "sde" and not last:
                t_d, tn_d = t.double(), t_next.double()
                h_d = tn_d - t_d
                alpha_d = (1.0 - float(gamma) * h_d).clamp(0.0, 1.0)
                te_d = alpha_d * t_d
                t_eff = te_d.to(z.dtype)
                eps = eps_fn(i) * config.denoiser_noise_scale
                a = alpha_d.to(z.dtype).view(-1, 1, 1)
                one_m_a = (1.0 - alpha_d).to(z.dtype).view(-1, 1, 1)
                z_in = a * z + one_m_a * eps
                dt = (tn_d - te_d).to(z.dtype).view(-1, 1, 1)
            else:
                t_eff, z_in = t, z
                dt = (t_next - t_eff).view(-1, 1, 1)
            v, x_pred = _forward_sample(
                model=model, z=z_in, t_batch=t_eff, x_pred_prev=x_pred, config=config,
                cfg_scale=1.0, self_cond_cfg_scale=sc_cfg,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
            )
            z = z_in + dt * v
            if hook is not None and not last:
                z, x_pred = hook(i, t, t_next, t_eff, z, x_pred)
    return z


@torch.no_grad()
def decode_latents(model, z, config, sc_cfg, chunk=32):
    """Official _dlm_decode_batch in chunks (the (B, L, vocab) fp32 logits dominate memory)."""
    out = [_dlm_decode_batch(z=z[s:s + chunk], model=model, t_final_val=1.0, config=config,
                             self_cond_cfg_scale=sc_cfg) for s in range(0, z.shape[0], chunk)]
    return torch.cat(out, 0)


@torch.no_grad()
def generate(model, config, tokenizer, n, seed, steps, method="sde", gamma=1.5, sc_cfg=3.0,
             batch_size=64, hook=None, return_traj=False, time_schedule=None, index_offset=0,
             pad_batches=True, decode_chunk=32, progress=False):
    """Unconditional generation.

    Returns dict(texts, token_ids (n, L) int32, n_elf_fwd, nfe_elf (per sample), n_enc_fwd,
    wall_s, traj (optional)). Same seed => same samples regardless of batch_size.
    """
    global n_enc_fwd
    device = _device_of(model)
    base = unwrap(model)
    L, D = int(config.max_length), int(base.text_encoder_dim)
    ts = time_schedule or getattr(config, "time_schedule", "logit_normal")
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
    pad_id = get_pad_token_id(tokenizer)
    counter = CounterWrapper(model)
    enc0 = n_enc_fwd
    t0 = time.time()
    texts, ids_all, traj = [], [], []
    n_batches = 0
    for start in range(0, n, batch_size):
        idx = list(range(index_offset + start, index_offset + min(start + batch_size, n)))
        real = len(idx)
        if pad_batches and real < batch_size and n > batch_size:
            # Pad to the full batch so torch.compile does not recompile; rows are independent.
            idx = idx + [10**9 + k for k in range(batch_size - real)]
        z = initial_noise(idx, seed, L, D, config.denoiser_noise_scale, device)
        t_steps = sample_schedules(idx, seed, steps, ts, config.denoiser_p_mean, config.denoiser_p_std, device)
        eps_fn = PerSampleNoise(idx, seed, (L, D), device) if method == "sde" else None
        h = hook
        if return_traj and start == 0:
            def h(i, t, tn, te, zz, xp, _inner=hook):
                if _inner is not None:
                    zz, xp = _inner(i, t, tn, te, zz, xp)
                traj.append(dict(i=i, t=t[:4].float().cpu(), x_pred=xp[:4].to(torch.float16).cpu()))
                return zz, xp
        zf = rollout(counter, z, t_steps, config, method, gamma, sc_cfg, eps_fn=eps_fn, hook=h)
        if not torch.isfinite(zf).all():
            raise FloatingPointError(f"non-finite latent in batch starting at {start}")
        pred = decode_latents(counter, zf, config, sc_cfg, chunk=decode_chunk)
        pred = mask_after_eos(pred, eos_token_id=eos_id, pad_token_id=pad_id)[:real]
        ids_all.append(pred.to(torch.int32).cpu().numpy())
        texts.extend(tokenizer.decode(r.tolist(), skip_special_tokens=True) for r in pred.cpu())
        n_batches += 1
        if progress:
            print(f"  generated {start + real}/{n}", end="\r")
    if device.type == "cuda":
        torch.cuda.synchronize()
    out = dict(texts=texts, token_ids=np.concatenate(ids_all, 0) if ids_all else np.zeros((0, L), np.int32),
               n_elf_fwd=counter.count, nfe_elf=counter.count / max(n_batches, 1),
               n_enc_fwd=n_enc_fwd - enc0, wall_s=time.time() - t0)
    if return_traj:
        out["traj"] = traj
    return out


def maybe_compile(model, config, enabled=True, warmup_batch=8):
    """torch.compile(model, dynamic=False) with eager fallback. Returns (model, compile_ok)."""
    if not enabled or not torch.cuda.is_available():
        return model, False
    try:
        from torch import _dynamo
        _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 64)
        cm = torch.compile(model, dynamic=False)
        base = unwrap(model)
        L, D = int(config.max_length), int(base.text_encoder_dim)
        z = torch.randn((warmup_batch, L, D), device="cuda") * config.denoiser_noise_scale
        rollout(cm, z, torch.linspace(0, 1, 3, device="cuda"), config, "ode", 0.0, 3.0)
        decode_latents(cm, z, config, 3.0, chunk=warmup_batch)
        torch.cuda.synchronize()
        return cm, True
    except Exception as e:  # pragma: no cover
        print(f"[maybe_compile] torch.compile failed -> eager. {type(e).__name__}: {str(e)[:300]}")
        return model, False


def _is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def find_max_batch(model, config, candidates=(256, 128, 64, 32, 16, 8, 4), method="sde", decode_chunk=32):
    """Largest batch that survives a 3-step rollout + decode at L=config.max_length."""
    import gc
    base = unwrap(model)
    L, D = int(config.max_length), int(base.text_encoder_dim)
    dev = _device_of(model)
    for b in candidates:
        z = zf = None
        try:
            torch.cuda.empty_cache()
            z = torch.randn((b, L, D), device=dev) * config.denoiser_noise_scale
            eps = (lambda i, _b=b: torch.randn((_b, L, D), device=dev)) if method == "sde" else None
            zf = rollout(model, z, torch.linspace(0, 1, 4, device=dev), config, method, 1.0, 3.0, eps_fn=eps)
            decode_latents(model, zf, config, 3.0, chunk=decode_chunk)
            torch.cuda.synchronize()
            del z, zf
            torch.cuda.empty_cache()
            return b
        except Exception as e:
            is_oom = _is_oom(e)
            e.__traceback__ = None
            del z, zf, e
            gc.collect()
            torch.cuda.empty_cache()
            if not is_oom:
                raise
    raise RuntimeError("No batch size fits in memory")
