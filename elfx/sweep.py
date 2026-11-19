"""Idempotent experiment runner: one cfg dict -> samples, metrics, summary.json, registry row.

cfg keys (uncond): run_id, model, method ('ode'|'sde'), steps, gamma, sc_cfg, seed, n,
                   [batch_size], [hook (name in HOOKS)], [mauve (bool, default True)],
                   [time_schedule], [notes]
A sweep is just a list of cfgs; every cfg persists on completion, so a Colab disconnect loses at
most one cfg. Re-running skips finished cfgs (summary.json exists) and reuses saved samples if
generation finished but scoring did not.
"""
import json
import os
import time

import numpy as np
import torch

from . import RUNS
from .registry import append_registry, is_done, load_json, log_jsonl, run_dir, save_json_atomic

# Named hooks for later notebooks (e.g. ECP). Each entry: name -> factory(model, encoder, tokenizer,
# config, **kwargs) -> hook(i, t, t_next, t_eff, z, x_pred) -> (z, x_pred)
HOOKS = {}

_LOADED = {}
_COMPILE = {"enabled": True}
_BATCH_CACHE = os.path.join(RUNS, "_batch_sizes.json")


def set_compile(enabled):
    _COMPILE["enabled"] = bool(enabled)


def unload_all():
    _LOADED.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_model(name):
    """Load (and cache) an ELF model; keeps at most one model resident to save VRAM."""
    from .load import load_elf
    from .sample import maybe_compile

    if name not in _LOADED:
        unload_all()
        model, encoder, tokenizer, config = load_elf(name)
        cmodel, ok = maybe_compile(model, config, enabled=_COMPILE["enabled"])
        _LOADED[name] = dict(model=cmodel, eager=model, encoder=encoder, tokenizer=tokenizer,
                             config=config, compile_ok=ok)
        print(f"[get_model] {name}: compile_ok={ok}")
    return _LOADED[name]


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def batch_size_for(name, method="sde"):
    """Auto batch size (cached per GPU+model in RUNS/_batch_sizes.json)."""
    from .sample import find_max_batch

    cache = load_json(_BATCH_CACHE, {})
    key = f"{gpu_name()}|{name}"
    if key not in cache:
        m = get_model(name)
        cache[key] = find_max_batch(m["eager"], m["config"], method=method)
        save_json_atomic(_BATCH_CACHE, cache)
        print(f"[batch_size_for] {key} -> {cache[key]}")
    return int(cache[key])


def _save_samples(rd, texts, token_ids):
    log_jsonl(os.path.join(rd, "samples.jsonl"), [{"i": i, "text": t} for i, t in enumerate(texts)])
    np.savez_compressed(os.path.join(rd, "tokens.npz"), token_ids=token_ids)


def _load_samples(rd):
    p = os.path.join(rd, "samples.jsonl")
    if not os.path.exists(p) or not os.path.exists(os.path.join(rd, "gen_meta.json")):
        return None
    with open(p) as f:
        texts = [json.loads(l)["text"] for l in f]
    return texts, load_json(os.path.join(rd, "gen_meta.json"))


def score_texts(texts, mauve=True, ref_feats=None, bootstrap_B=10000):
    """All unconditional metrics for a list of texts."""
    from .metrics import bootstrap_ci, corpus_ppl, gpt2_features, mauve_from_features, score_gpt2, text_stats

    t0 = time.time()
    sc = score_gpt2(texts)
    st = text_stats(texts)
    ppl = corpus_ppl(sc["nll"], sc["count"])
    lo, hi = bootstrap_ci(sc["nll"], sc["count"], B=bootstrap_B)
    mv = float("nan")
    if mauve:
        if ref_feats is None:
            from .refs import get_ref_features
            ref_feats = get_ref_features()
        mv = mauve_from_features(gpt2_features(texts), ref_feats)
    out = dict(gen_ppl=ppl, ppl_ci_lo=lo, ppl_ci_hi=hi,
               entropy=float(np.nanmean(sc["entropy"])), entropy_textstats=st["entropy"],
               mauve=mv, rep4=st["rep4"], distinct1=st["distinct1"], distinct2=st["distinct2"],
               distinct3=st["distinct3"], degen=st["degen"], n_empty=int(sc["empty"].sum()),
               mean_gpt2_len=st["mean_gpt2_len"], score_s=time.time() - t0)
    per_sample = dict(nll=sc["nll"], count=sc["count"], entropy=sc["entropy"], empty=sc["empty"],
                      rep4=st["per_sample_rep4"], degen=st["per_sample_degen"])
    return out, per_sample


def run_config(cfg, verbose=True, write_registry=True):
    """Idempotent: generate -> save samples -> score -> summary.json -> registry row."""
    from .sample import generate

    run_id = cfg["run_id"]
    if is_done(run_id):
        s = load_json(os.path.join(RUNS, run_id, "summary.json"))
        if verbose:
            print(f"[skip] {run_id} (done): PPL={s.get('gen_ppl', float('nan')):.2f} H={s.get('entropy', float('nan')):.3f}")
        return s
    rd = run_dir(run_id)
    save_json_atomic(os.path.join(rd, "config.json"), cfg)
    kind = cfg.get("kind", "uncond")

    if kind == "dummy":  # used by the smoke test (registry idempotency)
        summary = dict(cfg, status="done", timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
        save_json_atomic(os.path.join(rd, "summary.json"), summary)
        return summary

    if kind == "ref":
        from .refs import get_owt_refs
        texts = get_owt_refs(cfg.get("n", 1000))
        gen_meta = dict(n_elf_fwd=0, nfe_elf=0, n_enc_fwd=0, wall_s=0.0)
        mauve = False
    else:
        cached = _load_samples(rd)
        if cached is not None and len(cached[0]) == cfg["n"]:
            texts, gen_meta = cached
            if verbose:
                print(f"[reuse] {run_id}: found {len(texts)} saved samples; scoring only")
        else:
            m = get_model(cfg["model"])
            bs = cfg.get("batch_size") or batch_size_for(cfg["model"], cfg["method"])
            hook = None
            if cfg.get("hook"):
                hook = HOOKS[cfg["hook"]](m["eager"], m["encoder"], m["tokenizer"], m["config"],
                                          **cfg.get("hook_kwargs", {}))
            g = generate(m["model"], m["config"], m["tokenizer"], n=cfg["n"], seed=cfg["seed"],
                         steps=cfg["steps"], method=cfg["method"], gamma=cfg.get("gamma", 0.0),
                         sc_cfg=cfg.get("sc_cfg", 1.0), batch_size=bs, hook=hook,
                         time_schedule=cfg.get("time_schedule"))
            texts = g["texts"]
            gen_meta = dict(n_elf_fwd=g["n_elf_fwd"], nfe_elf=g["nfe_elf"], n_enc_fwd=g["n_enc_fwd"],
                            wall_s=g["wall_s"], batch_size=bs, compile_ok=m["compile_ok"], gpu=gpu_name())
            _save_samples(rd, texts, g["token_ids"])
            save_json_atomic(os.path.join(rd, "gen_meta.json"), gen_meta)
        mauve = cfg.get("mauve", True)

    metrics, per_sample = score_texts(texts, mauve=mauve)
    np.savez_compressed(os.path.join(rd, "per_sample.npz"), **per_sample)
    summary = dict(cfg)
    summary.update(metrics)
    summary.update(gen_meta)
    summary.update(nfe_enc=gen_meta.get("n_enc_fwd", 0), status="done",
                   timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    save_json_atomic(os.path.join(rd, "summary.json"), summary)
    if write_registry:
        append_registry(summary)
    if verbose:
        print(f"[done] {run_id}: PPL={metrics['gen_ppl']:.2f} [{metrics['ppl_ci_lo']:.2f},{metrics['ppl_ci_hi']:.2f}] "
              f"H={metrics['entropy']:.3f} MAUVE={metrics['mauve']:.3f} rep4={metrics['rep4']:.3f} "
              f"degen={metrics['degen']:.3f} empty={metrics['n_empty']} gen={gen_meta.get('wall_s', 0):.0f}s "
              f"score={metrics['score_s']:.0f}s")
    return summary


def run_sweep(cfgs, stop_on_error=False):
    out = []
    for c in cfgs:
        try:
            out.append(run_config(c))
        except Exception as e:
            print(f"[error] {c.get('run_id')}: {type(e).__name__}: {e}")
            if stop_on_error:
                raise
    return out
