"""Runs through the OFFICIAL ELF generation code paths (generation.py), for gates and cross-checks."""
import glob
import json
import os
import time

import torch

from .registry import append_registry, is_done, load_json, run_dir, save_json_atomic
from . import RUNS


def _collect_metrics(off_dir):
    res = []
    for mp in sorted(glob.glob(os.path.join(off_dir, "*", "metrics.jsonl"))):
        with open(mp) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        if lines:
            res.append(dict(name=os.path.basename(os.path.dirname(mp)), **lines[-1]))
    return res


def _seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_cond_official(name, run_id, n=1000, batch_size=64, seed=42, write_registry=True):
    """XSum / WMT14 de-en evaluation via generation.test_generation_cond (eval.py construction).

    Sampling configs come from the model YAML's sampling_configs_path (cond_sampling_configs.yml:
    64-step ODE, CFG=2, SC-CFG=1). Uses the first n validation examples.
    """
    if is_done(run_id):
        s = load_json(os.path.join(RUNS, run_id, "summary.json"))
        print(f"[skip] {run_id}: ROUGE-L={s.get('rougeL')} BLEU={s.get('bleu')}")
        return s
    from generation import test_generation_cond
    from utils.data_utils import load_dataset_split

    from .load import load_elf
    from .sweep import unload_all

    unload_all()
    model, encoder, tokenizer, config, state = load_elf(name, return_state=True)
    del model
    rd = run_dir(run_id)
    off = os.path.join(rd, "official")
    config.output_dir, config.num_samples, config.online_eval = off, n, True
    config.use_compile, config.use_wandb, config.hf_repo_id = False, False, None
    ds = load_dataset_split(config.eval_data_path)
    print(f"[cond] {name}: eval set {config.eval_data_path} ({len(ds)} rows); using first {n}")
    save_json_atomic(os.path.join(rd, "config.json"),
                     dict(run_id=run_id, model=name, kind="cond", n=n, seed=seed, batch_size=batch_size,
                          sampling_configs=[vars(s) for s in config.sampling_configs]))
    gen = torch.Generator(device="cpu").manual_seed(seed)  # eval.py: CPU generator for cond
    _seed_all(seed)
    t0 = time.time()
    for sc in config.sampling_configs:
        test_generation_cond(state=state, encoder=encoder, tokenizer=tokenizer, generator=gen,
                             config=config, sampling_config=sc, dataset=ds,
                             num_samples=n, batch_size=batch_size)
    wall = time.time() - t0
    res = _collect_metrics(off)
    if not res:
        raise RuntimeError(f"No metrics.jsonl produced under {off}")
    sc0, r0 = config.sampling_configs[0], res[0]
    summary = dict(run_id=run_id, model=name, kind="cond", method=sc0.sampling_method,
                   steps=sc0.num_sampling_steps[0], gamma=getattr(sc0, "sde_gamma", 0.0),
                   cfg=sc0.cfgs[0], sc_cfg=sc0.self_cond_cfg_scales[0], seed=seed, n=n,
                   rouge1=r0["rouge1"], rouge2=r0["rouge2"], rougeL=r0["rougeL"], bleu=r0["bleu"],
                   all_results=res, wall_s=wall, status="done", timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    save_json_atomic(os.path.join(rd, "summary.json"), summary)
    if write_registry:
        append_registry(summary)
    print(f"[done] {run_id}: R1={r0['rouge1']:.2f} R2={r0['rouge2']:.2f} RL={r0['rougeL']:.2f} "
          f"BLEU={r0['bleu']:.2f} ({wall:.0f}s)")
    del state, encoder
    torch.cuda.empty_cache()
    return summary


def run_uncond_official(name, run_id, method="sde", steps=32, gamma=1.5, sc_cfg=3.0, n=1000,
                        batch_size=64, seed=0, write_registry=False):
    """Cross-check: generation.test_generation_uncond end-to-end (its own sampler + official PPL).

    NOTE: eval.py passes a CPU torch.Generator; on CUDA, _sde_step then calls
    torch.randn(..., generator=<cpu>, device='cuda'), which raises. We pass a CUDA generator.
    """
    if is_done(run_id):
        return load_json(os.path.join(RUNS, run_id, "summary.json"))
    from configs.config import SamplingConfig
    from generation import test_generation_uncond

    from .load import load_elf
    from .sweep import unload_all

    unload_all()
    model, encoder, tokenizer, config, state = load_elf(name, return_state=True)
    del model, encoder
    rd = run_dir(run_id)
    off = os.path.join(rd, "official")
    config.output_dir, config.num_samples, config.online_eval = off, n, True
    config.use_compile, config.use_wandb, config.hf_repo_id = False, False, None
    sc = SamplingConfig(sampling_method=method, num_sampling_steps=[steps], cfgs=[1], sde_gamma=gamma,
                        self_cond_cfg_scales=[sc_cfg], time_schedule="logit_normal")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=dev).manual_seed(seed)
    _seed_all(seed)
    t0 = time.time()
    test_generation_uncond(state=state, tokenizer=tokenizer, generator=gen, config=config,
                           sampling_config=sc, num_samples=n, batch_size=batch_size)
    wall = time.time() - t0
    res = _collect_metrics(off)
    if not res:
        raise RuntimeError(f"No metrics.jsonl produced under {off}")
    texts = []
    for p in glob.glob(os.path.join(off, "*", "all_generated_*.jsonl")):
        with open(p) as f:
            texts = [json.loads(l)["generated"] for l in f]
    summary = dict(run_id=run_id, model=name, kind="uncond_official", method=method, steps=steps,
                   gamma=gamma, sc_cfg=sc_cfg, seed=seed, n=n, gen_ppl=res[0]["ppl"],
                   entropy=res[0]["mean_entropy"], wall_s=wall, status="done",
                   timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if texts:
        from .metrics import corpus_ppl, score_gpt2
        s = score_gpt2(texts)
        summary["elfx_rescored_ppl"] = corpus_ppl(s["nll"], s["count"])
    save_json_atomic(os.path.join(rd, "summary.json"), summary)
    if write_registry:
        append_registry(summary)
    print(f"[done] {run_id}: official PPL={summary['gen_ppl']:.2f} H={summary['entropy']:.3f} "
          f"(elfx rescoring of the same texts: {summary.get('elfx_rescored_ppl', float('nan')):.2f}) {wall:.0f}s")
    del state
    torch.cuda.empty_cache()
    return summary
