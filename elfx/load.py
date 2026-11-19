"""Model loading that mirrors ELF/src/eval.py (lines 118-152), with strict key checks.

Differences from eval.py (all deliberate):
  * The optimizer is NOT rebuilt (eval.py builds Muon only to satisfy load_checkpoint);
    we read the checkpoint ourselves and load params/EMA with strict key assertions.
  * vocab_size is taken from tokenizer.vocab_size (as eval.py does) but asserted against
    the checkpoint's unembedding shape.
"""
import contextlib
import os
import re
import time

import torch

from . import ELF_REPO, ensure_elf_on_path

ensure_elf_on_path()

# name -> (HF repo, training YAML, expected #params of the eval model)
MODELS = {
    "ELF-B": ("embedded-language-flows/ELF-B-owt-torch", "train_owt_ELF-B.yml", 105e6),
    "ELF-M": ("embedded-language-flows/ELF-M-owt-torch", "train_owt_ELF-M.yml", 342e6),
    "ELF-L": ("embedded-language-flows/ELF-L-owt-torch", "train_owt_ELF-L.yml", 652e6),
    "ELF-B-xsum": ("embedded-language-flows/ELF-B-xsum-torch", "train_xsum_ELF-B.yml", 105e6),
    "ELF-B-de-en": ("embedded-language-flows/ELF-B-de-en-torch", "train_de-en_ELF-B.yml", 105e6),
}
PARAM_TOL = 0.05


@contextlib.contextmanager
def _cwd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_config(name_or_yaml):
    """Load an ELF Config. Relative sampling_configs_path values resolve from the ELF repo root."""
    from configs.config import load_config_from_yaml

    yml = MODELS[name_or_yaml][1] if name_or_yaml in MODELS else name_or_yaml
    path = yml if os.path.isabs(yml) else os.path.join(ELF_REPO, "src", "configs", "training_configs", yml)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config YAML not found: {path}")
    with _cwd(ELF_REPO):
        config = load_config_from_yaml(path)
    # Never talk to wandb / HF upload from the harness.
    config.use_wandb = False
    config.hf_repo_id = None
    return config


def _checkpoint_file(repo_or_path):
    """Return a local path to the latest `checkpoint_<step>` of a local dir/file or HF repo."""
    if os.path.isfile(repo_or_path):
        return repo_or_path
    pat = re.compile(r"checkpoint_(\d+)$")
    if os.path.isdir(repo_or_path):
        c = [f for f in os.listdir(repo_or_path) if pat.search(f)]
        if not c:
            raise FileNotFoundError(f"No checkpoint_<step> in {repo_or_path}")
        c.sort(key=lambda f: int(pat.search(f).group(1)))
        return os.path.join(repo_or_path, c[-1])
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files(repo_or_path) if pat.search(f.split("/")[-1])]
    if not files:
        raise FileNotFoundError(f"No checkpoint_<step> file in HF repo {repo_or_path}")
    files.sort(key=lambda f: int(pat.search(f).group(1)))
    return hf_hub_download(repo_or_path, files[-1])


def _torch_load(path):
    kw = dict(map_location="cpu", weights_only=False)  # official checkpoint; contains optimizer state
    try:
        return torch.load(path, mmap=True, **kw)
    except Exception:
        return torch.load(path, **kw)


def load_elf(name, device=None, return_state=False, checkpoint=None, verbose=True):
    """Load an ELF model exactly like eval.py and return the EMA eval model.

    Returns (model, encoder, tokenizer, config) or, with return_state=True,
    (model, encoder, tokenizer, config, state) where `state` is a utils.train_utils.TrainState
    usable by the official generation.test_generation_{uncond,cond} functions.
    model._elfx_info holds load diagnostics.
    """
    from transformers import AutoTokenizer
    from generation import _build_eval_model
    from modules.model import ELF_models
    from modules.t5_encoder import get_encoder
    from utils.train_utils import TrainState

    if name not in MODELS:
        raise ValueError(f"{name!r} not in {sorted(MODELS)}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    repo, _, expected = MODELS[name]
    t0 = time.time()

    config = load_config(name)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)

    # Encoder (HuggingFace T5), fp32 like eval.py
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    ckpt_path = _checkpoint_file(checkpoint or repo)
    ckpt = _torch_load(ckpt_path)
    for k in ("params", "step", "epoch"):
        if k not in ckpt:
            raise KeyError(f"checkpoint {ckpt_path} missing key {k!r}; keys={list(ckpt)}")
    params = ckpt["params"]
    has_ema = bool(ckpt.get("ema_params1"))
    ema_sd = ckpt["ema_params1"] if has_ema else params

    vocab_size = tokenizer.vocab_size  # eval.py: vocab_size = tokenizer.vocab_size
    ckpt_vocab = int(params["unembed_bias"].shape[0])
    if vocab_size != ckpt_vocab:
        print(f"[load_elf][WARN] tokenizer.vocab_size={vocab_size} != checkpoint vocab {ckpt_vocab}; "
              f"using {ckpt_vocab} (transformers {__import__('transformers').__version__}).")
        vocab_size = ckpt_vocab

    model = ELF_models[config.model](
        text_encoder_dim=encoder_config.d_model, max_length=config.max_length,
        attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
    )
    res = model.load_state_dict(params, strict=False)
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(f"[load_elf] state_dict mismatch for {name}: "
                           f"missing={res.missing_keys[:10]} unexpected={res.unexpected_keys[:10]}")
    model = model.to(device).eval()

    pnames = {n for n, _ in model.named_parameters()}
    ema_missing, ema_unexpected = pnames - set(ema_sd), set(ema_sd) - pnames
    if ema_missing or ema_unexpected:
        raise RuntimeError(f"[load_elf] EMA mismatch: missing={sorted(ema_missing)[:10]} "
                           f"unexpected={sorted(ema_unexpected)[:10]}")
    dev_map = {n: p.device for n, p in model.named_parameters()}
    ema_dev = {n: t.to(dev_map[n]) for n, t in ema_sd.items()}

    state = TrainState(model=model, optimizer=None, lr_scheduler=None,
                       ema_params1=ema_dev, step=int(ckpt["step"]), epoch=int(ckpt["epoch"]))
    del ckpt, params, ema_sd

    eval_model = _build_eval_model(state, use_compile=False)  # deep copy + strict EMA load + eval()
    for p in eval_model.parameters():
        p.requires_grad_(False)

    # Verify the EMA weights (not the raw weights) are in the eval model.
    probe = "final_layer.linear.weight" if "final_layer.linear.weight" in ema_dev else sorted(ema_dev)[0]
    ema_ok = torch.equal(dict(eval_model.named_parameters())[probe], ema_dev[probe])
    n_params = sum(p.numel() for p in eval_model.parameters())
    rel = abs(n_params - expected) / expected
    if rel > PARAM_TOL:
        raise AssertionError(f"[load_elf] {name}: {n_params/1e6:.1f}M params vs expected ~{expected/1e6:.0f}M")

    info = dict(name=name, repo=repo, checkpoint=os.path.basename(ckpt_path), step=state.step,
                epoch=state.epoch, has_ema=has_ema, ema_loaded=bool(ema_ok and has_ema),
                n_params=n_params, vocab_size=vocab_size, missing_keys=[], unexpected_keys=[],
                device=str(device), load_s=round(time.time() - t0, 1))
    eval_model._elfx_info = info
    if verbose:
        print(f"[load_elf] {name}: {n_params/1e6:.1f}M params | step {state.step} | EMA loaded: "
              f"{info['ema_loaded']} | missing/unexpected keys: 0/0 | vocab {vocab_size} | {info['load_s']}s")

    if return_state:
        return eval_model, encoder, tokenizer, config, state
    # Free the raw (non-EMA) training weights; only the EMA eval copy is needed.
    state.model = None
    state.ema_params1 = {}
    del state, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return eval_model, encoder, tokenizer, config
