"""OpenWebText reference texts for MAUVE / reference-anchor metrics."""
import json
import os

import numpy as np

from . import REFS, ensure_elf_on_path

ensure_elf_on_path()

OWT_REPO = "embedded-language-flows/openwebtext-t5"


def _decode_t5(rows, tokenizer):
    import torch
    from utils.generation_utils import mask_after_eos

    out = []
    for ids, sl in rows:
        ids = np.asarray(ids, dtype=np.int64)
        if sl is not None:
            ids = ids[: int(sl)]
        t = torch.from_numpy(ids)[None]
        t = mask_after_eos(t, eos_token_id=tokenizer.eos_token_id or 1, pad_token_id=tokenizer.pad_token_id or 0)
        out.append(tokenizer.decode(t[0].tolist(), skip_special_tokens=True))
    return out


def get_owt_refs(n=1000, pool=5000, seed=0, tokenizer=None):
    """n reference docs: rows sampled (seed) from the LAST `pool` rows of the last OWT-T5 shard.

    Cached at REFS/owt_ref_{n}.jsonl. Only one ~0.5 GB shard is downloaded.
    """
    path = os.path.join(REFS, f"owt_ref_{n}.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            return [json.loads(l)["text"] for l in f]
    from transformers import AutoTokenizer

    tokenizer = tokenizer or AutoTokenizer.from_pretrained("t5-small")
    try:
        from datasets import Dataset
        from huggingface_hub import hf_hub_download, list_repo_files

        shards = sorted(f for f in list_repo_files(OWT_REPO, repo_type="dataset") if f.endswith(".arrow"))
        local = hf_hub_download(OWT_REPO, shards[-1], repo_type="dataset")
        ds = Dataset.from_file(local)
        N = len(ds)
        rng = np.random.default_rng(seed)
        rows_idx = (N - pool + rng.choice(pool, size=n, replace=False)).tolist()
        sub = ds.select(rows_idx)
        has_len = "sequence_length" in sub.column_names
        rows = [(r["input_ids"], r["sequence_length"] if has_len else None) for r in sub]
        texts = _decode_t5(rows, tokenizer)
        meta = dict(source=f"{OWT_REPO}:{shards[-1]}", rows=rows_idx, n=n, pool=pool, seed=seed)
    except Exception as e:
        print(f"[refs][WARN] OWT-T5 shard path failed ({type(e).__name__}: {e}); trying Skylion007 fallback")
        from datasets import load_dataset

        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        buf = []
        for k, r in enumerate(ds):
            if k >= 200_000 + pool:
                break
            if k >= 200_000:
                buf.append(r["text"])
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(buf), size=n, replace=False)
        texts = []
        for i in pick:
            ids = tokenizer(buf[i], truncation=True, max_length=1024)["input_ids"]
            texts.append(tokenizer.decode(ids, skip_special_tokens=True))
        meta = dict(source="Skylion007/openwebtext[stream 200k:+pool] (fallback)", n=n, pool=pool, seed=seed)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for t in texts:
            f.write(json.dumps({"text": t}) + "\n")
    os.replace(tmp, path)
    with open(path.replace(".jsonl", "_meta.json"), "w") as f:
        json.dump(meta, f)
    return texts


def get_ref_features(n=1000, model_name=None):
    """GPT-2 Large MAUVE features of the references, cached as .npy on Drive."""
    from .metrics import PPL_MODEL, gpt2_features

    tag = (model_name or PPL_MODEL).replace("/", "_")
    path = os.path.join(REFS, f"owt_ref_{n}_{tag}_feats.npy")
    if os.path.exists(path):
        return np.load(path)
    feats = gpt2_features(get_owt_refs(n), model_name=model_name)
    np.save(path, feats)
    return feats
