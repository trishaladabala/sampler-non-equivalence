"""Metrics that replicate ELF/src/utils/metrics_utils.py exactly, plus extra diagnostics.

* Gen-PPL: CORPUS token-weighted perplexity exp(sum NLL / sum valid tokens), GPT-2 Large in bf16,
  pad = eos, truncation at 1024, first EOS counted, padding masked, batch size 64, empty samples
  dropped before batching -- i.e. exactly what generation.test_generation_uncond does.
  score_gpt2 reuses Metrics._eval_retokenize and Metrics._compute_batch_nlls, so per-sample
  numbers aggregate to the official value.
* Entropy: ELF's per-sample unigram entropy over GPT-2 tokens (nats), averaged over samples.
  NOT decoder softmax entropy.
"""
import math
import os

import numpy as np
import torch

from . import ensure_elf_on_path

ensure_elf_on_path()

PPL_MODEL = os.environ.get("ELFX_PPL_MODEL", "gpt2-large")
# Auto-select GPT-2 Large batch size by VRAM: 16 on <=24GB GPUs (L4/T4), 32 on A100/H100.
_VRAM_GB = (torch.cuda.get_device_properties(0).total_memory / (2 ** 30)) if torch.cuda.is_available() else 0.0
PPL_BATCH = 16 if _VRAM_GB < 35.0 else 32
PPL_MAX_LEN = 1024   # Config.eval_ppl_max_length
_METRICS = {}


def get_ppl_metrics(model_name=None, batch_size=PPL_BATCH, context=PPL_MAX_LEN):
    """An official utils.metrics_utils.Metrics with its eval LM preloaded (bf16 on CUDA)."""
    from transformers import AutoModelForCausalLM
    from utils.metrics_utils import Metrics

    model_name = model_name or PPL_MODEL
    if model_name not in _METRICS:
        m = Metrics(gen_ppl_eval_model_name_or_path=model_name, eval_ppl_batch_size=batch_size,
                    eval_context_size=context)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lm = AutoModelForCausalLM.from_pretrained(model_name)
        lm = lm.to(device=dev, dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32).eval()
        m._eval_model, m._eval_device = lm, dev
        _METRICS[model_name] = m
    m = _METRICS[model_name]
    m.eval_ppl_batch_size = batch_size
    return m


def free_ppl_model():
    import gc
    for m in _METRICS.values():
        m._eval_model = None
    _METRICS.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _nonempty(texts):
    return [i for i, s in enumerate(texts) if isinstance(s, str) and s.strip()]


def _is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


@torch.no_grad()
def score_gpt2(texts, max_length=PPL_MAX_LEN, batch_size=None, model_name=None):
    """Per-sample (nll_sum, token_count, entropy) aligned with `texts`.

    Empty samples get nll=0, count=0, entropy=nan and empty=True (excluded, as officially).
    """
    import gc
    global PPL_BATCH
    batch_size = batch_size or PPL_BATCH
    m = get_ppl_metrics(model_name, batch_size=batch_size, context=max_length)
    n = len(texts)
    keep = _nonempty(texts)
    nll = np.zeros(n, np.float64)
    cnt = np.zeros(n, np.float64)
    ent = np.full(n, np.nan, np.float64)
    if keep:
        sub = [texts[i] for i in keep]
        samples, attn, ctx = m._eval_retokenize(sub, max_length=max_length)
        bs = min(batch_size, samples.shape[0])
        while True:
            try:
                s_nll = np.zeros(len(sub), np.float64)
                s_cnt = np.zeros(len(sub), np.float64)
                for b0 in range(0, samples.shape[0], bs):
                    b1 = min(b0 + bs, samples.shape[0])
                    bsamp, battn = samples[b0:b1], attn[b0:b1]
                    for c0 in range(0, bsamp.shape[1], ctx):
                        c1 = min(c0 + ctx, bsamp.shape[1])
                        ii = torch.from_numpy(bsamp[:, c0:c1]).to(m._eval_device).long()
                        aa = torch.from_numpy(battn[:, c0:c1]).to(m._eval_device).long()
                        nl, vt = m._compute_batch_nlls(ii, aa)
                        nl = nl.detach().cpu().numpy().astype(np.float64)
                        vt = vt.detach().cpu().numpy().astype(np.float64)
                        s_nll[b0:b1] += (nl * vt).sum(-1)
                        s_cnt[b0:b1] += vt.sum(-1)
                break
            except Exception as e:
                is_oom = _is_oom(e)
                e.__traceback__ = None
                del e
                gc.collect()
                torch.cuda.empty_cache()
                if not is_oom or bs == 1:
                    raise
                bs //= 2
                PPL_BATCH = bs
                m.eval_ppl_batch_size = bs
                print(f"[score_gpt2][WARN] OOM -> batch {bs}.")
        # Per-sample unigram entropy, identical to metrics_utils lines 221-228.
        s_ent = np.zeros(len(sub), np.float64)
        for k in range(samples.shape[0]):
            vl = int(attn[k].sum())
            _, counts = np.unique(samples[k, :vl], return_counts=True)
            probs = counts.astype(np.float32) / counts.sum()
            s_ent[k] = float(-np.sum(probs * np.log(probs + 1e-10)))
        nll[keep], cnt[keep], ent[keep] = s_nll, s_cnt, s_ent
    empty = np.ones(n, bool)
    empty[keep] = False
    return dict(nll=nll, count=cnt, entropy=ent, empty=empty)


def corpus_ppl(nll, count):
    nll, count = np.asarray(nll, np.float64), np.asarray(count, np.float64)
    return float(math.exp(nll.sum() / max(count.sum(), 1.0)))


def official_ppl(texts, max_length=PPL_MAX_LEN, model_name=None):
    """Run the official Metrics.record_generative_perplexity (fresh accumulator) for testing."""
    import gc
    from utils.metrics_utils import Metrics

    global PPL_BATCH
    src = get_ppl_metrics(model_name, batch_size=PPL_BATCH)
    bs = int(src.eval_ppl_batch_size or PPL_BATCH)
    keep = [texts[i] for i in _nonempty(texts)]
    while True:
        m = Metrics(gen_ppl_eval_model_name_or_path=src.gen_ppl_eval_model_name_or_path,
                    eval_ppl_batch_size=bs, eval_context_size=max_length)
        m._eval_model, m._eval_device = src._eval_model, src._eval_device
        m.reset()
        try:
            return m.record_generative_perplexity(text_samples=keep, max_length=max_length, retokenize=True)
        except Exception as e:
            is_oom = _is_oom(e)
            e.__traceback__ = None
            del e
            gc.collect()
            torch.cuda.empty_cache()
            if not is_oom or bs <= 1:
                raise
            bs //= 2
            PPL_BATCH = bs
            src.eval_ppl_batch_size = bs
            print(f"[official_ppl][WARN] OOM -> batch {bs}.")


# ---------------------------------------------------------------- text statistics (GPT-2 tokens)
def gpt2_token_ids(texts, max_length=PPL_MAX_LEN, model_name=None):
    tok = get_ppl_metrics(model_name).tokenizer
    return [tok(s, truncation=True, max_length=max_length)["input_ids"] if isinstance(s, str) and s.strip() else []
            for s in texts]


def _ngrams(ids, n):
    return [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]


def unigram_entropy(texts=None, ids=None):
    """Mean per-sample unigram entropy (nats) over non-empty samples (ELF definition)."""
    ids = ids if ids is not None else gpt2_token_ids(texts)
    vals = []
    for x in ids:
        if not x:
            continue
        _, c = np.unique(np.asarray(x), return_counts=True)
        p = c.astype(np.float32) / c.sum()
        vals.append(float(-np.sum(p * np.log(p + 1e-10))))
    return float(np.mean(vals)) if vals else float("nan")


def rep_n_per_sample(ids, n=4):
    out = np.full(len(ids), np.nan)
    for k, x in enumerate(ids):
        g = _ngrams(x, n)
        if g:
            out[k] = 1.0 - len(set(g)) / len(g)
    return out


def rep_n(texts=None, n=4, ids=None):
    ids = ids if ids is not None else gpt2_token_ids(texts)
    return float(np.nanmean(rep_n_per_sample(ids, n)))


def distinct_n(texts=None, n=1, ids=None):
    ids = ids if ids is not None else gpt2_token_ids(texts)
    allg = [g for x in ids for g in _ngrams(x, n)]
    return float(len(set(allg)) / len(allg)) if allg else float("nan")


def degeneracy(texts=None, ids=None):
    """Degenerate if rep-4 > 0.5, or unique/len < 0.15, or < 16 GPT-2 tokens."""
    ids = ids if ids is not None else gpt2_token_ids(texts)
    r4 = rep_n_per_sample(ids, 4)
    out = np.zeros(len(ids), bool)
    for k, x in enumerate(ids):
        out[k] = len(x) < 16 or (len(set(x)) / len(x) < 0.15) or (not np.isnan(r4[k]) and r4[k] > 0.5)
    return out


def text_stats(texts):
    ids = gpt2_token_ids(texts)
    r4 = rep_n_per_sample(ids, 4)
    deg = degeneracy(ids=ids)
    return dict(entropy=unigram_entropy(ids=ids), rep4=float(np.nanmean(r4)),
                distinct1=distinct_n(ids=ids, n=1), distinct2=distinct_n(ids=ids, n=2),
                distinct3=distinct_n(ids=ids, n=3), degen=float(deg.mean()),
                mean_gpt2_len=float(np.mean([len(x) for x in ids])),
                per_sample_rep4=r4, per_sample_degen=deg)


# ---------------------------------------------------------------- MAUVE
@torch.no_grad()
def gpt2_features(texts, batch_size=16, max_length=1024, model_name=None):
    """MAUVE features: final-layer (post ln_f) hidden state of the last token, GPT-2 Large."""
    m = get_ppl_metrics(model_name)
    lm, tok = m._eval_model, m.tokenizer
    body = getattr(lm, "transformer", None)
    texts = [s if isinstance(s, str) and s.strip() else tok.eos_token for s in texts]
    feats = []
    for b0 in range(0, len(texts), batch_size):
        enc = tok(texts[b0:b0 + batch_size], return_tensors="pt", truncation=True,
                  max_length=max_length, padding=True)
        ii = enc["input_ids"].to(m._eval_device)
        aa = enc["attention_mask"].to(m._eval_device)
        if body is not None:
            h = body(input_ids=ii, attention_mask=aa).last_hidden_state
        else:
            h = lm(input_ids=ii, attention_mask=aa, output_hidden_states=True).hidden_states[-1]
        last = aa.sum(1) - 1  # GPT-2 tokenizer pads on the right
        feats.append(h[torch.arange(h.shape[0], device=h.device), last].float().cpu().numpy())
    return np.concatenate(feats, 0)


def mauve_from_features(gen_feats, ref_feats, seed=25):
    try:
        import mauve
    except Exception as e:
        print(f"[mauve][WARN] mauve-text not importable ({e}); returning NaN")
        return float("nan")
    try:
        out = mauve.compute_mauve(p_features=ref_feats, q_features=gen_feats, verbose=False, seed=seed)
        return float(out.mauve)
    except Exception as e:
        print(f"[mauve][WARN] compute_mauve failed ({type(e).__name__}: {e}); returning NaN")
        return float("nan")


def mauve_vs_ref(texts, ref_texts=None, ref_feats=None):
    if ref_feats is None:
        ref_feats = gpt2_features(ref_texts)
    return mauve_from_features(gpt2_features(texts), ref_feats)


# ---------------------------------------------------------------- frontier / statistics
def ppl_at_entropy(points, H=5.15):
    """points: iterable of (entropy, ppl) along ONE method's knob. Linear interp of log-PPL."""
    pts = sorted((float(h), math.log(float(p))) for h, p in points if np.isfinite(h) and p > 0)
    for (h1, l1), (h2, l2) in zip(pts, pts[1:]):
        if h1 <= H <= h2:
            w = 0.0 if h2 == h1 else (H - h1) / (h2 - h1)
            return float(math.exp(l1 + w * (l2 - l1)))
    return float("nan")


def bootstrap_ci(per_sample_nll, per_sample_count, B=10000, alpha=0.05, seed=0, chunk=1000):
    """Percentile CI of the corpus PPL, resampling samples with replacement."""
    nll = np.asarray(per_sample_nll, np.float64)
    cnt = np.asarray(per_sample_count, np.float64)
    n = len(nll)
    rng = np.random.default_rng(seed)
    vals = []
    for s in range(0, B, chunk):
        idx = rng.integers(0, n, size=(min(chunk, B - s), n))
        vals.append(np.exp(nll[idx].sum(1) / np.maximum(cnt[idx].sum(1), 1.0)))
    vals = np.concatenate(vals)
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap(nll_a, cnt_a, nll_b, cnt_b, B=10000, seed=0, chunk=1000):
    """Paired resampling over shared sample indices. Returns dict with log-PPL diff CI and p-value
    for H0: PPL_a >= PPL_b (i.e. small p => A has lower PPL)."""
    a_n, a_c = np.asarray(nll_a, np.float64), np.asarray(cnt_a, np.float64)
    b_n, b_c = np.asarray(nll_b, np.float64), np.asarray(cnt_b, np.float64)
    assert len(a_n) == len(b_n), "paired bootstrap needs aligned samples"
    n = len(a_n)
    rng = np.random.default_rng(seed)
    d = []
    for s in range(0, B, chunk):
        idx = rng.integers(0, n, size=(min(chunk, B - s), n))
        la = a_n[idx].sum(1) / np.maximum(a_c[idx].sum(1), 1)
        lb = b_n[idx].sum(1) / np.maximum(b_c[idx].sum(1), 1)
        d.append(la - lb)
    d = np.concatenate(d)
    obs = a_n.sum() / max(a_c.sum(), 1) - b_n.sum() / max(b_c.sum(), 1)
    return dict(log_ppl_diff=float(obs), ci=(float(np.quantile(d, .025)), float(np.quantile(d, .975))),
                p_value=float((d >= 0).mean()))


# ---------------------------------------------------------------- conditional metrics (official)
def bleu_rouge(hypotheses, references):
    from utils.metrics_utils import compute_bleu, compute_rouge
    return dict(bleu=compute_bleu(hypotheses, references), **compute_rouge(hypotheses, references))
