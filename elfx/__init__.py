"""elfx: reusable evaluation harness for Embedded Language Flows (ELF, PyTorch port).

Paths are configurable through environment variables so the same code runs in Colab
(defaults to /content/drive/MyDrive/elf_research when mounted) and locally (automatically
falls back to the local repository root when /content/drive is not present).
"""
import os
import sys

_LOCAL_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_DRIVE_MIRROR = os.path.join(_LOCAL_REPO_ROOT, "content", "drive", "MyDrive", "elf_research")
_DEFAULT_ROOT = (
    "/content/drive/MyDrive/elf_research"
    if os.path.isdir("/content/drive")
    else (_LOCAL_DRIVE_MIRROR if os.path.isdir(_LOCAL_DRIVE_MIRROR) else _LOCAL_REPO_ROOT)
)
_DEFAULT_ELF_SRC = (
    "/content/repo/ELF/src"
    if os.path.isdir("/content/repo/ELF/src")
    else os.path.join(_LOCAL_REPO_ROOT, "ELF", "src")
)

ROOT = os.environ.get("ELF_ROOT", _DEFAULT_ROOT)
ELF_SRC = os.environ.get("ELF_SRC", _DEFAULT_ELF_SRC)
ELF_REPO = os.path.dirname(ELF_SRC.rstrip("/"))
CODE = os.path.join(ROOT, "code") if os.path.isdir("/content/drive") else _LOCAL_REPO_ROOT
RUNS = os.path.join(ROOT, "runs")
REFS = os.path.join(ROOT, "refs")

__version__ = "0.2.1"


def _patch_t5_3d_mask():
    """Make 3D (B, L, L) 0/1 attention masks work with transformers >= 4.53 / 5.x T5EncoderModel."""
    try:
        import torch
        from transformers import masking_utils
        orig = getattr(masking_utils, "_preprocess_mask_arguments", None)
        if orig is not None and not getattr(orig, "_elfx_3d_patched", False):
            def _patched_preprocess(config, inputs_embeds, attention_mask, *args, **kwargs):
                if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 3:
                    dtype = inputs_embeds.dtype
                    min_val = torch.finfo(torch.bfloat16).min if dtype in (torch.float32, torch.bfloat16) else torch.finfo(dtype).min
                    mask_bool = attention_mask[:, None, :, :].to(device=inputs_embeds.device, dtype=torch.bool)
                    attention_mask = torch.where(mask_bool, torch.zeros((), dtype=dtype, device=inputs_embeds.device), min_val)
                return orig(config, inputs_embeds, attention_mask, *args, **kwargs)
            _patched_preprocess._elfx_3d_patched = True
            masking_utils._preprocess_mask_arguments = _patched_preprocess
    except Exception:
        pass


def ensure_elf_on_path():
    """Put ELF/src first on sys.path (same as ELF/scripts/launch.sh PYTHONPATH)."""
    if ELF_SRC not in sys.path and os.path.isdir(ELF_SRC):
        sys.path.insert(0, ELF_SRC)
    _patch_t5_3d_mask()


ensure_elf_on_path()
for _d in (ROOT, RUNS, REFS):
    try:
        os.makedirs(_d, exist_ok=True)
    except OSError:
        pass
