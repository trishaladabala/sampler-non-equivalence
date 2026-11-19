#!/usr/bin/env python3
"""Training script for the EFM-style continuous flow model.

Supports all local-time conditioning modes (continuous / quantized / lowrank).
Uses flow matching + cross-entropy decode + insertion losses.

Usage:
    # Smoke test on Mac (tiny model, WikiText-2)
    python -m efm_pipeline.train_efm --model_size tiny --max_steps 100 --dataset wikitext2

    # A4000 experiment (small model, OpenWebText)
    python -m efm_pipeline.train_efm --model_size small --max_steps 50000 --time_mode continuous

    # Quantized time conditioning
    python -m efm_pipeline.train_efm --model_size small --time_mode quantized --time_K 4

    # Low-rank time conditioning
    python -m efm_pipeline.train_efm --model_size small --time_mode lowrank --time_K 8
"""

import argparse
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ELF imports.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ELF_SRC = os.path.normpath(os.path.join(_THIS_DIR, "..", "ELF", "src"))
if _ELF_SRC not in sys.path:
    sys.path.insert(0, _ELF_SRC)
_PROJ_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from utils.sampling_utils import add_noise, sample_timesteps

from efm_pipeline.efm_model import EFM_Tiny, EFM_Small, EFMModel
from efm_pipeline.insertion_head import InsertionHead
from efm_pipeline.utils import (
    set_seed, get_device, device_summary, ensure_dir,
    amp_autocast, gpu_memory_mb, Timer,
)
from efm_pipeline.logger import ExperimentLogger


# ---------------------------------------------------------------------------
# Simple dataset (pre-embedded or on-the-fly)
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """Dataset that returns pre-computed T5 embeddings.

    For smoke tests, generates random embeddings. For real runs,
    loads from a cached .pt file or computes on-the-fly.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seq_length: int = 64,
        embed_dim: int = 512,
        vocab_size: int = 32128,
        cache_path: Optional[str] = None,
    ):
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}")
            data = torch.load(cache_path, map_location="cpu")
            self.embeddings = data["embeddings"]
            self.token_ids = data["token_ids"]
            self.num_samples = self.embeddings.shape[0]
        else:
            # Generate random data for smoke tests.
            print(f"Generating {num_samples} random embedding samples (seq_len={seq_length})")
            self.num_samples = num_samples
            self.embeddings = torch.randn(num_samples, seq_length, embed_dim)
            self.token_ids = torch.randint(0, vocab_size, (num_samples, seq_length))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "embeddings": self.embeddings[idx],
            "token_ids": self.token_ids[idx],
        }

    @staticmethod
    def from_text(
        dataset_name: str = "wikitext2",
        encoder_name: str = "t5-small",
        seq_length: int = 64,
        max_samples: int = 5000,
        cache_dir: str = "data_cache",
    ) -> "EmbeddingDataset":
        """Create dataset by encoding text with T5.

        Caches the result to avoid re-encoding.
        """
        cache_path = os.path.join(
            cache_dir, f"{dataset_name}_{encoder_name}_len{seq_length}.pt"
        )
        if os.path.exists(cache_path):
            return EmbeddingDataset(cache_path=cache_path)

        ensure_dir(cache_dir)
        print(f"Encoding {dataset_name} with {encoder_name} (seq_len={seq_length})...")

        from transformers import T5EncoderModel, AutoTokenizer
        from datasets import load_dataset

        tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        encoder = T5EncoderModel.from_pretrained(encoder_name).eval()

        if dataset_name == "wikitext2":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            texts = [t for t in ds["text"] if len(t.strip()) > 20]
        elif dataset_name == "openwebtext":
            ds = load_dataset("openwebtext", split="train", streaming=True)
            texts = []
            for item in ds:
                if len(item["text"].strip()) > 20:
                    texts.append(item["text"])
                if len(texts) >= max_samples:
                    break
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        texts = texts[:max_samples]
        print(f"  {len(texts)} texts loaded")

        # Tokenize and encode.
        all_embeddings = []
        all_ids = []
        batch_size = 32

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded = tokenizer(
                    batch_texts, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=seq_length,
                )
                outputs = encoder(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
                embs = outputs.last_hidden_state  # (B, S, D)
                all_embeddings.append(embs.cpu())
                all_ids.append(encoded["input_ids"].cpu())
                if (i // batch_size + 1) % 10 == 0:
                    print(f"  Encoded {min(i+batch_size, len(texts))}/{len(texts)}")

        embeddings = torch.cat(all_embeddings, dim=0)
        token_ids = torch.cat(all_ids, dim=0)

        # Save cache.
        torch.save({"embeddings": embeddings, "token_ids": token_ids}, cache_path)
        print(f"  Cached to {cache_path}")

        ds = EmbeddingDataset(cache_path=cache_path)
        return ds


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class EFMConfig:
    """Minimal config object compatible with ELF's add_noise."""
    def __init__(self):
        self.denoiser_noise_scale = 1.0
        self.t_eps = 5e-2


def train(
    model: EFMModel,
    dataset: EmbeddingDataset,
    device: torch.device,
    max_steps: int = 50000,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_fraction: float = 0.1,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
    save_every: int = 5000,
    log_every: int = 100,
    checkpoint_dir: str = "checkpoints",
    experiment_name: str = "efm_train",
    output_dir: str = "results",
    flow_loss_weight: float = 1.0,
    decode_loss_weight: float = 0.5,
    insertion_loss_weight: float = 0.1,
) -> str:
    """Train the EFM model.

    Returns path to the final checkpoint.
    """
    model = model.to(device)
    model.train()

    logger = ExperimentLogger(output_dir, experiment_name)
    timer = Timer().start()

    # Optimizer.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    # LR schedule: linear warmup + cosine decay.
    warmup_steps = int(max_steps * warmup_fraction)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler.
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and device.type == "cuda"))

    # Data loader.
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    loader_iter = iter(loader)

    efm_config = EFMConfig()
    ckpt_dir = ensure_dir(os.path.join(checkpoint_dir, experiment_name))

    print(f"\n{'='*60}")
    print(f"  Training: {experiment_name}")
    print(f"  Device: {device_summary(device)}")
    print(f"  Parameters: {sum(p.numel() for p in params):,}")
    print(f"  Steps: {max_steps}, Batch: {batch_size}, LR: {learning_rate}")
    print(f"  AMP: {use_amp}, Grad norm: {max_grad_norm}")
    print(f"{'='*60}\n")

    for step in range(1, max_steps + 1):
        # Get batch (cycle through dataset).
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        x0 = batch["embeddings"].to(device)          # (B, S, D)
        target_ids = batch["token_ids"].to(device)    # (B, S)
        B, S, D = x0.shape

        # Sample timesteps.
        t = sample_timesteps(B, P_mean=0.8, P_std=0.8,
                             time_schedule="logit_normal", device=device)

        # Noisy interpolation: z = t·x0 + (1-t)·noise.
        noise = torch.randn_like(x0)
        z = add_noise(x0, noise, t, efm_config)

        # Local times: for now, all tokens share the global time.
        local_times = t.unsqueeze(1).expand(-1, S)    # (B, S)

        optimizer.zero_grad()

        # Forward pass with AMP.
        amp_ctx = amp_autocast(device, enabled=use_amp)
        with amp_ctx:
            flow_out, decoder_logits = model(
                z, t, local_times=local_times,
                decoder_step_active=True, deterministic=False,
            )

            # --- Losses ---
            # 1. Flow matching loss: MSE(predicted_x0, actual_x0).
            loss_flow = F.mse_loss(flow_out, x0)

            # 2. Decoder cross-entropy loss.
            loss_decode = torch.tensor(0.0, device=device)
            if decoder_logits is not None:
                loss_decode = F.cross_entropy(
                    decoder_logits.reshape(-1, model.vocab_size),
                    target_ids.reshape(-1),
                    ignore_index=0,  # Ignore padding (id=0).
                )

            # 3. Insertion loss (if enabled).
            loss_insert = torch.tensor(0.0, device=device)
            if model.insertion_head is not None:
                # For training, use random target counts.
                # In real experiments, these come from the expansion schedule.
                rates = model.get_insertion_rates(
                    model.text_proj(z.float())  # Need hidden states
                )
                # Target: no insertion for now (baseline).
                target_counts = torch.zeros(B, S - 1, device=device)
                loss_insert = InsertionHead.poisson_nll_loss(rates, target_counts)

            # Combined loss.
            loss = (flow_loss_weight * loss_flow
                    + decode_loss_weight * loss_decode
                    + insertion_loss_weight * loss_insert)

        # Backward.
        scaler.scale(loss).backward()
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Logging.
        if step % log_every == 0 or step == 1:
            lr = scheduler.get_last_lr()[0]
            metrics = {
                "step": step,
                "loss": loss.item(),
                "loss_flow": loss_flow.item(),
                "loss_decode": loss_decode.item(),
                "loss_insert": loss_insert.item(),
                "lr": lr,
            }
            if device.type == "cuda":
                metrics["gpu_mb"] = gpu_memory_mb(device)["allocated_mb"]
            logger.log_step(metrics, print_every=log_every)

            # Check for numerical issues.
            if not np.isfinite(loss.item()):
                print(f"  WARNING: Non-finite loss at step {step}. Stopping.")
                break

        # Checkpointing.
        if step % save_every == 0 or step == max_steps:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss.item(),
            }, ckpt_path)
            print(f"  Checkpoint saved → {ckpt_path}")

            # Keep only last N checkpoints.
            all_ckpts = sorted(
                [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_")],
                key=lambda x: int(x.split("_")[1].split(".")[0]),
            )
            while len(all_ckpts) > 3:
                os.remove(os.path.join(ckpt_dir, all_ckpts.pop(0)))

    elapsed = timer.stop()
    logger.save_result({
        "final_step": step,
        "final_loss": loss.item(),
        "elapsed_seconds": round(elapsed, 1),
    })
    logger.close()

    print(f"\nTraining complete: {step} steps in {elapsed:.0f}s")
    return ckpt_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train EFM flow model")
    parser.add_argument("--model_size", type=str, default="tiny", choices=["tiny", "small"])
    parser.add_argument("--time_mode", type=str, default="continuous",
                        choices=["continuous", "quantized", "lowrank"])
    parser.add_argument("--time_K", type=int, default=16)
    parser.add_argument("--lowrank_basis", type=str, default="fourier", choices=["fourier", "learned"])
    parser.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "openwebtext"])
    parser.add_argument("--seq_length", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--insertion_enabled", action="store_true", default=False)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    # Build experiment name.
    if args.experiment_name is None:
        args.experiment_name = f"efm_{args.model_size}_{args.time_mode}_K{args.time_K}"

    # Build model.
    factory = EFM_Tiny if args.model_size == "tiny" else EFM_Small
    model = factory(
        local_time_mode=args.time_mode,
        local_time_K=args.time_K,
        lowrank_basis=args.lowrank_basis,
        insertion_enabled=args.insertion_enabled,
        max_length=args.seq_length * 2,  # Allow expansion room.
        gradient_checkpointing=(args.model_size == "small"),
    )
    print(f"Model: {args.model_size} ({sum(p.numel() for p in model.parameters()):,} params)")

    # Build dataset.
    if args.dataset == "wikitext2" and args.max_steps <= 200:
        # Quick random data for smoke tests.
        dataset = EmbeddingDataset(
            num_samples=500, seq_length=args.seq_length,
            embed_dim=model.text_encoder_dim,
        )
    else:
        dataset = EmbeddingDataset.from_text(
            dataset_name=args.dataset,
            seq_length=args.seq_length,
            max_samples=10000 if args.dataset == "wikitext2" else 50000,
        )

    # Train.
    train(
        model=model, dataset=dataset, device=device,
        max_steps=args.max_steps, batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_amp=args.use_amp and device.type == "cuda",
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
