from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from presto_sca_unofficial import TinyPrestoBlock


def main() -> None:
    torch.manual_seed(2026)
    clean = torch.randn(2, 16, 8, 64)
    noisy = clean + 0.1 * torch.randn_like(clean)
    subcaptions = torch.randn(2, 4, 12, 64)

    block = TinyPrestoBlock(embed_dim=64, num_heads=4)
    loss = block.toy_denoising_loss(noisy, clean, subcaptions)
    loss.backward()
    output = block(noisy, subcaptions)

    print(f"loss: {loss.item():.6f}")
    print(f"output: {output.shape}")


if __name__ == "__main__":
    main()
