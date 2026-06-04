# Presto-SCA-Unofficial

> Unofficial PyTorch implementation starter for **Long Video Diffusion Generation with Segmented Cross-Attention and Content-Rich Video Data Curation** (CVPR 2025).
>
> If this repo helps you understand segmented cross-attention faster, please star it and follow [@StaryMoon](https://github.com/StaryMoon). I am building honest open reproduction starters for recent vision papers.

## Status

This repository is an **independent, unofficial, work-in-progress starter**.

- Paper: [Long Video Diffusion Generation with Segmented Cross-Attention and Content-Rich Video Data Curation](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_Long_Video_Diffusion_Generation_with_Segmented_Cross-Attention_and_Content-Rich_Video_CVPR_2025_paper.html)
- Project: [Presto](https://presto-video.github.io/)
- Venue: CVPR 2025, pp. 3184-3194
- Reproduction status: **benchmarks and generation quality are not reproduced yet**.

## What Is Implemented

This v0.1.0 starter implements the key structural idea in a compact form:

- temporal segmentation of video tokens
- per-segment cross-attention to sub-caption tokens
- parameter-free segmented attention routing
- tiny DiT-style block wrapper
- smoke-test script

The goal is to make the SCA mechanism readable and easy to reuse inside toy video models.

## What Is Not Implemented Yet

- full video diffusion model
- text encoder
- VAE / latent video tokenizer
- LongTake-HD data curation
- training pipeline
- model weights or generation demos

## Quick Start

```bash
git clone https://github.com/StaryMoon/Presto-SCA-Unofficial.git
cd Presto-SCA-Unofficial
pip install -r requirements.txt
python scripts/smoke_test.py
```

Expected output:

```text
loss: ...
output: torch.Size([2, 16, 8, 64])
```

## Minimal Usage

```python
import torch

from presto_sca_unofficial import TinyPrestoBlock

video_tokens = torch.randn(2, 16, 8, 64)      # [B, T, P, C]
subcaption_tokens = torch.randn(2, 4, 12, 64) # [B, S, L, C]

block = TinyPrestoBlock(embed_dim=64, num_heads=4)
out = block(video_tokens, subcaption_tokens)
```

## Roadmap

- [ ] Add SCA variants described in the paper.
- [ ] Add simple text-subcaption parser.
- [ ] Add toy denoising diffusion loop.
- [ ] Add latent video tokenizer interface.
- [ ] Add visualization of temporal segment routing.
- [ ] Reproduce a small ablation on synthetic videos.

## Citation

If you use the method, please cite the original paper:

```bibtex
@InProceedings{Yan_2025_CVPR,
  author = {Yan, Xin and Cai, Yuxuan and Wang, Qiuyue and Zhou, Yuan and Huang, Wenhao and Yang, Huan},
  title = {Long Video Diffusion Generation with Segmented Cross-Attention and Content-Rich Video Data Curation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2025},
  pages = {3184--3194}
}
```

## License

MIT License. The original paper, project, and official materials remain owned by their respective authors / publishers.
