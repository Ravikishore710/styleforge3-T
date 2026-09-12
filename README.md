# AliasForge

**A StyleGAN3-T-class alias-free GAN built from scratch in TensorFlow — trained on FFHQ from random initialization.**

AliasForge implements the full StyleGAN3-style generative system by hand: the
signal-processing pipeline (Kaiser FIR filters, filtered up/downsampling,
continuous-domain leaky ReLU), the Fourier synthesis input, style-modulated
convolutions, the residual discriminator with minibatch stddev, non-saturating
logistic losses with lazy R1 regularization, an EMA generator, mixed precision,
checkpointing, and a complete inference + evaluation stack (FID, precision /
recall, StyleGAN3-style equivariance metrics, spectral analysis).

No pretrained weights. No `tensorflow_gan`. No NVIDIA code — a clean-room
TensorFlow reimplementation of the architecture family described in

> Karras et al., *Alias-Free Generative Adversarial Networks* (StyleGAN3), NeurIPS 2021.

> Disclaimer: this is a faithful-in-mechanics reimplementation, not a
> bit-exact port of `NVlabs/stylegan3`. Filter schedules and layer tables are
> documented in the module docstrings; training hyperparameters (Adam, R1
> gamma = 32.8, G/D LRs, EMA kimg) follow the published official FFHQ
> configuration as a starting baseline.

---

## Why StyleGAN3-T

| System        | 1024² A100 time/kimg | Memory | Notes                          |
| ------------- | -------------------: | -----: | ------------------------------ |
| StyleGAN2     |             ~14.6 s  | ~6 GB  | baseline                       |
| **StyleGAN3-T** |           **~20.0 s** | **~6.6 GB** | **translation equivariant** |
| StyleGAN3-R   |              ~23.4 s | ~10 GB | + rotation equivariance        |

For FFHQ faces we build **StyleGAN3-T** as the primary system (the blueprint's
decision), keeping 3-R as a reference point.

And a clarification baked into the design: **GANs have no diffusion-style
sampler.** Generation is a forward map `z -> G(z) -> image`. The "sampling
strategy" is the latent engine: Gaussian latents, seeded determinism,
truncation (`w' = w_mean + psi(w - w_mean)`), and spherical interpolation.

## The system

```
z ~ N(0, I)                       real FFHQ images (mirror-augmented)
      |                                   |
Mapping network (2-layer, w)              |
      |                                   |
styles (per-layer w)                      |
      |                                   |
Fourier synthesis input -------+          |
(band-limited sinusoids,       |          |
 learned transform)            |          |
      |                        |          |
alias-free synthesis layers:   |          |
mod conv3x3 -> noise ->        |          |
[2x up, anti-image filter] ->  |          |
bias -> leaky ReLU (continuous) |         |
[low-pass @ cutoff]            |          |
      |                        |          |
toRGB (1x1 mod conv)           |          |
      v                        v          v
   fake image <-----------> Discriminator: residual blocks,
      |                      filtered downsample, minibatch stddev
      +----> GAN loss (non-saturating logistic) + R1 (lazy)
      |                      Adam (G: 2.5e-3, D: 2.0e-3)
      v                      EMA generator = inference generator
```

Cutoff frequencies follow a geometric progression from the input Nyquist
(2 cycles/image at 4px) to the output Nyquist (`res/2`), doubling per stage;
the last `num_critical` layers operate at the target sampling rate.

## Quickstart (Colab or Kaggle)

```bash
git clone https://github.com/<you>/aliasforge.git
cd aliasforge
pip install -r requirements.txt

# data: full FFHQ at your working resolution
python scripts/prepare_ffhq.py --source drive --resolution 256
#   or point at a folder/Kaggle dataset:
# python scripts/prepare_ffhq.py --source folder --src /path/to/images --resolution 256

# verify the foundations before training (minutes)
pytest tests/ -v
jupyter nbconvert --to notebook --execute notebooks/02_signal_processing.ipynb

# ladder: 64 -> 128 -> 256 -> 512 -> 1024
python scripts/train.py --config configs/ffhq_64.yaml
python scripts/train.py --config configs/ffhq_256.yaml --resume   # after a disconnect

# inference + evaluation
python scripts/generate.py --config configs/ffhq_256.yaml --num-images 32 --seed 42
python scripts/generate.py --config configs/ffhq_256.yaml --truncation-grid
python scripts/generate.py --config configs/ffhq_256.yaml --interpolate
python scripts/evaluate.py --config configs/ffhq_256.yaml --fid-images 10000
```

The six `notebooks/` each auto-detect Colab vs Kaggle and walk one subsystem
interactively: dataset, signal processing, generator, discriminator, training,
analysis.

## Repository

```
aliasforge/
├── configs/            # ffhq_{64,128,256,512,1024}.yaml — the resolution ladder
├── scripts/
│   ├── prepare_ffhq.py # folder / zip / official-drive acquisition + verify
│   ├── train.py        # from-scratch training (kimg-based, --resume)
│   ├── generate.py     # seeded samples, truncation grid, interpolation
│   ├── evaluate.py     # FID, P/R, equivariance, spectra, diversity
│   └── analyze_latent.py
├── src/
│   ├── ops/            # Kaiser/binomial FIR design, upfirdn, filtered leaky ReLU
│   ├── data/           # FFHQ scan -> decode -> [-1,1] -> mirror -> batch
│   ├── generator/      # mapping, Fourier input, synthesis layers, toRGB
│   ├── discriminator/  # residual blocks, minibatch stddev
│   ├── losses/         # non-saturating logistic + lazy R1 (with Adam adjustment)
│   ├── training/       # Trainer (AMP, EMA, checkpointing, JSONL+TB logging)
│   ├── inference/      # seeded sampling, truncation, slerp interpolation
│   ├── evaluation/     # FID (InceptionV3), improved P/R, equivariance, spectra
│   └── visualization/  # grids, spectra plots, training curves
├── tests/              # 9 files: filters, signal processing, mapping, generator,
│                       # discriminator, losses, EMA/checkpoint, inference, data+metrics
├── notebooks/          # 01_dataset ... 06_analysis (Colab + Kaggle auto-detect)
├── outputs/            # samples/checkpoints/metrics/spectra/interpolations/logs
├── README.md
├── REPORT.md           # template — fill in after the training ladder
└── requirements.txt
```

## Training configuration (locked baseline)

| Setting            | Value                        | Source                |
| ------------------ | ---------------------------- | --------------------- |
| architecture       | StyleGAN3-T class            | blueprint decision    |
| optimizer          | Adam (β1 .9, β2 .99, ε 1e-8) | StyleGAN family       |
| G / D learning rate| 0.0025 / 0.002               | official defaults     |
| R1 gamma           | 32.8                         | official FFHQ 1024    |
| lazy R1 interval   | 16 (+ Adam `c=16/15` adjust) | official lazy reg     |
| EMA                | 10 kimg, rampup 0.05         | official              |
| augmentation       | horizontal mirror only       | official FFHQ advice  |
| ADA                | off (full FFHQ target)       | official guidance     |
| mixed precision    | on (FP16 compute, FP32 losses)| official fp16 mode   |
| duration           | in **kimg** (25000 target at 1024) | official default |

## Evaluation

| Metric        | What it proves here                                             |
| ------------- | --------------------------------------------------------------- |
| FID (InceptionV3, 50k default) | image-distribution quality                 |
| Precision / Recall (improved, k=3) | quality vs coverage — catches mode collapse |
| `eqt50k_int` / `eqt50k_frac`  | integer / fractional translation equivariance of the alias-free pipeline |
| `eqr50k`      | output-space rotation behavior (bilinear resample comparison)    |
| Radial spectra | spectral match to real FFHQ; exposes aliasing artifacts         |
| Diversity     | VGG-feature pairwise distance stats                              |

Equivariance is measured through the public API: the Fourier input accepts a
pixel `shift`, so we compare `T(G(z))` against `G(z; grid shifted)` directly —
a true test of the alias-free claim.

## Roadmap (matches the blueprint phases)

- [x] Phase 0–2: infra, FFHQ pipeline, signal-processing engine + tests
- [x] Phase 3–7: mapping, alias-free synthesis, discriminator, losses, trainer
- [x] Phase 8–11: resolution ladder configs 64 → 1024
- [x] Phase 12–13: inference CLI + evaluation CLI
- [ ] **Run the ladder and fill REPORT.md with measured numbers** ← your part

## License

MIT (see LICENSE). FFHQ is courtesy of NVIDIA / its authors — obtain it
following the FFHQ dataset terms.
