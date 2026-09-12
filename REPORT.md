# AliasForge — Final Report (template)

Fill in every `—` with measured values after running the ladder. Do not
invent numbers; if a run crashed or misbehaved, document it in §Failure analysis.

## 1. Problem
High-quality human-face generation from random initialization on FFHQ.

## 2. Dataset
- Source: FFHQ (70,000 images, 1024², RGB) — via `scripts/prepare_ffhq.py`
- Pipeline: integrity scan → decode → resize → [-1,1] → horizontal mirror → batch
- Augmentation: mirror = on; ADA = off (full-dataset target)

## 3. Architecture
StyleGAN3-T class: 2-layer mapping (z_dim=512, w_dim=512), band-limited Fourier
synthesis input (4x4, bandwidth = input Nyquist), N upsampling layers with
geometric cutoff schedule, num_critical=2 critical layers, toRGB 1x1 mod conv.
Layer table (from `03_generator.ipynb`):

```
— paste layer inventory here —
```

## 4. Signal-processing formulation
Continuous signal → filtered sampling → alias-free synthesis. Kaiser FIRs
(60 dB), zero-stuffing ×2 + anti-imaging low-pass, bias + leaky ReLU applied in
the upsampled (continuous) domain, low-pass at the output cutoff before
returning to the pixel grid.

## 5. Generator — mathematical description
— w = M(z), ws = repeat(w, num_ws); x0 = sin(2π(F·T(w)·coords) + φ)·W —
— per layer: x ← filtered_lrelu(modconv(x, A_i(w_i)) + ν·ε, fu, fd) —
— image = modconv_1x1(x, A_last(w)) + b —

## 6. Discriminator — mathematical description
— from_rgb conv; residual blocks: conv3x3 → LReLU → conv3x3 → LReLU →
  filtered downsample [1,4,6,4,1]/16, skip: 1x1 → downsample; minibatch std;
  conv3x3 → FC → scalar —

## 7. Objective
L_G = E_z softplus(-D(G(z)));  L_D = E_x softplus(-D(x)) + E_z softplus(D(G(z)))
+ lazy R1: + interval · γ · ½ E_x‖∇_x D(x)‖² every 16 D-steps.

## 8. Optimization
Adam(G: 2.5e-3, D: 2.0e-3, β1=.9, β2=.99, ε=1e-8, lazy-reg adjusted c=16/15);
EMA 10 kimg + 0.05 rampup; mixed precision.

## 9. Training ladder
| Stage | Res | kimg reached | wall time | GPU | final FID |
|-------|-----|-------------|-----------|-----|-----------|
| dev   | 64  | — | — | — | — |
| dev   | 128 | — | — | — | — |
| main  | 256 | — | — | — | — |
| serious | 512 | — | — | — | — |
| flagship | 1024 | — | — | — | — |

## 10. Evaluation results (256² checkpoint)
| Metric | Value |
|---|---|
| FID (—k images) | — |
| Precision | — |
| Recall | — |
| eqt_int / eqt_frac / eqr | — / — / — |
| spectral L2 / slopes | — |
| diversity mean ± std | — |

## 11. Failure analysis
- Mode collapse signals: —
- Artifacts / spectral abnormalities: —
- Instability events (NaN spikes, D overpowering): —
- Premature conclusions avoided: —

## 12. Computational analysis
| Property | G | D |
|---|---|---|
| parameters | — | — |
| peak GPU memory (res/batch) | — | — |
| throughput img/s | — | — |
| inference time per 1k images | — | — |

## 13. Conclusions
— Did integer-shift equivariance beat the naive-roll baseline? By how much?
— What did the truncation sweep trade off? —
— What would you try next (R-variant? ADA on small subsets? longer 1024 run)? —
