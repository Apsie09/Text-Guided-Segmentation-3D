Validation partition: 857 samples / 178 images. Seeds: [0, 1, 2] (mean ± sample std). GPU: NVIDIA GeForce RTX 4060 Laptop GPU, latency at batch 1.
Subset sizes: multi-object img = 648, spatial words = 562, no spatial words = 295.

| variant | trainable params | mIoU | Dice | P@0.5 | multi-object img | spatial words | no spatial words | train mIoU | gap | full ms | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FiLM-nc baseline | 1,050,721 | 0.413 ± 0.002 | 0.519 ± 0.005 | 0.426 ± 0.013 | 0.396 ± 0.003 | 0.437 ± 0.004 | 0.366 ± 0.009 | 0.760 ± 0.055 | 0.347 ± 0.056 | 5.9 | 16.0 ± 3.6 |
| R1 dropout + wd | 1,050,721 | 0.424 ± 0.012 | 0.531 ± 0.011 | 0.453 ± 0.020 | 0.405 ± 0.014 | 0.445 ± 0.015 | 0.384 ± 0.012 | 0.746 ± 0.042 | 0.322 ± 0.040 | 6.0 | 18.0 ± 3.5 |
| R2 flip + L/R swap | 1,050,721 | 0.432 ± 0.016 | 0.537 ± 0.015 | 0.467 ± 0.028 | 0.416 ± 0.020 | 0.455 ± 0.018 | 0.388 ± 0.012 | 0.710 ± 0.053 | 0.279 ± 0.037 | 5.7 | 17.3 ± 2.5 |
| R3 R1 + R2 | 1,050,721 | 0.433 ± 0.013 | 0.536 ± 0.015 | 0.465 ± 0.012 | 0.415 ± 0.014 | 0.465 ± 0.015 | 0.372 ± 0.028 | 0.725 ± 0.027 | 0.293 ± 0.022 | 5.9 | 23.0 ± 4.4 |

Paired by seed, mIoU relative to 'FiLM-nc baseline':
- R1 dropout + wd: 0.011 ± 0.010 (ahead in 3/3 seeds)
- R2 flip + L/R swap: 0.019 ± 0.014 (ahead in 3/3 seeds)
- R3 R1 + R2: 0.020 ± 0.012 (ahead in 3/3 seeds)
train mIoU: selected checkpoint evaluated on the training partition in eval mode; gap = train mIoU − val mIoU.
