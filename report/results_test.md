Test partition: 789 samples / 179 images. Seeds: [0, 1, 2] (mean ± sample std). GPU: NVIDIA GeForce RTX 4060 Laptop GPU, latency at batch 1.
Subset sizes: multi-object img = 560, spatial words = 512, no spatial words = 277.

| variant | trainable params | mIoU | Dice | P@0.5 | multi-object img | spatial words | no spatial words | train mIoU | gap | full ms | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 image-only | 656,737 | 0.281 ± 0.002 | 0.397 ± 0.005 | 0.179 ± 0.023 | 0.258 ± 0.002 | 0.253 ± 0.002 | 0.333 ± 0.009 | 0.387 ± 0.088 | 0.107 ± 0.090 | 5.1 | 6.0 ± 4.4 |
| M1 FiLM | 1,050,977 | 0.375 ± 0.013 | 0.478 ± 0.013 | 0.375 ± 0.021 | 0.360 ± 0.013 | 0.375 ± 0.017 | 0.375 ± 0.006 | 0.733 ± 0.051 | 0.358 ± 0.045 | 5.1 | 14.3 ± 2.5 |
| M1b concat | 853,857 | 0.353 ± 0.012 | 0.456 ± 0.013 | 0.341 ± 0.019 | 0.341 ± 0.007 | 0.350 ± 0.007 | 0.358 ± 0.022 | 0.644 ± 0.009 | 0.291 ± 0.018 | 5.2 | 11.7 ± 0.6 |
| M1 FiLM, no coords | 1,050,721 | 0.375 ± 0.005 | 0.478 ± 0.009 | 0.381 ± 0.022 | 0.361 ± 0.005 | 0.370 ± 0.006 | 0.383 ± 0.003 | 0.760 ± 0.055 | 0.385 ± 0.057 | 5.2 | 16.0 ± 3.6 |
| M1 FiLM, BCE only | 1,050,977 | 0.332 ± 0.009 | 0.428 ± 0.011 | 0.319 ± 0.013 | 0.318 ± 0.012 | 0.325 ± 0.011 | 0.345 ± 0.011 | 0.716 ± 0.038 | 0.384 ± 0.029 | 5.2 | 18.0 ± 1.7 |
| R1 dropout + wd | 1,050,721 | 0.392 ± 0.013 | 0.495 ± 0.017 | 0.398 ± 0.005 | 0.379 ± 0.012 | 0.390 ± 0.016 | 0.396 ± 0.006 | 0.746 ± 0.042 | 0.354 ± 0.048 | 5.3 | 18.0 ± 3.5 |
| R2 flip + L/R swap | 1,050,721 | 0.393 ± 0.013 | 0.498 ± 0.013 | 0.400 ± 0.019 | 0.380 ± 0.008 | 0.386 ± 0.014 | 0.407 ± 0.014 | 0.710 ± 0.053 | 0.317 ± 0.039 | 5.4 | 17.3 ± 2.5 |
| R3 R1 + R2 | 1,050,721 | 0.403 ± 0.012 | 0.506 ± 0.017 | 0.414 ± 0.017 | 0.392 ± 0.006 | 0.403 ± 0.015 | 0.404 ± 0.008 | 0.725 ± 0.027 | 0.322 ± 0.032 | 5.4 | 23.0 ± 4.4 |
| CLIPSeg (reference) | 150,747,746 (all frozen) | 0.385 | 0.492 | 0.371 | 0.382 | 0.307 | 0.530 | – | – | 7.7 | – |

Paired by seed, mIoU relative to 'M0 image-only':
- M1 FiLM: 0.094 ± 0.012 (ahead in 3/3 seeds)
- M1b concat: 0.072 ± 0.011 (ahead in 3/3 seeds)
- M1 FiLM, no coords: 0.094 ± 0.007 (ahead in 3/3 seeds)
- M1 FiLM, BCE only: 0.051 ± 0.008 (ahead in 3/3 seeds)
- R1 dropout + wd: 0.112 ± 0.014 (ahead in 3/3 seeds)
- R2 flip + L/R swap: 0.113 ± 0.015 (ahead in 3/3 seeds)
- R3 R1 + R2: 0.123 ± 0.013 (ahead in 3/3 seeds)
- FiLM − concat: 0.021 ± 0.004 (FiLM ahead in 3/3 seeds)

CLIPSeg: pretrained, threshold 0.25 chosen on val; different pretraining (PhraseCut), so it is a reference point, not a controlled comparison.
train mIoU: selected checkpoint evaluated on the training partition in eval mode; gap = train mIoU − val mIoU.
