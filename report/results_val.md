Validation partition: 857 samples / 178 images. Seeds: [0, 1, 2] (mean ± sample std). GPU: NVIDIA GeForce RTX 4060 Laptop GPU, latency at batch 1.
Subset sizes: multi-object img = 648, spatial words = 562, no spatial words = 295.

| variant | trainable params | mIoU | Dice | P@0.5 | multi-object img | spatial words | no spatial words | full ms | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 image-only | 656,737 | 0.278 ± 0.008 | 0.397 ± 0.010 | 0.145 ± 0.018 | 0.260 ± 0.007 | 0.265 ± 0.004 | 0.302 ± 0.015 | 5.7 | 6.0 ± 4.4 |
| M1 FiLM | 1,050,977 | 0.405 ± 0.009 | 0.511 ± 0.010 | 0.412 ± 0.009 | 0.388 ± 0.007 | 0.428 ± 0.008 | 0.360 ± 0.026 | 5.8 | 14.3 ± 2.5 |
| M1b concat | 853,857 | 0.388 ± 0.002 | 0.494 ± 0.003 | 0.394 ± 0.006 | 0.369 ± 0.003 | 0.413 ± 0.002 | 0.341 ± 0.004 | 5.6 | 11.7 ± 0.6 |
| M1 FiLM, no coords | 1,050,721 | 0.413 ± 0.002 | 0.519 ± 0.005 | 0.426 ± 0.013 | 0.396 ± 0.003 | 0.437 ± 0.004 | 0.366 ± 0.009 | 5.6 | 16.0 ± 3.6 |
| M1 FiLM, BCE only | 1,050,977 | 0.363 ± 0.016 | 0.460 ± 0.020 | 0.373 ± 0.029 | 0.344 ± 0.020 | 0.388 ± 0.015 | 0.315 ± 0.023 | 5.6 | 18.0 ± 1.7 |
| CLIPSeg (reference) | 150,747,746 (all frozen) | 0.360 | 0.466 | 0.307 | 0.349 | 0.300 | 0.473 | 7.6 | – |

Paired by seed: FiLM − image-only mIoU = 0.127 ± 0.004; FiLM − concat mIoU = 0.016 ± 0.006 (FiLM ahead in 3/3 seeds).
CLIPSeg: pretrained, threshold 0.25 chosen on val; different pretraining (PhraseCut), so it is a reference point, not a controlled comparison.
