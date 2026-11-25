# Checks behind fixed settings (`scripts/check_settings.py`)

## Early-stopping patience (train.SETTINGS patience = 3)

| Run | Epochs run | Best epoch | Longest wait before a new best | Patience 2 keeps epoch | Stops before best |
|---|---|---|---|---|---|
| vgg16_s0 | 11 | 8 | 2 | 5 | yes |
| vgg16_s1 | 15 | 12 | 2 | 5 | yes |
| vgg16_s2 | 13 | 10 | 2 | 7 | yes |
| resnet50_s0 | 9 | 6 | 2 | 3 | yes |
| resnet50_s1 | 10 | 7 | 2 | 2 | yes |
| resnet50_s2 | 8 | 5 | 1 | 5 | no |
| mobilenet_v2_s0 | 9 | 6 | 1 | 6 | no |
| mobilenet_v2_s1 | 10 | 7 | 2 | 2 | yes |
| mobilenet_v2_s2 | 8 | 5 | 1 | 5 | no |

With patience 2, 6 of 9 runs would have stopped before their best epoch.

## Bootstrap resamples (metrics.N_BOOT = 2,000)

Seed-0 runs, 5 independent random streams (metrics.BOOT_SEED = 0 to 4).

| Interval | Count | Largest movement of a limit (pp) | Decision changed between streams |
|---|---|---|---|
| 95% drop interval (median width 4.1 pp) | 24 | 0.74 | 0 (above zero: share-recovered rule) |
| 95% RQ3 species gap | 3 | 1.05 | 0 (excludes zero) |
| 99.79% RQ1 paired comparison | 24 | 3.08 | 0 (significant in each stream: 20, 20, 20, 20, 20) |

## NIH colour reference (build_colour_crops.REFERENCE_CELLS = 1,000)

Five independent 1,000-cell samples of the 19,203 NIH training cells against all of them.

| Quantity | Largest difference over the five samples |
|---|---|
| Lab mean (L, a, b) | 0.25, 0.51, 0.41 |
| Lab within-cell std (L, a, b) | 0.09, 0.11, 0.10 |
| RGB quantiles 1-99% (grey levels, 0-255) | 4 |

For scale, |median Lab of each domain minus NIH| (`colour_channel_stats.csv`):

| Domain | L | a | b |
|---|---|---|---|
| bbbc041_masked/site_a | 28.88 | 4.13 | 27.74 |
| bbbc041_masked/site_b | 17.19 | 14.68 | 7.29 |
| mpidb_wholecell_masked/Falciparum | 12.45 | 5.71 | 9.55 |
| mpidb_wholecell_masked/Malariae | 18.99 | 0.44 | 4.13 |
| mpidb_wholecell_masked/Ovale | 8.84 | 3.10 | 18.46 |
| mpidb_wholecell_masked/Vivax | 3.05 | 1.06 | 23.85 |
| nihpoly/polygon_masked | 0.10 | 2.42 | 1.19 |
