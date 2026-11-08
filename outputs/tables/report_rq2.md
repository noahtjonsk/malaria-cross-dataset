| Model | Test set | Metric | Raw | Masked | + Reinhard | + Hist. matching | Share recovered (seeds) |
|---|---|---|---|---|---|---|---|
| VGG-16 | site_a | Sensitivity | 2.1 | -0.3 | 34.7 | -1.7 | 0.62 (2/3) |
|  |  | Specificity | 20.1 | 32.7 | 13.3 | 89.2 | -0.76 (3/3) |
|  |  | AUC | 3.5 | 3.9 | 14.2 | 8.4 | -0.14 (3/3) |
|  | site_b | Sensitivity | 7.1 | 6.9 | 11.2 | -1.6 | 0.06 (2/3) |
|  |  | Specificity | 1.3 | 5.6 | 11.4 | 94.5 | n/a |
|  |  | AUC | 2.2 | 3.0 | 5.9 | 24.6 | -0.38 (3/3) |
|  | MP-IDB *P. falciparum* | Sensitivity | 33.3 | 26.3 | 31.4 | -1.3 | 0.19 (3/3) |
|  | MP-IDB other | Sensitivity | 22.7 | 20.3 | 34.1 | -0.4 | 0.11 (3/3) |
| ResNet-50 | site_a | Sensitivity | 7.0 | 0.4 | 34.5 | -2.0 | 1.07 (2/3) |
|  |  | Specificity | 16.5 | 40.7 | 20.1 | 93.5 | -1.54 (3/3) |
|  |  | AUC | 5.6 | 5.9 | 19.9 | 21.7 | -0.01 (3/3) |
|  | site_b | Sensitivity | 30.4 | 17.0 | 14.9 | -0.7 | 0.43 (3/3) |
|  |  | Specificity | -0.6 | 1.8 | 9.2 | 91.5 | n/a |
|  |  | AUC | 2.9 | 3.3 | 6.5 | 33.9 | -0.11 (3/3) |
|  | MP-IDB *P. falciparum* | Sensitivity | 52.9 | 36.0 | 42.6 | -1.6 | 0.31 (3/3) |
|  | MP-IDB other | Sensitivity | 38.5 | 22.0 | 44.7 | -2.2 | 0.43 (3/3) |
| MobileNetV2 | site_a | Sensitivity | 21.8 | 20.2 | 63.4 | 2.3 | 0.10 (3/3) |
|  |  | Specificity | 5.9 | 18.0 | 3.9 | 62.7 | -2.20 (3/3) |
|  |  | AUC | 5.7 | 11.6 | 14.1 | 16.4 | -1.02 (3/3) |
|  | site_b | Sensitivity | 59.4 | 50.1 | 41.4 | 12.1 | 0.15 (3/3) |
|  |  | Specificity | -1.0 | 1.2 | 0.7 | 59.6 | n/a |
|  |  | AUC | 8.0 | 12.5 | 9.3 | 26.8 | -0.50 (3/3) |
|  | MP-IDB *P. falciparum* | Sensitivity | 63.1 | 63.1 | 59.3 | 4.1 | 0.01 (3/3) |
|  | MP-IDB other | Sensitivity | 61.5 | 60.1 | 63.4 | 9.8 | 0.03 (3/3) |
