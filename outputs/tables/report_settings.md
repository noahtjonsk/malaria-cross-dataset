| Setting | Value |
|---|---|
| Learning rate (Adam optimiser: step size of each weight update) | 0.0001 |
| Batch size (images per weight update) | 32 |
| Maximum epochs (passes over the training data) | 20 |
| Early stopping: epochs without lower NIH validation error before training stops | 3 |
| Weight decay (penalty on large weights) | none |
| Input size (pixels, square) | 224 |
| Training augmentation | horizontal and vertical flips, rotations by 90° |
| Decision threshold (output above it = parasitised) | 0.5 |
| Started from ImageNet weights, all layers trained | yes |
