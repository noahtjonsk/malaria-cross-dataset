#!/usr/bin/env bash
# Train, evaluate and summarise more seeds of all three architectures on Colab, one run after another.
#
#     cd /content/thesis
#     nohup bash scripts/run_seeds_colab.sh > /content/drive/MyDrive/malaria_thesis/run_seeds.log 2>&1 &
#     SEEDS="2" ARCHS="resnet50" nohup bash scripts/run_seeds_colab.sh > ... 2>&1 &   # a subset
#
# Run it from the repository root after the setup cells of notebooks/10_train_colab.ipynb (code and
# data unzipped, models/ and outputs/predictions linked to Drive). It is safe to start again after a
# disconnect: train.py --resume continues a run (or skips a finished one) and evaluate.py keeps the
# prediction files it already wrote. Every table is copied to Drive as soon as it exists, under the
# names it has in the repository's outputs/tables.
set -euo pipefail

DRIVE_DIR=${DRIVE_DIR:-/content/drive/MyDrive/malaria_thesis}
SEEDS=${SEEDS:-"1 2"}
ARCHS=${ARCHS:-"vgg16 resnet50 mobilenet_v2"}
VARIANTS="raw masked reinhard histmatch"
TABLES="$DRIVE_DIR/outputs/tables"
FIGURES="$DRIVE_DIR/outputs/figures"
mkdir -p "$TABLES" "$FIGURES"

for seed in $SEEDS; do
  for arch in $ARCHS; do
    tag="${arch}_s${seed}"
    echo "=== $tag start $(date -u +%FT%TZ)"
    python scripts/train.py --arch "$arch" --seed "$seed" --workers 2 --resume
    python scripts/evaluate.py --checkpoint "models/$tag.pt" --variants $VARIANTS --workers 2
    python scripts/summarise_results.py --model "$tag" --variants $VARIANTS
    cp outputs/tables/rq*_"$tag"*.csv "outputs/tables/confusion_$tag.csv" "$TABLES/"
    cp "outputs/figures/R_$tag.png" "outputs/figures/CM_$tag.png" "$FIGURES/"
    cp "models/${tag}_log.json" "$TABLES/${tag}_training_log.json"
    cp "outputs/predictions/$tag/run.json" "$TABLES/${tag}_run.json"
    echo "=== $tag done $(date -u +%FT%TZ)"
  done
  # The comparison needs all three architectures of this seed (on Drive from this or an earlier session).
  if [ -f "outputs/predictions/vgg16_s$seed/run.json" ] && [ -f "outputs/predictions/resnet50_s$seed/run.json" ] \
     && [ -f "outputs/predictions/mobilenet_v2_s$seed/run.json" ]; then
    python scripts/compare_models.py --models "vgg16_s$seed" "resnet50_s$seed" "mobilenet_v2_s$seed" --name "s$seed"
    cp "outputs/tables/rq1_compare_s$seed.csv" "$TABLES/"
    cp "outputs/figures/R_compare_s$seed.png" "$FIGURES/"
  fi
done
echo "=== all done $(date -u +%FT%TZ)"
