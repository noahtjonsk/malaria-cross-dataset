# Malaria cross-dataset generalisation: the baseline pipeline, end to end.
#
# Midterm Review checks (no source datasets needed; run on data/sample/):
#   make install      pinned Python packages (requirements.txt)
#   make check        continuous-analysis checks: metric self-test, NIH split
#                     sizes and patient leakage, readable test cells
#   make smoke        the baseline arm on the committed sample, on a CPU:
#                     ingest -> split -> fit -> predict on held-out cells ->
#                     results tables and confusion matrices in outputs/sample/
#   make smoke-log    the same, with its output saved to outputs/sample/smoke_run.log
#   make all          check + smoke
#
# Full pipeline (datasets downloaded into the project root, see README.md):
#   make data         manifests and crops for every dataset
#   make eda          per-cell statistics and the short EDA notebook
#   make rq2-inputs   test crops with the background removed, then colour matched
#   make colab        archives for the Colab GPU run (notebooks/10_train_colab.ipynb)
#   make results      tables, figures and confusion matrices from the predictions
#                     of the seed-0 models, then the paired model comparison

PYTHON ?= python
ARCH ?= mobilenet_v2
MODELS_S0 = vgg16_s0 resnet50_s0 mobilenet_v2_s0

.PHONY: all install check smoke smoke-log clean-sample data eda rq2-inputs colab results

all: check smoke

install:
	$(PYTHON) -m pip install -r requirements.txt

# --- Midterm Review checks on the committed sample --------------------------
check: export MALARIA_SAMPLE = 1
check:
	$(PYTHON) -m malaria.metrics
	$(PYTHON) -m malaria.data

# One epoch on the sample's NIH train cells (train.py --smoke), then the frozen
# model scores the sample's NIH hold-out, BBBC041 and MP-IDB cells (raw crops).
smoke: export MALARIA_SAMPLE = 1
smoke:
	$(PYTHON) scripts/train.py --arch $(ARCH) --smoke --workers 0
	$(PYTHON) scripts/evaluate.py --checkpoint outputs/sample/models/$(ARCH)_s0_smoke.pt --workers 0 --force
	$(PYTHON) scripts/summarise_results.py --model $(ARCH)_s0_smoke --variants raw

smoke-log:
	$(MAKE) --no-print-directory check smoke > outputs/sample/smoke_run.log 2>&1; \
	status=$$?; cat outputs/sample/smoke_run.log; exit $$status

clean-sample:
	rm -rf outputs/sample/models outputs/sample/predictions

# --- Full pipeline ----------------------------------------------------------
data:
	$(PYTHON) scripts/build_crops.py
	$(PYTHON) scripts/build_crops.py --mpidb-wholecell

eda:
	$(PYTHON) scripts/compute_stats.py --workers 6
	$(PYTHON) scripts/compute_stats.py --manifest mpidb_wholecell_cells.csv
	$(PYTHON) -m nbconvert --to notebook --execute --inplace notebooks/03_eda_short.ipynb

rq2-inputs:
	$(PYTHON) scripts/build_masked_crops.py --method otsu-gray --datasets bbbc041 mpidb_wholecell
	$(PYTHON) scripts/build_colour_crops.py

colab:
	$(PYTHON) scripts/pack_for_colab.py

results:
	for m in $(MODELS_S0); do $(PYTHON) scripts/summarise_results.py --model $$m || exit 1; done
	$(PYTHON) scripts/compare_models.py
