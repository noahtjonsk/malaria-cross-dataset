# Malaria cross-dataset generalisation: the baseline pipeline in five stages.
#
#   make all          data -> split -> train -> predict -> results
#
# By default every stage runs on the committed sample in tests/fixtures/sample/
# (MALARIA_SAMPLE=1), so `make all` needs no download and finishes in about a
# minute on a CPU: MobileNetV2 is fine-tuned for one epoch on the sample's NIH
# training cells and scored on its NIH hold-out, BBBC041 and MP-IDB cells.
# Outputs go to outputs/sample/. The sample shows that the pipeline runs end to
# end; its numbers are not results.
#
#   make all FULL=1 ARCH=vgg16   the same stages on the full datasets (downloaded
#                                into the project root, see README.md); training
#                                needs a GPU (notebooks/10_train_colab.ipynb)
#
# Other targets
#   make install      pinned Python packages (requirements.txt)
#   make selftest     metric self-test (perfect predictor, AUC against scikit-learn)
#   make log          make selftest all, saved to outputs/sample/smoke_run.log
#   make tables       rebuild every results table and figure of the seed-0 models
#                     from their predictions, then the paired model comparison
#   make eda | rq2-inputs | colab   EDA statistics, RQ2 test crops, Colab archives

PYTHON ?= python
ARCH ?= mobilenet_v2
FULL ?= 0
MODELS_S0 = vgg16_s0 resnet50_s0 mobilenet_v2_s0

ifeq ($(FULL),1)
RUN = $(ARCH)_s0
CKPT = models/$(RUN).pt
TRAIN_ARGS = --seed 0 --resume
EVAL_ARGS = --variants raw masked reinhard histmatch
export MALARIA_SAMPLE = 0
else
RUN = $(ARCH)_s0_smoke
CKPT = outputs/sample/models/$(RUN).pt
TRAIN_ARGS = --smoke --workers 0
EVAL_ARGS = --workers 0 --force
export MALARIA_SAMPLE = 1
endif

.PHONY: all data split train predict results install selftest log clean-sample \
        tables eda rq2-inputs colab

all: results

# 1. Data ingestion: manifests and crops (full data only; the sample ships them),
#    then every cell is checked to be labelled, readable and model-ready.
data:
ifeq ($(FULL),1)
	$(PYTHON) scripts/build_crops.py
	$(PYTHON) scripts/build_crops.py --mpidb-wholecell
endif
	$(PYTHON) -m malaria.data

# 2. Data split: the pinned patient-grouped split (data/manifests/nih_split.csv);
#    on the full data it is recomputed and must match the pinned file.
split: data
	$(PYTHON) -m malaria.splits

# 3. Model fitting, baseline only: fine-tune one ImageNet-pretrained CNN on NIH.
train: split
	$(PYTHON) scripts/train.py --arch $(ARCH) $(TRAIN_ARGS)

# 4. Prediction with the frozen model on the NIH hold-out patients and the
#    external test sets.
predict: train
	$(PYTHON) scripts/evaluate.py --checkpoint $(CKPT) $(EVAL_ARGS)

# 5. Artifacts: results table (sensitivity, specificity, AUC and drop, with
#    intervals), confusion matrices and the metrics figure.
results: predict
	$(PYTHON) scripts/summarise_results.py --model $(RUN) $(if $(filter 1,$(FULL)),,--variants raw)

install:
	$(PYTHON) -m pip install -r requirements.txt

selftest:
	$(PYTHON) -m malaria.metrics

log:
	mkdir -p outputs/sample
	$(MAKE) --no-print-directory selftest all > outputs/sample/smoke_run.log 2>&1; \
	status=$$?; cat outputs/sample/smoke_run.log; exit $$status

clean-sample:
	rm -rf outputs/sample/models outputs/sample/predictions outputs/sample/tables outputs/sample/figures

# --- Full data only ---------------------------------------------------------
tables: export MALARIA_SAMPLE = 0
tables:
	for m in $(MODELS_S0); do $(PYTHON) scripts/summarise_results.py --model $$m || exit 1; done
	$(PYTHON) scripts/compare_models.py

eda: export MALARIA_SAMPLE = 0
eda:
	$(PYTHON) scripts/compute_stats.py --workers 6
	$(PYTHON) scripts/compute_stats.py --manifest mpidb_wholecell_cells.csv
	$(PYTHON) -m nbconvert --to notebook --execute --inplace notebooks/03_eda_short.ipynb

rq2-inputs: export MALARIA_SAMPLE = 0
rq2-inputs:
	$(PYTHON) scripts/build_masked_crops.py --method otsu-gray --datasets bbbc041 mpidb_wholecell
	$(PYTHON) scripts/build_colour_crops.py

colab: export MALARIA_SAMPLE = 0
colab:
	$(PYTHON) scripts/pack_for_colab.py
