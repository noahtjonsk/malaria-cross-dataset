# Cross-Dataset Generalisation of Deep Learning for Malaria Classification

Capstone project. Train on NIH (Bangladesh), test on BBBC041 (Broad) and
MP-IDB (CHUV Lausanne).

The trained models are research prototypes for studying why malaria classifiers lose
performance across datasets. They have no clinical validation and must not be used to
diagnose or screen patients.

## Midterm Review

The tagged commit `v1.0-final` holds the three Midterm Review components:

- the Literature Review, [`LiteratureReview_NoahTjonSienKie.pdf`](LiteratureReview_NoahTjonSienKie.pdf);
- the Methodology Overview, [`docs/methodology_overview_v3.md`](docs/methodology_overview_v3.md) (PDF: [`docs/methodology_overview_v3.pdf`](docs/methodology_overview_v3.pdf));
- the proof-of-concept code for the baseline arm: three ImageNet-pretrained CNNs
  (VGG-16, ResNet-50, MobileNetV2) fine-tuned on NIH and applied unchanged to BBBC041 and MP-IDB.

**Quick check, no datasets needed.** The source datasets are several GB, so a small
sample of the same cells is committed in [`tests/fixtures/sample/`](tests/fixtures/sample/README.md)
(340 crops, 9 MB). From a clean Python 3.13 environment:

```bash
make install      # pip install -r requirements.txt
make all          # data -> split -> train -> predict -> results on the sample (CPU, about a minute)
```

`make all` fine-tunes MobileNetV2 for one epoch on the sample's NIH training cells,
scores its NIH hold-out, BBBC041 and MP-IDB cells, and writes the results table,
confusion matrices and metrics figure to `outputs/sample/`. The log of a complete run,
with the metric self-test, is [`outputs/sample/smoke_run.log`](outputs/sample/smoke_run.log)
(`make log`). The sample over-represents parasitised cells and trains for one epoch, so
its numbers only show that the pipeline runs end to end. `make all FULL=1 ARCH=vgg16`
runs the same stages on the full datasets; training then needs a GPU.

**The five Makefile stages**, and what they produced on the full data:

| Stage | Code | Output |
|---|---|---|
| `make data`: ingestion and transformation | `scripts/build_crops.py` (+ `--mpidb-wholecell`); `malaria/manifests.py`, `crops.py`; checked by `python -m malaria.data` | one manifest per dataset, the cell crops |
| `make split` | `malaria/splits.py` (`python -m malaria.splits` recomputes the split and checks it equals the pinned file) | [`data/manifests/nih_split.csv`](data/manifests/nih_split.csv): 70/15/15 by patient |
| `make train`: baseline only | `scripts/train.py`, `malaria/models.py`, `malaria/data.py`; run on a Colab T4 through `notebooks/10_train_colab.ipynb` | weights; training logs [`outputs/tables/<model>_s0_training_log.json`](outputs/tables/) |
| `make predict`: held-out sets | `scripts/evaluate.py` | per-cell probabilities for the NIH hold-out patients, BBBC041 and MP-IDB; record of inputs [`outputs/tables/<model>_s0_run.json`](outputs/tables/) |
| `make results`: results table and error visualisation | `scripts/summarise_results.py`, `scripts/compare_models.py`, `malaria/metrics.py` | [`rq1_<model>_s0.csv`](outputs/tables/rq1_vgg16_s0.csv), [`confusion_<model>_s0.csv`](outputs/tables/confusion_vgg16_s0.csv), [`CM_<model>_s0.png`](outputs/figures/CM_vgg16_s0.png), [`R_<model>_s0.png`](outputs/figures/R_vgg16_s0.png), [`rq1_compare_s0.csv`](outputs/tables/rq1_compare_s0.csv) |

**How the baseline line of the results table is built.** `evaluate.py` writes one
probability per cell; `summarise_results.py --model <model>_s0` applies the fixed 0.5
threshold, computes sensitivity, specificity and AUC per test set with 95% intervals
from resampling source images (NIH hold-out: patients), and writes the drop from the NIH
hold-out value in percentage points to `rq1_<model>_s0.csv`. These rows are Table 8 of
the Methodology Overview. The confusion matrices `CM_<model>_s0.png` are drawn from the
same predictions and reproduce those sensitivities and specificities exactly.
`make tables` rebuilds every table and figure from the predictions.

**Tracked tables and figures are results, not raw data.** No image from the source
datasets is tracked apart from the sample. What is tracked as CSV is derived: the pinned
split (`data/manifests/nih_split.csv`, patient code and label per cell), the MP-IDB
stage-label audit, and the summary tables of the completed seed-0 runs in
`outputs/tables/`. The training predictions they were computed from cannot be shared
in git, so these tables are the record of those runs, and the Methodology Overview
links to them. Some EDA figures and the executed EDA notebook are over 1 MB.

**Trained models and predictions** are not in git because of their size (VGG-16 is
0.5 GB). They are kept on the author's Google Drive (`MyDrive/malaria_thesis/`) and
are available on request. What they produced is committed: training logs,
`run.json` records (checkpoint hash, code commit, manifest hashes and a digest of every
crop scored), and all result tables and figures.

## Run order

```bash
pip install -r requirements.txt               # pinned versions this was last run with
python scripts/build_crops.py                 # manifests + ~86k cached crops (~15 min, ~2.9 GB)
python scripts/compute_stats.py --workers 8   # per-image statistics for 115k cells (~5 min)
# NIH source release, recut cells and background methods (needs NIH-NLM-ThinBloodSmearsPf/ in the project root)
python scripts/check_nih_overlap.py           # how that release relates to cell_images
python scripts/build_crops.py --nihpoly       # NIH cells recut from the Polygon Set (~3 min)
python scripts/compute_stats.py --manifest nihpoly_cells.csv
python scripts/validate_background.py         # scores the background-removal methods (~40 min)
python scripts/build_crops.py --mpidb-wholecell       # MP-IDB framed on the host red cell (~15 min)
python scripts/build_masked_crops.py --method otsu-gray --datasets bbbc041 mpidb_wholecell --uninfected-per-split 3000
python scripts/compute_stats.py --manifest masked_otsu-gray_sample_cells.csv
python scripts/compute_colour_stats.py --masked otsu-gray_sample
python -m papermill notebooks/01_eda.ipynb outputs/01_eda_run.ipynb --cwd notebooks
# short EDA: findings and the decisions they lead to (about a minute)
python scripts/compute_stats.py --manifest mpidb_wholecell_cells.csv
python -m nbconvert --to notebook --execute --inplace notebooks/03_eda_short.ipynb
python -m nbconvert --to html --no-input notebooks/03_eda_short.ipynb --output-dir outputs
# RQ2 test inputs, every cell (Otsu ~15 min, colour ~20 min)
python scripts/build_masked_crops.py --method otsu-gray --datasets bbbc041 mpidb_wholecell
python scripts/build_colour_crops.py          # Reinhard + histogram matching to NIH train cells
# models: smoke checks on the laptop, real runs on Colab (notebooks/10_train_colab.ipynb)
python scripts/train.py --arch vgg16 --smoke --no-pretrained
python scripts/evaluate.py --checkpoint models/vgg16_s0_smoke.pt --smoke
python scripts/pack_for_colab.py                                     # -> colab/code.zip, data_raw.zip, data_rq2.zip
# on Colab (notebooks/10_train_colab.ipynb), for each of vgg16, resnet50, mobilenet_v2:
python scripts/train.py --arch vgg16 --seed 0 --resume
python scripts/evaluate.py --checkpoint models/vgg16_s0.pt --variants raw masked reinhard histmatch
python scripts/summarise_results.py --model vgg16_s0                 # rq1/F1/rq2/rq3 tables + R_ figure
python scripts/compare_models.py                                     # RQ1: paired differences in drop
# figures for the Methodology Overview (docs/methodology/figures/), the MO as LaTeX/PDF
# (TinyTeX: pdflatex + biber) and the ELSA report as PDF (needs Microsoft Edge)
python scripts/make_mo_figures.py
python scripts/build_mo_latex.py
python scripts/build_mo_pdf.py docs/elsa_report.md
```

The Methodology Overview (D3) is [`docs/methodology_overview_v3.md`](docs/methodology_overview_v3.md)
(PDF in the literature review's LaTeX layout: [`docs/methodology_overview_v3.pdf`](docs/methodology_overview_v3.pdf),
built from [`docs/methodology/mo_v3.tex`](docs/methodology/mo_v3.tex)); the earlier
Markdown version is [`docs/methodology_overview.md`](docs/methodology_overview.md).
The ELSA report (D5: the final commented checklist, mitigations and panel-review reflections) is
[`docs/elsa_report.md`](docs/elsa_report.md); the earlier working checklist is
[`docs/elsa_checklist.md`](docs/elsa_checklist.md).

## Where to look at the cells

Every crop a model is scored on is saved as a PNG, organised by dataset, split and class,
so it can be browsed in File Explorer:

| Folder | What is in it |
|---|---|
| `cell_images_NIH/cell_images/{Parasitized,Uninfected}/` | NIH training and hold-out cells, as released |
| `data/crops/bbbc041/{site_a,site_b}/<class>/` | BBBC041 cells cut from the smears, background kept (RQ1 input) |
| `data/crops/mpidb_wholecell/<species>/<stage>/` | MP-IDB parasites framed on their host red cell (RQ1 input) |
| `data/crops/masked/otsu-gray/...` | the same cells, background removed and cut tight (RQ2 step 1) |
| `data/crops/colour/reinhard/...` | then colour-matched to NIH with Reinhard transfer (RQ2 step 2) |
| `data/crops/colour/histmatch/...` | or with histogram matching (RQ2 step 2) |
| `data/crops/nihpoly/`, `data/crops/mpidb/` | EDA only: NIH recut from polygons, MP-IDB parasite close-ups |

Two artefacts are visible in the colour-matched crops. Very dark BBBC041 site_a pixels fall
below the black level (`imagestats.BLACK_LEVEL`) and are treated as padding, leaving black
holes. Reinhard transfer also amplifies the noise on flat, grey site_b cells, because it
stretches their small colour spread to NIH's.

## Models

`malaria/models.py` builds VGG-16, ResNet-50 or MobileNetV2 from torchvision's ImageNet
weights with a one-logit head. Every layer is fine-tuned and asserted trainable.
`scripts/train.py` uses one fixed schedule for all three: Adam at 1e-4, batch 32, at most
20 epochs, and early stopping on NIH validation loss with patience 3.

Training augmentation is flips and 90° rotations only. There is no colour or brightness
jitter, so RQ2's test-time colour step is not partly done in training. This is a
departure from the proposal, which listed "minor brightness adjustments" (decided
29 Sept 2026). `malaria/metrics.py` holds the Table 2 definitions: the 0.5 threshold,
the drop in percentage points, the share recovered, and 95% intervals from resampling
source images.

Training runs on a Colab T4 through `notebooks/10_train_colab.ipynb`, which resumes
from the per-epoch checkpoint on Drive after a disconnect.

There are two EDA notebooks:

- `notebooks/03_eda_short.ipynb` is the one to read first: a ten-minute version
  (about 1,600 words) that puts the findings → decisions summary first, with six short
  evidence sections after it.
- `notebooks/01_eda.ipynb` is the full record and appendix, generated as described below.

03 measures MP-IDB on the whole-cell crops that every RQ evaluates. It is edited
directly in Jupyter.

`build_crops.py` skips crops that already exist, so re-running is cheap. Pass
`--rebuild` to re-encode everything, `--dry-run` to build manifests only.

The notebook is generated by `scripts/make_notebook.py` rather than edited by
hand. Change the analysis there and regenerate, so the code stays diffable.

## Layout

| Path | What it is |
|---|---|
| `Makefile` | the five stages (`make all`), on the sample by default or the full data (`FULL=1`) |
| `tests/fixtures/sample/` | committed sample of 340 crops with its manifests (`scripts/make_sample.py`) |
| `outputs/sample/` | results of `make all` on the sample (regenerated; only the log is tracked) |
| `malaria/paths.py` | every dataset root and output directory; edit here only |
| `malaria/crops.py` | **the cropping rule**, documented in the module docstring |
| `malaria/manifests.py` | per-cell manifests, one schema for all three datasets |
| `malaria/imagestats.py` | the per-image features used by the EDA |
| `malaria/splits.py` | the slide-grouped NIH train/val/test split |
| `malaria/plots.py` | figure helpers, one palette across every figure |
| `malaria/background.py` | background removal for the test crops, and the Dice / retention scores |
| `malaria/colour.py` | colour spaces (HSV, Lab, YCbCr, Giemsa and H&E deconvolution) and colour transforms |
| `NIH-NLM-ThinBloodSmearsPf/` | the full NIH photographs with expert outlines (not in version control) |
| `data/crops/nihpoly/` | NIH cells recut from the Polygon Set: `raw`, `masked`, and the outline `mask` |
| `data/crops/mpidb_wholecell/` | MP-IDB crops framed on the host red cell rather than the parasite |
| `data/crops/masked/<method>/` | test crops with the background removed, in the NIH format |
| `data/manifests/` | `all_cells.csv`, per-dataset manifests, `nih_split.csv` |
| `data/crops/` | cached cell crops for BBBC041 and MP-IDB (not in version control) |
| `outputs/figures/` | the EDA figures |
| `outputs/tables/` | `cell_stats.csv` plus the summary tables |

## Decisions that differ from the proposal

**MP-IDB crops are regenerated from the `gt/` masks, not taken from `crops/`.**
The dataset ships `crops/` for Falciparum and Vivax only, and the two use
different conventions (Falciparum background-masked, Vivax not). Regenerating all
four species under one rule gives RQ3 its full species breakdown and removes a
confound between crop format and species. Expert stage labels are preserved by
matching left-to-right against the shipped crops; the component decomposition was
validated against the authors' own on all 144 images that have crops.

**BBBC041 is treated as two test sets.** `training.json` (1208 images, 1200x1600
PNG) and `test.json` (120 images, 1383x1944 JPG) are two acquisition batches that
differ from each other about as much as either differs from NIH. They are labelled
`site_a` / `site_b` by that observable rather than by country, which the source
publication does not pin down per file.

**BBBC041 `difficult` and `leukocyte` cells are excluded from the primary metric**
and reported separately: `difficult` is ambiguous by the annotators' own account,
and leukocytes are a class the model never saw, since the NIH negative class is
uninfected red blood cells.

**The NIH split is grouped by slide** (`C###` prefix, 200 slides). A random split
puts cells from one slide on both sides and inflates the baseline that every
cross-dataset drop is measured against.

**MP-IDB has 210 source images in this copy, not the 229 the proposal quotes**
(104 Falciparum, 37 Malariae, 29 Ovale, 40 Vivax). Section 4.1 of the proposal
should be corrected to the count actually used.

**RQ2 tests the causes of the cross-dataset drop, not robustness to blur, noise
and contrast** (decided 15 September 2026). Section J of the notebook shows that
the test domains sit on both sides of NIH on sharpness: blurring moves NIH away
from the three sharper sets (site_a, Falciparum, Malariae) and reaches only site_b,
Ovale and Vivax, and additive noise raises sharpness past site_a and Malariae
without making NIH resemble them; the sweeps cannot reproduce the shift to BBBC041
or MP-IDB. Kanamugire & Djoumessi (2026) and Hou et al. (2026) have also measured
such corruptions on NIH cells. RQ2 now asks what share of the RQ1 drop (percentage points of
sensitivity and specificity) is recovered when test cells are matched to NIH at
test time, first in crop format (background removed) and then in stain colour
(Reinhard or histogram matching). Every dataset is cropped as a whole cell in
every RQ, so MP-IDB is evaluated on `mpidb_wholecell` and RQ2's crop-format step
only removes the background. The trained models stay fixed:
there is no colour-augmentation arm and no retraining on NIH cells with their
background.

## Data checks behind the method choices

Notebook sections O, P and Q cover the NIH source release, background removal on
the test crops, and the cells in other colour spaces.

**NIH-NLM-ThinBloodSmearsPf is the source of cell_images, not a new dataset**
(Kassim et al., 2020). 192 of the 200 cell_images patients and 960 of its 965
photographs are shared, and per-photograph parasitised counts agree exactly on 137
of 165 Polygon Set photographs and 690 of 795 Point Set ones
(`outputs/tables/nih_source_overlap.csv`). cell_images kept 1,087 of 1,142
annotated parasitised cells on the Polygon Set photographs but only 1,814 of 33,071
uninfected ones. The release is therefore never a test set, and every recut cell
inherits its patient's split from `nih_split.csv`. Its real prevalence, 3.3%
parasitised, is close to BBBC041 site_a's 2.7%.

**NIH cells recut from the polygons reproduce the cell_images format.** The masked
variant, cut tight to the outline, matches cell_images crops from the same
photographs on tissue brightness, contrast, colour and sharpness (separation 0.00 to
0.10); it is about 7 px wider and slightly blacker (0.30 vs 0.26 to 0.29) because the
hand-drawn outlines sit just outside the automatic segmentation. The raw variant
keeps the background, which is the test-set format.

**General-purpose background removal fails on these cells; an Otsu threshold does
not.** Scored against the NIH hand-drawn outlines and on whether MP-IDB parasite
pixels survive (`outputs/tables/background_method_summary.csv`): rembg's U^2-Net
finds no foreground on 21% of NIH crops (median Dice 0.03) and 48% of Falciparum
crops; IS-Net on 49% and 89%. Both were trained on everyday photographs, and a pale
red cell on pale plasma is not a salient object to them. Lowering U^2-Net's mask cut-off to 0.05, 0.15 or 0.3 does not rescue it (best
median Dice 0.15, `background_method_summary_u2net_threshold.csv`). BiRefNet needs about 31 s
per crop on this CPU, some 750 CPU-hours for the test sets, and was not run. An Otsu
threshold on brightness never comes back empty and reaches median Dice 0.90, so it is
the method used. Otsu merges cells that touch, so on BBBC041 the kept
region can include a neighbour. The annotated parasite pixels are added back to
every MP-IDB mask, so no parasite is ever removed.

**MP-IDB crops are framed on the host red cell** (`build_crops.py --mpidb-wholecell`,
`background.host_cell_box`). MP-IDB annotates parasites, not cells, so the original
crops were parasite close-ups (Falciparum median 78 px) with the cell edge outside.
Each parasite is reframed on the red cell holding it: Otsu in a window three cell
widths across (`MPIDB_CELL_SIDE`, 140 px, measured on MP-IDB), with touching cells
cut apart by a distance-transform watershed. A region wider than 1.4 cell widths is
treated as an unsplit clump (2.0 for P. vivax, P. ovale and gametocytes, which
enlarge or elongate their host) and falls back to a cell-sized square centred on the
parasite. The host cell was found for 74% of Falciparum parasites (338 fall back),
86% of Malariae and every Ovale and Vivax parasite; median crop sides are now 168,
164, 201 and 206 px. Background removal on these crops keeps 98 to 100% of each
parasite's pixels (Falciparum close-ups: 68%). Parasites, stage labels and cell IDs
are unchanged, which the build asserts. Every RQ evaluates these whole-cell crops;
the parasite close-ups are not an evaluation set.

**BBBC041 is P. vivax** (Broad Institute dataset page). NIH is P. falciparum only, so
every NIH-to-BBBC041 result mixes a species shift with the imaging shift; MP-IDB
Falciparum is the only same-species external test set.

**Colour separates the datasets, along different axes** (notebook section Q,
`outputs/tables/colour_channel_stats.csv`). On tissue pixels of the NIH-format crops,
Giemsa colour deconvolution (scikit-image `bex_from_rgb`) shows BBBC041 cells
absorbing three to four times as much methyl blue as NIH cells (median 0.065 to 0.090
against 0.022; separation 0.97 to 1.00) and MP-IDB cells up to about twice as much.
The direction differs per test set: site_a is darker and bluer, site_b darker and
nearly grey, Falciparum yellow-brown, Vivax and Ovale blue at close to NIH brightness.
Shifting hue and saturation toward NIH moves methyl blue further away (separation 0.88
against 0.72), while matching Lab statistics (Reinhard) or channel histograms brings
the blue-yellow axis and both stain channels close to NIH
(`outputs/tables/colour_transform_effect.csv`). A Lab- or stain-based normalisation is
therefore the candidate for the colour step of RQ2.

## Main EDA findings

See section M of the notebook. In short: the largest NIH-vs-test difference is
crop format, not imaging. NIH cells are segmented onto black and about a quarter
of every NIH crop is padding, which inflates whole-crop brightness, contrast and
sharpness enough to reverse all three comparisons, and dilutes every whole-crop
colour average the same way, so all colour comparisons are made tissue-only.
Measured that way, tissue brightness (separation 0.96), sharpness (0.98) and
stain colour (0.80) all differ, but sharpness differs in both directions: site_a
and Malariae are sharper than NIH and site_b is blurrier, which is why blur is a
poor proxy for the cross-dataset shift. On the stain axis NIH matches Falciparum
while BBBC041 flips its sign entirely.

Effect sizes are reported both as Cohen's d and as a rank-based separation, and
the second is the one to trust here: several features are skewed enough that d is
unreliable on them.
