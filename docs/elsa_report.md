# ELSA report: cross-dataset malaria classification

**Deliverable D5, Capstone Portfolio**<br>
**Programme:** MSc Data Science & Society, Tilburg University<br>
**Student:** Noah Tjon Sien Kie (2103939)<br>
**Supervisor:** Dr. Sharon Ong<br>
**Project:** Cross-Dataset Generalisation of Deep Learning for Malaria Classification in Microscopy Images<br>
**Repository:** https://github.com/noahtjonsk/malaria-cross-dataset<br>
**Date:** October 2026

## Purpose

This report is the final, commented ELSA checklist for the project. Part A started from the deon base checklist, generated with `python -m deon -o checklist.md` (deon 0.3.0), and adds items specific to this project, marked **+**, at least two per category. Part B gives a mitigation for each item judged critical, marked (Part B), and says where in the code or the Methodology Overview it is implemented. The last section answers the five example panel-review questions of the D5 worksheet. The same mitigations appear in the Methodology Overview ([`methodology_overview_v3.md`](methodology_overview_v3.md), §2.3). The earlier working version of the checklist is [`elsa_checklist.md`](elsa_checklist.md).

A ticked box means the item is dealt with: mitigated, shown not to apply, or stated as a limitation. An open box means work remains.

## Part A: checklist

### A. Data collection
- [x] **A.1 Informed consent.** The images come from patients. Consent was the responsibility of the original studies (Rajaraman et al., 2018; Ljosa et al., 2012; Loddo et al., 2019). This project collects no new data.
- [x] (Part B) **A.2 Collection bias.** Each dataset comes from a single source, and NIH has a single species and a curated 50/50 class balance.
- [x] **A.3 Limit PII exposure.** The images carry no personal data. NIH patients are identified only by a code (`C###`), which is used to group the split and for nothing else.
- [x] (Part B) **A.4 Downstream bias mitigation.** No release records sex, age or ethnicity, so results cannot be tested for these groups.
- [x] (Part B) **+ A.5 Licence and terms of reuse.** BBBC041 is CC BY-NC-SA 3.0 (non-commercial, share-alike). NIH-NLM-ThinBloodSmearsPf requires its notice to be kept and NLM to be credited. MP-IDB is MIT-licensed.
- [x] (Part B) **+ A.6 Label quality and provenance.** Who annotated each dataset, and how uncertain are the labels? BBBC041's annotators marked some cells `difficult`, and MP-IDB's life-stage labels had to be traced from its file names to individual parasites.
- [x] (Part B) **+ A.7 Overlapping releases.** NIH-NLM-ThinBloodSmearsPf photographs the same patients as cell_images.

### B. Data storage
- [x] **B.1 Data security.** The data are public and contain no PII. They are stored locally and in a private Google Drive folder for Colab.
- [x] **B.2 Right to be forgotten.** Not applicable, because no individual can be identified. A takedown by a data owner would be handled by deleting the local and Drive copies.
- [ ] **B.3 Data retention plan.** After the final panel review, the Drive copies (`colab/*.zip`, trained weights, per-cell predictions) are deleted. The manifests and the pinned split stay in the repository; with the public datasets they are enough to rebuild everything. Open until the deletion is done.
- [x] **+ B.4 Redistribution of derived data.** The crops are derived from BBBC041 and fall under its share-alike term. None is published (see A.5).
- [x] **+ B.5 Cloud processing.** The training and test crops are uploaded to a private Google Drive folder and processed on Colab. The data are public and PII-free, and the copies are deleted under B.3.

### C. Analysis
- [ ] **C.1 Missing perspectives.** Clinicians and microscopists from endemic regions were not consulted. Domain input comes from the literature and the supervisor, and this is stated as a limitation.
- [x] (Part B) **C.2 Dataset bias.** The class balance at training time (50%) differs from prevalence in the field (2.7% and 5.1% in the two BBBC041 batches).
- [x] **C.3 Honest representation.** Distributions are shown instead of only means, and every reported value has an interval from resampling source images.
- [x] **C.4 Privacy in analysis.** No PII is available to display.
- [x] **C.5 Auditability.** Pinned requirements, a pinned split, fixed seeds, checks built into the scripts, a README run order, and a `run.json` per evaluated model that records the code commit, manifest hashes and crop digests.
- [x] (Part B) **+ C.6 Leakage between patients.** Cells from one slide share its stain and lighting, so cells of one patient on both sides of the split would inflate the hold-out score.
- [x] **+ C.7 Confounded shifts.** BBBC041 changes species (*P. vivax*) and imaging at the same time. MP-IDB holds four species from one laboratory, so RQ3's comparison of *P. falciparum* with the other species is the one place where species is varied without changing the site.

### D. Modeling
- [x] **D.1 Proxy discrimination.** There are no demographic variables to proxy for. Crop format, meaning the black NIH frame, could still act as a shortcut; see D.8.
- [x] (Part B) **D.2 Fairness across groups.** Error rates are compared across the subgroups that exist: acquisition batch, species and life stage.
- [x] (Part B) **D.3 Metric selection.** Accuracy hides missed parasites at low prevalence.
- [x] (Part B) **D.4 Explainability.** Can the models' behaviour be explained?
- [x] **D.5 Communicate bias.** The limits are stated in the Methodology Overview (§2.3 and the model information sheet in Appendix B), in this report, and will be repeated in the final report.
- [x] **+ D.6 Architecture and deployment context.** VGG-16 has 134.3 million parameters, too many for the low-resource clinics this work is meant for. RQ1 therefore includes MobileNetV2 (2.2 million). At seed 0 the lighter model did not lose less on the external sets, which the final report will discuss.
- [x] **+ D.7 Fixed operating point.** A threshold tuned on test data would overstate performance. The threshold is fixed at 0.5 before testing, and no test set is used to move it.
- [x] (Part B) **+ D.8 Shortcut learning.** The model may learn the NIH black frame instead of the parasite.

### E. Deployment
- [x] (Part B) **E.1 Redress, and + regulatory status.** A diagnostic model on the market would be a medical device under the EU MDR, and a high-risk AI system under the EU AI Act.
- [x] **E.2 Roll back.** Not applicable: nothing is deployed.
- [x] **E.3 Concept drift.** This is what the project studies: the shift from one dataset to another.
- [x] **E.4 Unintended use.** The models must not be used on patients. The README, the model information sheet and this report say so.
- [x] (Part B) **+ E.5 Untested populations.** There are no smears from sub-Saharan Africa, which carries most of the malaria burden.
- [x] (Part B) **+ E.6 Automation bias.** If a model like this were used next to a microscope, a confident wrong call could be trusted over the microscopist's own reading.

## Part B: mitigations

| Item | How this project deals with it | Where |
|---|---|---|
| A.2 Collection bias | Each source is treated as a separate domain and reported apart: site_a and site_b, and *P. falciparum* against the other species. There is no single pooled score that one site could dominate. The claims are limited to these three sources. | `data.test_cells`, `summarise_results.py`; MO §5.6 |
| A.4 No protected attributes | This is stated as a limitation, not as a test that passed: disparities by sex, age or ethnicity cannot be measured. The subgroups that can be measured (acquisition batch, species, life stage) are analysed instead in RQ3. | MO §2.3 and §5.5 |
| A.5 Licences | Only code, manifests and summary tables go into the repository. Raw images and crops stay out of git, and so do the Drive copies, which are kept for this non-commercial study only. BBBC041's share-alike term therefore never applies to a redistributed file, and each dataset paper is cited and NLM credited. | `.gitignore`; MO §2.1 |
| A.6 Label quality | BBBC041's annotators marked 446 cells as `difficult`; these and the 103 leukocytes stay in the manifests but are left out of the primary metric. For each of MP-IDB's 210 images, the tracked `mpidb_stage_audit.csv` records how many parasites it holds, how many crops were shipped and where the stage labels came from. NIH labels are used as released; checking them again is out of scope and is stated as a limitation. | `manifests.py`, `data.test_cells`, `mpidb_stage_audit.csv`; MO §3.4 |
| A.7 Overlapping releases | 192 of the 200 cell_images patients and 960 of the release's 965 photographs appear in both NIH releases (`nih_source_overlap.csv`), so NIH-NLM-ThinBloodSmearsPf is never used as a test set. It is used only to recut NIH cells for the EDA, and every recut cell inherits its patient's split (`inherit_nih_split`), so no patient crosses splits even there. | `check_nih_overlap.py`, `splits.py`; MO §2.1 and §3.3 |
| C.2 Prevalence mismatch | The NIH classes are not rebalanced, because the model trains at NIH's 50/50 split. The mismatch is handled at evaluation instead: sensitivity and specificity are reported separately, and neither depends on prevalence. Accuracy is not reported for BBBC041, and F1 is reported only as a secondary measure. | `metrics.py`; MO §5.5 |
| C.6 Patient leakage | The NIH split is by patient, pinned in `nih_split.csv`, and checked before every training run. The overlapping NIH-NLM release is never used as a test set. | `splits.py`, `data.nih_cells`; MO §3.2 and §3.3 |
| D.2 Disparate error rates | Sensitivity is compared across species, with an interval on the difference, and across BBBC041 life stages, as share of misses and miss rate. Specificity is compared across the two BBBC041 batches. Every interval comes from resampling source images. | `summarise_results.py`; MO §5.5 |
| D.3 Metric choice | A missed parasite (a patient left untreated) and a false alarm (unnecessary antimalarials) cost different things, so they are reported apart. The threshold is fixed at 0.5 before testing. AUC is only a threshold-free check. | `metrics.THRESHOLD`; MO §5.5 |
| D.4 Explainability | Saliency methods such as Grad-CAM are out of scope, as the proposal already stated. RQ2 is a controlled test of what the models depend on: if removing the background or matching colour changes their decisions, the models were using those cues. | RQ2; MO §5.2 |
| D.8 Shortcut learning | RQ2's first step puts the test cells into the NIH black-frame format. The share of the drop this recovers shows how much the models relied on the frame instead of the parasite. | `build_masked_crops.py`; MO §5.2 |
| E.1 Regulation and redress | This is a research prototype, trained and tested on public data and never used on patients. The EU AI Act excludes AI systems developed only for scientific research (Art. 2(6)). A deployed diagnostic version would be a medical device (MDR 2017/745) and a high-risk AI system, requiring clinical validation, which this project does not do. | README; MO Appendix B |
| E.5 Untested populations | The three sources are a hospital in Chittagong (NIH), the two acquisition batches of BBBC041 and a laboratory in Lausanne (MP-IDB); none is from sub-Saharan Africa. Every claim is limited to these sources, and each is reported on its own so that no pooled score hides one of them. The gap is stated as a limitation in the Methodology Overview and the final report. | MO §2.3 |
| E.6 Automation bias | The models are research prototypes and are never used on patients; the README, the model information sheet and this report say so. Each output is a probability cut at a fixed 0.5 threshold, not a diagnosis, and the seed-0 results show how often it is wrong on new data: sensitivity on MP-IDB *P. falciparum* is between 28.6% and 57.4% across the three models. Any clinical use would need a microscopist to confirm every call and a separate clinical validation. | README; MO Appendix B and §5.6 |

## Panel-review reflections

### Which laws and regulations might apply?

The project is research on public data, and no model is used on patients. The EU AI Act excludes AI systems developed and put into service only for scientific research (Regulation (EU) 2024/1689, Art. 2(6)). A diagnostic version placed on the market would be a medical device under the MDR (Regulation (EU) 2017/745) and therefore a high-risk AI system, which would need clinical validation; this project does not attempt that. The dataset licences apply directly: BBBC041 is CC BY-NC-SA 3.0, NIH-NLM-ThinBloodSmearsPf requires NLM's notice and credit, and MP-IDB is MIT-licensed. The repository therefore publishes code, manifests and summary tables, and no image or crop.

### How representative are the datasets, and which attributes are missing?

Not very, and the project says so. NIH comes from one hospital in Chittagong, one staining protocol and one smartphone-on-microscope setup, with a single species, *P. falciparum*. Its 50/50 class balance is curated: in the source photographs only 3.3% of annotated cells are parasitised. BBBC041 adds *P. vivax* in two acquisition batches, and MP-IDB adds four species from one laboratory in Lausanne, but no smears come from sub-Saharan Africa. The attributes that would have helped are patient sex, age and ethnicity, parasite density per slide, the staining protocol and stain batch, the microscope and camera, and the origin of each BBBC041 batch. Without them, results cannot be broken down by patient group, and stain or device effects can only be read from the images themselves.

### How was bias identified and mitigated?

Bias was looked for at three points. In the data, the EDA compared every source on brightness, colour and crop format, and found that crop format and Giemsa stain colour separate NIH from the test sets (Methodology Overview §4). In the split, NIH is divided by patient and checked before every run, so cells from one slide never sit on both sides. In the evaluation, every source is reported on its own, sensitivity and specificity are reported separately because BBBC041 is only 2.7 to 5.1% parasitised, and RQ3 breaks the misses down by species and life stage. The 0.5 threshold was fixed before testing. Bias by sex, age or ethnicity cannot be checked, because no release records them.

### To what extent do the results generalise?

Generalisation is what the project measures. At seed 0, all three models do well on NIH patients they never saw (AUC 0.996 to 0.997), and all three lose on at least one metric on every external set. The largest drops are in sensitivity on MP-IDB *P. falciparum*: 40.9 percentage points for VGG-16, 44.6 for ResNet-50 and 68.5 for MobileNetV2. VGG-16 also loses 31.0 points of specificity on BBBC041 site_a. The design kept the external sets out of every choice from the start: they are never used for training, early stopping, the colour reference or the threshold, and the NIH split is by patient, so the NIH hold-out score is not inflated by slides shared with training. These are single-seed results; three seeds per model will show how much they vary.

### To what extent are the results explainable?

No saliency method such as Grad-CAM is used; this was out of scope from the proposal on. Instead, RQ2 is a controlled test of what the models rely on: the test cells are changed one property at a time while the models stay fixed. At seed 0, removing the background raised sensitivity on MP-IDB *P. falciparum* from 57.4% to 71.9% for VGG-16 and from 52.6% to 63.9% for ResNet-50. This suggests that both models partly rely on the black NIH frame. Histogram matching pushed almost every cell to parasitised, so changing the colour moved the scores more than it improved how well the classes were separated. These results explain behaviour in terms of input properties, not single decisions.

## References

European Parliament and Council of the European Union. (2017). Regulation (EU) 2017/745 on medical devices. *Official Journal of the European Union, L 117*, 1–175. https://eur-lex.europa.eu/eli/reg/2017/745/oj

European Parliament and Council of the European Union. (2024). Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act). *Official Journal of the European Union, L 2024/1689*. https://eur-lex.europa.eu/eli/reg/2024/1689/oj

Ljosa, V., Sokolnicki, K. L., & Carpenter, A. E. (2012). Annotated high-throughput microscopy image sets for validation. *Nature Methods, 9*(7), 637. https://doi.org/10.1038/nmeth.2083

Loddo, A., Di Ruberto, C., Kocher, M., & Prod'Hom, G. (2019). MP-IDB: The Malaria Parasite Image Database for image processing and analysis. In *Processing and Analysis of Biomedical Information* (pp. 57–65). Springer. https://doi.org/10.1007/978-3-030-13835-6_7

Rajaraman, S., Antani, S. K., Poostchi, M., Silamut, K., Hossain, M. A., Maude, R. J., Jaeger, S., & Thoma, G. R. (2018). Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images. *PeerJ, 6*, e4568. https://doi.org/10.7717/peerj.4568
