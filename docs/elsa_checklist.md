# ELSA checklist: cross-dataset malaria classification

This is the living ELSA checklist for D5 (the ELSA worksheet). **Part A** takes the
[deon](https://deon.drivendata.org) base checklist, generated with
`python -m deon -o checklist.md` (deon 0.3.0), and adds project-specific items marked
**+**. **Part B** gives a mitigation for each item marked (Part B). These
mitigations also appear in the Methodology Overview (`docs/methodology/mo.tex`,
section "Ethical and legal aspects").

> Draft status (29 Sept 2026): the added items and the mitigations were drafted with AI
> assistance and still need rewriting in my own words. Facts in them (licences, counts,
> code references) were checked against the dataset pages and the repository.

## Part A: checklist

### A. Data collection
- [x] **A.1 Informed consent.** The images come from patients. Consent was the
  responsibility of the original studies ([Rajaraman et al., 2018](https://doi.org/10.7717/peerj.4568);
  [Ljosa et al., 2012](https://doi.org/10.1038/nmeth.2083);
  [Loddo et al., 2019](https://doi.org/10.1007/978-3-030-13835-6_7)). This project
  collects no new data.
- [ ] (Part B) **A.2 Collection bias.** Each dataset comes from a single site, and NIH has a
  single species and a curated 50/50 class balance.
- [x] **A.3 Limit PII exposure.** The images carry no personal data. NIH patients are
  identified only by a code (`C###`).
- [ ] (Part B) **A.4 Downstream bias mitigation.** No release has sex, age or ethnicity, so
  results cannot be tested for these groups.
- [ ] (Part B) **+ A.5 Licence and terms of reuse.** BBBC041 is CC BY-NC-SA 3.0
  (non-commercial, share-alike). NIH-NLM-ThinBloodSmearsPf requires its notice to be kept
  and NLM to be credited. MP-IDB is MIT.
- [ ] **+ A.6 Label quality and provenance.** Who annotated each dataset, and how
  uncertain are the labels? Examples: BBBC041 `difficult` cells, and MP-IDB stage labels
  recovered from filenames.
- [ ] **+ A.7 Overlapping releases.** NIH-NLM-ThinBloodSmearsPf photographs the same
  patients as cell_images.

### B. Data storage
- [x] **B.1 Data security.** The data are public and have no PII. They are stored
  locally and in a private Google Drive folder for Colab.
- [x] **B.2 Right to be forgotten.** Not applicable: no individual can be identified.
  A takedown by a data owner is handled by deleting the local and Drive copies.
- [ ] **B.3 Data retention plan.** Delete the Drive copies (`colab/*.zip`) after the
  final panel review, and keep the manifests, which are enough to rebuild everything.
- [ ] **+ B.4 Redistribution of derived data.** The crops are derived from BBBC041 and
  fall under share-alike.
- [ ] **+ B.5 Cloud processing.** The training data are uploaded to Google Drive and
  Colab.

### C. Analysis
- [ ] **C.1 Missing perspectives.** Clinicians and microscopists from endemic regions
  were not consulted. Limitation: domain input comes from the literature and the
  supervisor.
- [ ] (Part B) **C.2 Dataset bias.** Class balance at training time (50%) differs from
  prevalence in the field (2.7–5.1% in BBBC041).
- [x] **C.3 Honest representation.** Distributions are shown instead of only means, and
  every value has an image-resampled interval.
- [x] **C.4 Privacy in analysis.** No PII is available to display.
- [x] **C.5 Auditability.** Pinned requirements, a pinned split, seeds, checks built into
  the scripts, and a README run order.
- [ ] (Part B) **+ C.6 Leakage between patients.** Cells from one slide share its stain and
  lighting.
- [ ] **+ C.7 Confounded shifts.** BBBC041 changes species (*P. vivax*) and imaging at the
  same time.

### D. Modeling
- [ ] **D.1 Proxy discrimination.** There are no demographic variables to proxy for.
  Crop format, meaning the black NIH frame, could still act as a shortcut; see D.8.
- [ ] (Part B) **D.2 Fairness across groups.** Error rates by the subgroups that do exist:
  acquisition site, species and life stage.
- [ ] (Part B) **D.3 Metric selection.** What accuracy hides at low prevalence.
- [ ] (Part B) **D.4 Explainability.** Can a decision be explained?
- [ ] **D.5 Communicate bias.** Limits are stated in the Methodology Overview and in the
  final report.
- [ ] **+ D.6 Architecture and deployment context.** The heavy models (VGG-16 has 134 M
  parameters) do not fit the low-resource clinics this work is meant for.
- [ ] **+ D.7 Fixed operating point.** A threshold tuned on test data would overstate
  performance.
- [ ] (Part B) **+ D.8 Shortcut learning.** The model may learn the NIH black frame instead of
  the parasite.

### E. Deployment
- [ ] (Part B) **E.1 Redress, and + regulatory status.** A diagnostic model on the market would
  be a medical device under the EU MDR, and a high-risk AI system under the EU AI Act.
- [x] **E.2 Roll back.** Not applicable: nothing is deployed.
- [ ] **E.3 Concept drift.** This is what the project studies: cross-dataset shift.
- [ ] **E.4 Unintended use.** The models must not be used on patients. The README and
  the report say so.
- [ ] **+ E.5 Untested populations.** There are no smears from sub-Saharan Africa, which
  carries most of the malaria burden.

## Part B: mitigations (items marked Part B)

| Item | How this project deals with it | Where |
|---|---|---|
| A.2 Collection bias | Each source is treated as a separate domain and reported apart: site_a and site_b, and *P. falciparum* against the other species. There is no single pooled score that one site could dominate. The claims are limited to these three sources. | `data.test_cells`, `summarise_results.py` |
| A.4 No protected attributes | The report states this as a limitation, not as a test that passed: disparities by sex, age or ethnicity cannot be measured. The subgroups that can be measured (site, species, stage) are analysed instead in RQ3. | MO §3.4, RQ3 |
| A.5 Licences | Only code, manifests and summary tables go into the repository. Raw images and crops stay out of git, and so do the Drive copies, which are kept for this non-commercial study only. BBBC041's share-alike term therefore never applies to a redistributed file. The report credits NLM and cites each dataset paper. | `.gitignore`, MO Table 1 |
| C.2 Prevalence mismatch | The NIH classes are not rebalanced, because the model trains at NIH's 50/50 split. The mismatch is handled at evaluation instead: sensitivity and specificity are reported separately, and neither depends on prevalence. Accuracy is not reported for BBBC041. | `metrics.py` |
| C.6 Patient leakage | The NIH split is by patient, pinned in `nih_split.csv`, and checked before every training run. The overlapping NIH-NLM release is never used as a test set. | `splits.py`, `data.nih_cells` |
| D.2 Disparate error rates | Sensitivity is compared across species (with an interval on the difference) and across BBBC041 life stages (share of misses and miss rate). Specificity is compared across the two BBBC041 sites. Every interval comes from resampling source images. | RQ3, `summarise_results.py` |
| D.3 Metric choice | A missed parasite (patient left untreated) and a false alarm (unnecessary antimalarials) cost different things, so they are reported apart. The threshold is fixed at 0.5 before testing. AUC is only a threshold-free check. | `metrics.THRESHOLD` |
| D.4 Explainability | Saliency methods (Grad-CAM) are out of scope, as the proposal already stated. RQ2 is a controlled test of what the model depends on: if removing the background or matching colour changes its decisions, the model was using those cues. | RQ2 |
| D.8 Shortcut learning | RQ2's first step puts the test cells into the NIH black-frame format. The size of the recovery then shows how much the model relied on the frame instead of on the parasite. | `build_masked_crops.py` |
| E.1 Regulation and redress | This is a research prototype, trained and tested on public data and never used on patients. The EU AI Act excludes AI systems developed only for scientific research (Art. 2(6)). A deployed diagnostic version would be a medical device (MDR 2017/745) and a high-risk AI system, requiring clinical validation, which this project does not do. The report says this plainly. | README, final report |
