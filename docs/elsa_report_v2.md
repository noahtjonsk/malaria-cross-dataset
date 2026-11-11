# ELSA report: cross-dataset malaria classification

**Deliverable D5, Capstone Portfolio**<br>
**Project:** Cross-Dataset Generalisation of Deep Learning for Malaria Classification in Microscopy Images<br>
**Student:** Noah Tjon Sien Kie (2103939), MSc Data Science & Society, Tilburg University<br>
**Supervisor:** Dr. Sharon Ong<br>
**Repository:** https://gitlab.uvt.nl/capstone/2026-fall/u927271<br>
**Date:** November 2026

## 1. Societal relevance and stakeholders

Malaria caused an estimated 282 million cases and 610,000 deaths in 2024 (World Health Organization, 2025). About 95% of both were in the WHO African Region, where about three quarters of the deaths were children under five (World Health Organization, n.d.). Diagnosis still depends on a trained microscopist reading a stained blood smear (Rajaraman et al., 2018).

Software that sorts red blood cells into infected and healthy could help where microscopists are scarce. Published models score almost perfectly on the public NIH dataset, and that score is what a clinic or funder sees first. The question of this project is whether the score holds in another laboratory. Three networks were trained on NIH cells and tested, unchanged, on two other datasets (BBBC041 and MP-IDB). On NIH patients they had never seen, they find 97 to 98% of infected cells (sensitivity) and clear 97 to 98% of healthy cells (specificity). On the other datasets sensitivity falls by up to 63 percentage points on average. The project's value to society is that measurement: a good score at home says little about another laboratory.

| Stakeholder | What is at stake for them |
|---|---|
| Patients in malaria regions | They would bear the errors. A missed infection goes untreated; a false alarm means medicine they do not need. |
| Microscopists | They would work next to the model and could come to trust a confident wrong answer. |
| Clinics, funders and regulators | They decide on buying and approving such tools, and need proof from their own laboratory. |
| Researchers | They can reuse the test protocol and the code. |

## 2. ELSA analysis, mitigation and results

The issues come from the project's ELSA checklist ([`elsa_checklist.md`](elsa_checklist.md)). Each topic below gives the issue, what was done, what was found and what remains.

### 2.1 Data: fit, quality and representativeness

**Issue.** Each dataset comes from one place. NIH is one hospital in Bangladesh, one staining method, one camera setup and one parasite species. BBBC041 has another species, photographed in two batches. MP-IDB has four species from one Swiss laboratory and no healthy cells. None is from sub-Saharan Africa, and none records the patients' sex, age or ethnicity.

**What was done.** Every source is scored on its own, so a strong one cannot hide a weak one. Cells that the BBBC041 annotators marked as doubtful are left out of the scores.

**What was found.** The data fit the question asked, which is transfer between public datasets. They do not support a claim about performance in the field.

**What remains.** The region with 95% of the burden is untested. Results cannot be split by patient group. In BBBC041 the species and the laboratory change together, so the loss there cannot be assigned to one of them.

### 2.2 Ethics and law

**Issue.** The images come from patients, the datasets have licences, and diagnostic software is regulated.

**What was done.** No new data were collected. The images are public and anonymous; NIH patients appear only as a code, which is used to keep each patient on one side of the split. Consent was handled by the original studies. The licences are followed: BBBC041 is non-commercial and share-alike (CC BY-NC-SA 3.0), NIH asks for credit, MP-IDB is MIT. The repository holds code, tables and a 340-crop sample shared under the same licence; the full datasets and the trained models are not redistributed.

**What was found.** Privacy law (GDPR) does not apply to this work, because no person can be identified. The EU AI Act exempts systems built only for research (Regulation (EU) 2024/1689, Art. 2(6)).

**What remains.** A product would be different. Software that reads a blood sample to support a diagnosis is an in vitro diagnostic device (Regulation (EU) 2017/746; Medical Device Coordination Group, 2019), which also makes it a high-risk AI system. It would need clinical evidence this project does not have, and if it stored smears with patient records it would process health data. The non-commercial licence also means these models could not be sold. One task is open: the copies on Google Drive are deleted after the panel review.

### 2.3 Bias: found, tested, mitigated

| Issue | What was done | What was found |
|---|---|---|
| Cells of one patient in both training and test data would inflate the score | Split by patient, checked before every run | Training, validation and test scores agree within 1.2 points, so the NIH score is honest |
| Half the training cells are infected; in BBBC041 only 3 to 5% are | Missed infections and false alarms are reported separately, never as one accuracy figure | On one BBBC041 batch the largest model still finds 96% of infected cells, but only about 1 in 9 of its "infected" calls is right |
| An average can hide a group the model fails on | Results split by batch, species and parasite life stage | The species the models trained on is detected less often than the others in 8 of 9 runs; the youngest stage (rings) is the largest share of misses in 5 of 6 cases |

Two further safeguards: the decision threshold was fixed before testing and the external datasets were never used for any choice, and every model was trained three times. The repeats mattered. The same model lost between 13 and 31 points of specificity on one batch depending only on the random start.

**What remains.** These steps made the measurement fair. They did not make the models fair. Two corrections were tested on the test images, removing the background and matching the stain colour, and neither improved how well the models rank infected above healthy cells on average. The smallest model, the one a low-resource clinic could run, lost the most: it finds 38% of infected cells on one batch. The gaps between groups are reported as remaining bias. Bias by sex, age or ethnicity could not be checked.

### 2.4 Transparency and explainability

**Issue.** A neural network with millions of weights cannot be read like a rule. A user cannot see why a cell was called infected.

**What was done.** For transparency, the code, settings, data split and random seeds are in the repository, and every result file records the code version and data it came from. For explainability, two methods were used, the second added because of the first. The first changes one property of the test images at a time (background, then colour) while the model stays the same, which shows what the models depend on across a whole dataset. It suggested the background mattered little. Grad-CAM (Selvaraju et al., 2017), which draws a heat map of where a model looks in one image, was then added to check that on single cells.

**What was found.** The heat maps agree: the models mostly look at the parasite, for the cells they miss as much as for the cells they find. The failures therefore look like a recognition problem, not a shortcut on the background.

**What remains.** Grad-CAM was run on one training run and 100 cells per group, and the smallest model often returned empty maps for the cells it missed. A heat map also does not tell a microscopist why a call is wrong.

## 3. Personal stance: should this be implemented?

**Verdict.** Deploying these models, or any model checked only on NIH, is not desirable. The study and its test protocol are.

**Benefit.** Faster reading of smears where microscopists are scarce.

**Why not deploy.**

1. The errors are large and unpredictable. One model on one batch misses about six in ten infected cells; another model on the other batch raises about eight false alarms for every correct call. Which failure a clinic gets depends on the model and even on the random start, so the NIH score cannot predict it.
2. The people who carry most of the burden are not in the data, and the species the models learned from is the one they detect worst.
3. The simple fixes did not work, and a confident wrong answer next to a microscope can push aside a correct human reading.

**Why the study was worth doing.** The mitigations did their job: the reference score is honest, errors are split by type and by group, and the effect of chance is visible. The cost was low: about three and a half hours of free cloud GPU time, public data, no new patients. Clinicians were not consulted, which is a gap.

**What would make a tool like this acceptable.**

1. Developers train on several laboratories and regions, including sub-Saharan Africa, and report results per site and per repeat.
2. Each laboratory tests the tool on its own labelled smears, sets the threshold there, and has a microscopist confirm every call.
3. Performance is monitored and re-checked whenever the stain batch or camera changes, and regulators ask for evidence from more than one site.

Until then this work should be used as a warning and a test protocol, not as a diagnostic.

## References

European Parliament and Council of the European Union. (2017). Regulation (EU) 2017/746 on in vitro diagnostic medical devices. *Official Journal of the European Union, L 117*, 176–332. https://eur-lex.europa.eu/eli/reg/2017/746/oj

European Parliament and Council of the European Union. (2024). Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act). *Official Journal of the European Union, L 2024/1689*. https://eur-lex.europa.eu/eli/reg/2024/1689/oj

Medical Device Coordination Group. (2019). *MDCG 2019-11: Guidance on qualification and classification of software in Regulation (EU) 2017/745 (MDR) and Regulation (EU) 2017/746 (IVDR)*. European Commission. https://health.ec.europa.eu/system/files/2020-09/md_mdcg_2019_11_guidance_en_0.pdf

Rajaraman, S., Antani, S. K., Poostchi, M., Silamut, K., Hossain, M. A., Maude, R. J., Jaeger, S., & Thoma, G. R. (2018). Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images. *PeerJ, 6*, e4568. https://doi.org/10.7717/peerj.4568

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *2017 IEEE International Conference on Computer Vision (ICCV)* (pp. 618–626). https://doi.org/10.1109/ICCV.2017.74

World Health Organization. (2025). *World malaria report 2025*. https://www.who.int/teams/global-malaria-programme/reports/world-malaria-report-2025

World Health Organization. (n.d.). *Malaria* [Fact sheet]. Retrieved November 10, 2026, from https://www.who.int/news-room/fact-sheets/detail/malaria
