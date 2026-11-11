# ELSA report: cross-dataset malaria classification

**Deliverable D5, Capstone Portfolio**<br>
**Project:** Cross-Dataset Generalisation of Deep Learning for Malaria Classification in Microscopy Images<br>
**Course:** Capstone Project, MSc Data Science & Society, Tilburg University<br>
**Student:** Noah Tjon Sien Kie (2103939)<br>
**Supervisor:** Dr. Sharon Ong<br>
**Repository:** https://gitlab.uvt.nl/capstone/2026-fall/u927271<br>
**Date:** 11 November 2026

## 1. Societal relevance and stakeholders

Malaria caused an estimated 282 million cases and 610,000 deaths in 2024 (World Health Organization, 2025). About 95% of both were in the WHO African Region, where about three quarters of the deaths were children under five (World Health Organization, n.d.). Diagnosis still depends on a trained microscopist reading a stained blood smear (Rajaraman et al., 2018).

Software that sorts red blood cells into infected and healthy could help where microscopists are scarce. The question of this project is whether a near-perfect score on the public NIH dataset holds in another laboratory. Three networks were trained on NIH cells and tested, unchanged, on two other datasets (BBBC041 and MP-IDB). On NIH patients they had never seen, they find 97 to 98% of infected cells (sensitivity) and clear 97 to 98% of healthy cells (specificity). On the other datasets sensitivity falls by up to 63 percentage points on average. The project's value to society is that measurement: a good score at home says little about another laboratory.

| Stakeholder | What is at stake for them |
|---|---|
| Patients in malaria regions | They would bear the errors. A missed infection goes untreated; a false alarm means medicine they do not need. |
| Microscopists | They would work next to the model and could come to trust a confident wrong answer. |
| Clinics, funders and regulators | They decide on buying and approving such tools, and need proof from their own laboratory. |
| Researchers | They can reuse the test protocol and the code. |

## 2. ELSA analysis, mitigation and results

The issues come from the project's ELSA checklist ([`elsa_checklist.md`](elsa_checklist.md)). Their mitigations were planned in the Methodology Overview ([`methodology_overview_v4.md`](methodology_overview_v4.md), §2.3 and Table 2; fair comparison in §5.4).

### 2.1 Data: fit, quality and representativeness

Each dataset comes from one place. NIH is one hospital in Bangladesh, one staining method, one camera setup and one parasite species. BBBC041 has another species, photographed in two batches. MP-IDB has four species from one Swiss laboratory and no healthy cells. None is from sub-Saharan Africa, and none records the patients' sex, age or ethnicity. For this reason every source is scored on its own, so a strong one cannot hide a weak one, and cells that the BBBC041 annotators marked as doubtful are left out of the scores.

The data therefore fit the question asked, which is transfer between public datasets, but they do not support a claim about performance in the field. Two limits remain. The region with 95% of the burden is untested, and in BBBC041 the species and the laboratory change together, so the loss there cannot be assigned to one of them.

### 2.2 Ethics and law

No new data were collected. The images are public and anonymous; NIH patients appear only as a code, which is used to keep each patient on one side of the split, and consent was handled by the original studies. Privacy law (GDPR) therefore does not apply to this work, because no person can be identified. The EU AI Act exempts systems built only for research (Regulation (EU) 2024/1689, Art. 2(6)). The licences are followed (BBBC041 CC BY-NC-SA 3.0, non-commercial and share-alike; NIH credit; MP-IDB MIT): only code, tables and a 340-crop sample are shared, not the datasets or the trained models.

A product would fall under stricter rules. Software that reads a blood sample to support a diagnosis is an in vitro diagnostic device (Regulation (EU) 2017/746; Medical Device Coordination Group, 2019), which also makes it a high-risk AI system. The Methodology Overview named the general medical device regulation (2017/745); the in vitro diagnostic regulation is the one that applies. It would need clinical evidence this project does not have, and if it stored smears with patient records it would process health data. The non-commercial licence also means these models could not be sold.

### 2.3 Bias: found, tested, mitigated

| Issue | What was done | What was found |
|---|---|---|
| Cells of one patient in training and test sets inflate the score | Split by patient, checked before every run | Training, validation and test scores agree within 1.2 points |
| Half the training cells are infected; in BBBC041 only 3 to 5% are | Missed infections and false alarms reported separately, no single accuracy | Largest model, one batch: finds 96% of infected cells, but only 1 in 9 of its "infected" calls is right |
| An average can hide a group the model fails on | Results split by batch, species and parasite life stage | The training species is detected less often than the others in 8 of 9 runs; rings, the youngest stage, are the largest share of misses in 5 of 6 cases |

Besides these biases in the data, my own choices could bias the results, mainly by tuning until the external results looked good. Every choice was therefore made on NIH data or before testing. The decision threshold of 0.5 was fixed before any test set was seen; the colour reference comes from NIH training cells; the background method was chosen on how well it traced hand-drawn NIH cell outlines, not on classification scores; and the three architectures share every setting (MO §5.4). Every model was trained three times, and the same model lost between 13 and 31 points of specificity on one batch depending only on the random start.

The groups compared here are batches, species and life stages, not people. Fairness is assessed as equal opportunity (Hardt et al., 2016): an infected cell should be found equally often whatever group it belongs to. Sensitivity is the measure because a missed infection is the costlier error, a patient left untreated. Specificity is compared across the two BBBC041 batches. On these measures the models trade one error for the other. VGG-16 keeps sensitivity the most even, with the smallest loss on every external set, but raises the most false alarms on BBBC041 site_a (specificity 77%); MobileNetV2 keeps specificity there at 92% and misses the most infections.

These steps made the measurement fair, but they did not make the models fair. Neither correction tested on the test images, background removal or stain-colour matching, improved on average how well the models rank infected above healthy cells. The gaps between groups are therefore reported as remaining bias.

### 2.4 Transparency and explainability

A network with millions of weights cannot be read like a rule, so a user cannot see why a cell was called infected. What can be made transparent is the process. The code, settings, data split and random seeds are in the repository, and every result file records the code version and data it came from.

For explainability, a test-time ablation changed the background and then the colour of the test images with the models fixed, which shows what they depend on across a dataset. It suggested the background mattered little, so Grad-CAM (Selvaraju et al., 2017), a heat map of where a model looks in one image, was added to check this on single cells. The heat maps agree: the models mostly look at the parasite, for the cells they miss as much as for the cells they find. The failures therefore look like a recognition problem, not a shortcut on the background. Both checks have limits. Grad-CAM was run on one training run and 100 cells per group, and the smallest model often returned empty maps for the cells it missed. A heat map also does not tell a microscopist why a call is wrong.

## 3. Personal stance: should this be implemented?

The possible benefit is faster reading of smears where microscopists are scarce. Even so, deploying these models, or any model checked only on NIH, is not desirable, for four reasons.

1. The errors are large and unpredictable. MobileNetV2, the model a low-resource clinic could run, misses about six in ten infected cells on one batch; VGG-16 raises about eight false alarms per correct call on the other. Which failure a clinic gets depends on the model and even on the random start, which the NIH score does not show.
2. The people who carry most of the burden are not in the data, and the species the models learned from is the one they detect worst.
3. The simple fixes did not work, and a confident wrong answer next to a microscope can push aside a correct human reading.
4. The headline number invites misreading: a non-technical reader or a vendor can quote 97 to 98% as the tool's accuracy, when it holds only for the training hospital. The likely misuse follows: installing the model in an untested laboratory, or letting it replace the microscopist.

The study and its test protocol, on the other hand, were worth doing. They measure the drop honestly and at low cost: three and a half hours of free cloud GPU time, public data, no new patients. A tool like this would become acceptable only under three conditions.

1. Developers train on several laboratories and regions, including sub-Saharan Africa, and report results per site and per repeat.
2. Each laboratory tests the tool on its own labelled smears, sets the threshold there, and has a microscopist confirm every call.
3. Performance is monitored and re-checked whenever the stain batch or camera changes, and regulators ask for evidence from more than one site. Every reported score names the data it was measured on.

## References

European Parliament and Council of the European Union. (2017). Regulation (EU) 2017/746 on in vitro diagnostic medical devices. *Official Journal of the European Union, L 117*, 176–332. https://eur-lex.europa.eu/eli/reg/2017/746/oj

European Parliament and Council of the European Union. (2024). Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act). *Official Journal of the European Union, L 2024/1689*. https://eur-lex.europa.eu/eli/reg/2024/1689/oj

Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. In *Advances in Neural Information Processing Systems 29* (pp. 3315–3323). https://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning

Medical Device Coordination Group. (2019). *MDCG 2019-11: Guidance on qualification and classification of software in Regulation (EU) 2017/745 (MDR) and Regulation (EU) 2017/746 (IVDR)*. European Commission. https://health.ec.europa.eu/system/files/2020-09/md_mdcg_2019_11_guidance_en_0.pdf

Rajaraman, S., Antani, S. K., Poostchi, M., Silamut, K., Hossain, M. A., Maude, R. J., Jaeger, S., & Thoma, G. R. (2018). Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images. *PeerJ, 6*, e4568. https://doi.org/10.7717/peerj.4568

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *2017 IEEE International Conference on Computer Vision (ICCV)* (pp. 618–626). https://doi.org/10.1109/ICCV.2017.74

World Health Organization. (2025). *World malaria report 2025*. https://www.who.int/teams/global-malaria-programme/reports/world-malaria-report-2025

World Health Organization. (n.d.). *Malaria* [Fact sheet]. Retrieved November 10, 2026, from https://www.who.int/news-room/fact-sheets/detail/malaria
