# Data sample

A small, fixed sample of the cells the baseline is trained and scored on, so the
pipeline can be checked end to end without the source datasets (`make smoke`).
Built by `scripts/make_sample.py` from the full manifests; the crops are copied
unchanged.

| Set | Cells |
|---|---|
| NIH cell_images | 40 train, 20 validation and 20 hold-out cells per class, from the pinned patient-grouped split |
| BBBC041 | 30 parasitised and 30 uninfected primary cells per site (site_a, site_b) |
| MP-IDB whole-cell crops | 30 *P. falciparum* and 30 other-species parasites |

The external sets are deliberately enriched for parasitised cells, so results on
the sample show that the code runs, not how large any effect is.

## Sources and licences

The crops are derived from three public datasets and keep their terms:

- **NIH cell_images**: Rajaraman et al. (2018), doi:10.7717/peerj.4568, from the
  U.S. National Library of Medicine
  ([malaria datasheet](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html)).
  Courtesy of the U.S. National Library of Medicine.
- **BBBC041** (`crops/bbbc041/`): Ljosa et al. (2012), doi:10.1038/nmeth.2083;
  Broad Bioimage Benchmark Collection, [BBBC041](https://bbbc.broadinstitute.org/BBBC041).
  Licensed under [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/):
  these derived crops are shared under the same licence, for non-commercial use only.
- **MP-IDB** (`crops/mpidb_wholecell/`): Loddo et al. (2019), doi:10.1007/978-3-030-13835-6_7,
  [MIT licence](https://github.com/andrealoddo/MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis).

These are the only images in the repository. The full datasets are not
redistributed.
