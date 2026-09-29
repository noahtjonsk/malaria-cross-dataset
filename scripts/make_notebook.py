"""Generate notebooks/01_eda.ipynb.

The notebook is generated rather than hand-edited so that it stays diffable and
so the analysis code lives next to the module it calls. Re-run this script after
changing the analysis, then execute the notebook with papermill.
"""
import ast
import sys
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "01_eda.ipynb"

cells = []


def md(text):
    cells.append(new_markdown_cell(text.strip()))


def code(src, tags=()):
    c = new_code_cell(src.strip())
    if tags:
        c.metadata["tags"] = list(tags)
    cells.append(c)


# ---------------------------------------------------------------- intro
md("""
# Cross-dataset malaria classification: exploratory data analysis

Training set NIH (Bangladesh), test sets BBBC041 (Broad) and MP-IDB (CHUV Lausanne).

The project claims the three datasets are different imaging environments and that
a model trained on one will degrade on the others. This notebook measures that
claim instead of asserting it, and answers the questions raised after the proposal
presentation:

- **image intensity and size** across the datasets;
- **the actual differences in lighting, blur and staining** between them, ranked,
  so the RQ2 stress test can be aimed at the factors that really differ;
- **dataset 1 characterised properly**, since the NIH held-out score is the
  baseline every cross-dataset drop is measured against.

Cells are read through the manifests built by `scripts/build_crops.py`, which cut
BBBC041 and MP-IDB cells with one fixed rule (see `malaria/crops.py`). Statistics
come from `scripts/compute_stats.py`. Both must be run before this notebook.
""")

code("""
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from malaria import paths, plots
from malaria.imagestats import (CENTER_FEATURES, FEATURES,
                                domain_gap_table, separation)
from malaria.manifests import load_manifest
from malaria.splits import (build_nih_split, check_no_patient_leakage,
                            split_summary)

plots.set_style()
pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 40)
""", tags=("parameters",))

code("""
stats = pd.read_csv(paths.TABLES / "cell_stats.csv", low_memory=False)
stats = plots.add_domain(stats)
print(f"{len(stats):,} cells x {stats.shape[1]} columns")
print(f"domains: {plots.domains_in(stats)}")
""")

# ---------------------------------------------------------------- A inventory
md("""
## A. What is actually in each dataset

Two things this table has to make explicit. First, BBBC041 ships two annotation
files that are two different acquisition batches (1208 images at 1200x1600 PNG,
120 at 1383x1944 JPG), so they are kept apart throughout as `site_a` and `site_b`.
Second, the parasite classes are rare: BBBC041 is overwhelmingly uninfected red
blood cells, and several MP-IDB stage/species cells number in the single digits.
That decides which per-class claims RQ3 can support.
""")

code("""
inv = (stats.groupby(["dataset", "source_split", "eval_group"], observed=True)
            .size().rename("cells").reset_index())
print(inv.to_string(index=False))
print()
print("class balance within the primary evaluation set:")
prim = stats[stats.eval_group == "primary"]
print(pd.crosstab(prim["domain"], prim["label_binary"],
                  margins=True, dropna=False).to_string())
""")

code("""
# Source-image structure of the test sets. RQ3 recall is estimated on cells that
# cluster within slide images, so the image count is the effective sample size
# for anything that varies between slides, and per-class intervals should be
# clustered by source image rather than treating every cell as independent.
test = stats[stats.dataset != "nih"]
per_img = (test.groupby(["domain", "source_image"], observed=True)
           .agg(cells=("cell_id", "size"),
                parasites=("label_binary", lambda s: int((s == 1).sum())))
           .reset_index())
by_dom = per_img.groupby("domain", observed=True)
img = pd.DataFrame({
    "images": by_dom.size(),
    "images_with_parasite": by_dom["parasites"].apply(lambda s: int((s > 0).sum())),
    "cells_per_image_median": by_dom["cells"].median(),
    "parasites_per_image_median": by_dom["parasites"].apply(
        lambda s: float(s[s > 0].median())),
})
print("test-set structure by source image:")
print(img.to_string())

# Crops clipped by the image edge come out smaller and non-square; they are kept
# and flagged rather than dropped, so the rate is reported here.
border = test["at_border"].astype(str).str.lower().eq("true")
print()
print("crops clipped by the image edge (at_border):")
print(border.groupby(test["domain"], observed=True).agg(rate="mean", n="sum")
      .round(4).to_string())
""")

code("""
# Parasite stage / species counts -- the sample sizes RQ3 has to live with.
par = stats[(stats.label_binary == 1) & stats.stage.notna()]
tab = pd.crosstab(par["domain"], par["stage"], dropna=False)
tab["total"] = tab.sum(axis=1)
print(tab.to_string())
print()
small = tab.drop(columns="total").stack()
small = small[(small > 0) & (small < 30)].sort_values()
print("stage/domain cells with n < 30 (too few for a stable recall estimate):")
print(small.to_string())
""")

code("""
# Wilson intervals show how little these small classes can support. A class with
# n=7 cannot distinguish 'the model catches gametocytes' from 'it misses half'.
from statsmodels.stats.proportion import proportion_confint as _ci  # noqa

def wilson_width(n, p=0.9):
    if n == 0:
        return np.nan
    lo, hi = _ci(int(round(p * n)), n, method="wilson")
    return hi - lo

rows = [{"domain": d, "stage": s, "n": int(n),
         "recall_95CI_width_at_90pct": round(wilson_width(int(n)), 3)}
        for (d, s), n in tab.drop(columns="total").stack().items() if n > 0]
print(pd.DataFrame(rows).sort_values("n").to_string(index=False))
""")

# ---------------------------------------------------------------- H NIH split
md("""
## B. Dataset 1: NIH structure and a defensible split

NIH filenames carry the slide the cell was cut from (the `C###` prefix). Cells
from one slide share a patient, a staining session and a lighting setup, so a
random split scores the model partly on having memorised a slide. Every
cross-dataset drop in this project is measured against the NIH held-out number,
so that baseline is split by slide instead.

The second panel is the more uncomfortable one. NIH negatives do not all come
from the same slides as the positives: a block of uninfected cells comes from
slides that contributed no parasitised cells at all. If those slides look
different, a model can separate the classes partly by recognising the slide
rather than the parasite, which would inflate the within-dataset baseline.
""")

code("""
nih = stats[stats.dataset == "nih"].copy()
per_slide = nih.groupby("patient_id", observed=True).agg(
    cells=("cell_id", "size"), parasitised=("label_binary", "mean"))
print(f"unique slide/patient prefixes: {len(per_slide)}")
print(per_slide["cells"].describe().round(1).to_string())
print()
print("slides by class composition:")
print(pd.cut(per_slide["parasitised"], [-0.01, 0.001, 0.999, 1.01],
             labels=["uninfected cells only", "mixed", "parasitised only"])
      .value_counts().to_string())
""")

code("""
# Do the uninfected-only slides look different from the uninfected cells that
# come from slides which also contain parasites? If they do, part of the NIH
# class boundary is a slide-appearance boundary rather than a parasite boundary.
mixed_slides = set(per_slide.index[(per_slide.parasitised > 0) &
                                   (per_slide.parasitised < 1)])
neg = nih[nih.label_binary == 0].copy()
neg["slide_type"] = np.where(neg.patient_id.isin(mixed_slides),
                             "negative from mixed slide",
                             "negative from uninfected-only slide")
cols = ["gray_mean_center", "gray_std_center", "sat_mean_center",
        "rb_diff_center", "lapvar224_center", "black_frac"]
print(neg.groupby("slide_type")[cols].median().round(2).to_string())
print()
print(neg["slide_type"].value_counts().to_string())

from malaria.imagestats import cohens_d
a = neg[neg.slide_type.str.contains("uninfected-only")]
b = neg[neg.slide_type.str.contains("mixed")]
print()
print("Cohen's d, uninfected-only slides vs mixed slides:")
for c in cols:
    print(f"  {c:14s} d = {cohens_d(np.asarray(a[c]), np.asarray(b[c])):+.3f}")
""")

code("""
# Is that difference actually exploitable? Predict the NIH label from colour
# alone -- no shape, no texture -- grouped by slide so the classifier cannot
# simply memorise a slide. Then repeat on mixed slides only, where every cell has
# a same-slide counterpart of the other class and the slide-composition effect is
# removed by construction. A large gap between the two would mean the class
# boundary is partly 'which slide is this'.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

# Tissue-only colour, so the black-padding fraction cannot leak into the
# prediction disguised as diluted colour.
COLOUR = ["r_mean_center", "g_mean_center", "b_mean_center", "rb_diff_center",
          "sat_mean_center", "hue_mean_center", "gray_mean_center"]

def label_auc(df, feats=COLOUR):
    X = df[feats].to_numpy(float)
    y = df["label_binary"].to_numpy(int)
    g = df["patient_id"].astype(str).to_numpy()
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                 n_jobs=4, random_state=0)  # bounded workers: keeps peak memory low, results identical
    p = cross_val_predict(clf, X, y, groups=g, cv=GroupKFold(n_splits=5),
                          method="predict_proba")[:, 1]
    return roc_auc_score(y, p)

mixed_only = nih[nih.patient_id.isin(mixed_slides)]
print("NIH label predicted from colour alone (grouped by slide):")
print(f"  all cells            AUC = {label_auc(nih):.4f}   n = {len(nih):,}")
print(f"  mixed slides only    AUC = {label_auc(mixed_only):.4f}   n = {len(mixed_only):,}")
""")

md("""
Two things follow, and they point in different directions.

The uninfected-only slides really do look different: tissue saturation d = -1.17,
and the stain axis flips sign entirely, at a median of +47 on mixed slides against
-4 on the uninfected-only ones. That is a large difference inside what is
nominally one class.

But it is mostly not what makes the classes separable. Removing those slides
barely moves the colour-only AUC (0.81 to 0.80), so the bulk of the colour signal
comes from parasitised cells genuinely carrying stained parasite material, which
is real morphology rather than a slide artefact. The honest reading is that NIH
does not have a large slide-identity shortcut, but that a colour-only model still
reaches about 0.81 AUC, so headline NIH accuracy should not be read as evidence
that a model has learned parasite morphology.

The split is grouped by slide regardless, which costs nothing and removes the
question.
""")

code("""
manifest = load_manifest()
nih_split = build_nih_split(manifest)
check_no_patient_leakage(nih_split)
print("no slide appears in more than one split")
print()
print(split_summary(nih_split).round(4).to_string())
print()
print(f"written to {paths.MANIFESTS / 'nih_split.csv'}")
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
axes[0].hist(per_slide["cells"], bins=40, color=plots.DOMAIN_COLORS["nih"])
axes[0].set(title="Cells per NIH slide", xlabel="cells", ylabel="slides")
axes[1].hist(per_slide["parasitised"], bins=30,
             color=plots.DOMAIN_COLORS["nih"])
axes[1].set(title="Parasitised fraction per NIH slide",
            xlabel="fraction parasitised", ylabel="slides")
fig.tight_layout()
plots.save(fig, "B_nih_slide_structure")
""")

md("""
### Duplicates, exact and near

Two checks, both read against the split. An exact duplicate is the same file
twice; a near duplicate is the same cell saved twice with different compression
or a one-pixel shift, which a perceptual hash catches and a byte hash does not.
Either kind straddling train and test would be leakage that no grouping prevents.

Perceptual hashes need care on this data. An uninfected red cell is a featureless
disc, and two featureless discs from different patients can share a hash without
being the same cell. So a hash match is only a candidate: a genuine repeat must
also come from the same patient, and the split is grouped by patient. The table
therefore reports every near match, and separately how many sit within one
patient, which is the only set that can be real repeats.
""")

code("""
import hashlib
import imagehash
from PIL import Image

cells = nih_split.reset_index(drop=True)
md5, bits = {}, []
for cid, p in zip(cells["cell_id"], cells["path"]):
    full = paths.ROOT / p
    md5.setdefault(hashlib.md5(full.read_bytes()).hexdigest(), []).append(cid)
    with Image.open(full) as im:
        bits.append(imagehash.phash(im).hash.flatten())
exact = [v for v in md5.values() if len(v) > 1]
print(f"exact duplicates: {sum(len(v) for v in exact)} files in {len(exact)} groups")

# Hamming distance between every pair of 64-bit hashes, in blocks. Equal bits are
# ones in common plus zeros in common, which is two matrix products.
B = np.asarray(bits, dtype=np.float32)
del bits
ones, zeros = B.T.copy(), (1 - B).T.copy()   # hoisted out of the loop
BLOCK = 400   # keeps each temporary near 45 MB rather than 220 MB
pairs = []
for i0 in range(0, len(B), BLOCK):
    blk = B[i0:i0 + BLOCK]
    ham = 64 - (blk @ ones + (1 - blk) @ zeros)
    ii, jj = np.where(ham <= 4)
    pairs += [(i0 + a, int(b), int(round(ham[a, b])))
              for a, b in zip(ii, jj) if b > i0 + a]
    del ham
pairs = pd.DataFrame(pairs, columns=["i", "j", "d"]).astype(int)  # int even when empty

sp = cells["split"].to_numpy()
pt = cells["patient_id"].astype(str).to_numpy()
lb = cells["label_binary"].astype(int).to_numpy()
rows = []
for thr in (0, 2, 4):
    s = pairs[pairs.d <= thr]
    i, j = s.i.to_numpy(), s.j.to_numpy()
    within = s[pt[i] == pt[j]]
    wi, wj = within.i.to_numpy(), within.j.to_numpy()
    rows.append({"hamming <=": thr, "hash matches": len(s),
                 "different class": int((lb[i] != lb[j]).sum()),
                 "different patient": int((pt[i] != pt[j]).sum()),
                 "same patient (possible repeats)": len(within),
                 "same patient, across splits": int((sp[wi] != sp[wj]).sum())})
print(pd.DataFrame(rows).to_string(index=False))

cand = pairs[pt[pairs.i.to_numpy()] == pt[pairs.j.to_numpy()]].sort_values("d").head(8)
if len(cand):
    fig = plots.contact_sheet(
        [("candidate A", cells.iloc[cand.i.to_numpy()]),
         ("candidate B", cells.iloc[cand.j.to_numpy()])],
        n=len(cand), sample=False,
        title="Same-patient near-duplicate candidates, one pair per column")
    plots.save(fig, "B_near_duplicates")
""")

md("""
No file appears twice. The perceptual hash does produce matches, but they are
collisions between look-alike cells rather than repeats: at Hamming distance 4 or
less, nearly all of them join two different patients and about a third join a
parasitised cell to an uninfected one, which the same cell cannot do. The few
that sit within one patient are the only possible repeats, and they stay on one
side of the split by construction; the eight closest of them are shown above and
are visibly different cells. There is no duplicate leakage across train,
validation and test.
""")

# ---------------------------------------------------------------- size
md("""
## C. Size

`upscale_factor` is the multiplier needed to reach the 224 px network input,
taken on the shorter side, since the square resize stretches that side hardest.
Above 1 the image is being invented by interpolation, and a cell that arrives at
4x upscale carries no more real detail than its original 55 px.
""")

code("""
fig, _ = plots.feature_panels(
    stats, ["w", "area", "upscale_factor"],
    titles={"w": "Crop width (px)", "area": "Crop area (px^2)",
            "upscale_factor": f"Upscale needed to reach {paths.MODEL_INPUT}px"},
    log=("area",), ncols=3)
plots.save(fig, "C_size")

print(stats.groupby("domain", observed=True)[["w", "h", "area", "upscale_factor"]]
      .median().round(2).to_string())
""")

code("""
# The small end of the size distribution. A component 10 px across is upscaled
# more than 20x to reach the network input and carries no morphology, so it
# cannot be scored fairly; these are listed so RQ3 can hold them out the way
# BBBC041 'difficult' cells are, rather than absorb them silently.
small = stats.assign(min_side=stats[["w", "h"]].min(axis=1))
print(small.groupby("domain", observed=True)["min_side"]
      .describe(percentiles=[0.01, 0.05, 0.5]).round(0).to_string())

tiny = (small[(small.dataset == "mpidb") & (small.min_side < 30)]
        .sort_values("min_side"))
print()
print(f"MP-IDB crops with min side < 30 px: {len(tiny)}")
print(tiny[["cell_id", "stage", "w", "h", "upscale_factor"]]
      .round(2).to_string(index=False))

fig = plots.contact_sheet([("smallest 1-10", tiny.iloc[:10]),
                           ("smallest 11-20", tiny.iloc[10:20])],
                          n=10, sample=False,
                          title="The 20 smallest MP-IDB crops, at 224 px")
plots.save(fig, "C_smallest_crops")
""")

# ---------------------------------------------------------------- intensity
md("""
## D. Intensity and lighting

`gray_mean` is overall brightness and `gray_std` is contrast, the two
quantities the follow-up e-mail asked for.

These have to be read twice, because NIH cells are segmented onto a black
background and roughly a quarter of every NIH crop is that padding. Whole-crop
brightness and contrast therefore partly describe the padding rather than the
cell. Both views are reported: the whole-crop numbers describe what the network
is fed, and the `_center` numbers, taken on the central disc with any padding
masked out, describe the imaging itself. The two disagree about NIH in opposite directions, so quoting
only the first would put the wrong conclusion in the thesis.
""")

code("""
fig, _ = plots.feature_panels(
    stats, ["gray_mean", "gray_std", "r_mean", "b_mean",
            "gray_mean_center", "gray_std_center"],
    titles={"gray_mean": "Brightness, whole crop",
            "gray_std": "Contrast, whole crop",
            "r_mean": "Red channel mean, whole crop",
            "b_mean": "Blue channel mean, whole crop",
            "gray_mean_center": "Brightness, tissue only",
            "gray_std_center": "Contrast, tissue only"},
    ncols=3)
plots.save(fig, "D_intensity")

print("whole crop (what the network is fed) vs tissue only (the imaging):")
print(stats.groupby("domain", observed=True)
      [["gray_mean", "gray_mean_center", "gray_std", "gray_std_center"]]
      .median().round(2).to_string())
""")

# ---------------------------------------------------------------- colour
md("""
## E. Staining colour

Giemsa stain puts parasite chromatin in the purple/blue range, so saturation and
the red-minus-blue difference track the staining protocol and the white balance
of the capture device.

Like section D, this has to be read twice. A black padding pixel carries zero
saturation, zero hue and zero red-minus-blue, so on NIH the whole-crop colour
averages are diluted by the mask while the unpadded test sets are measured
honestly. The tissue-only panels compare the staining itself, and they move NIH
substantially: saturation rises from 45 to 65 and the stain axis from +29 to +42
once the padding is excluded. One caveat on hue: OpenCV hue is circular and wraps
at 180, and an arithmetic mean near the wrap is unreliable, so hue supports the
coarse contrasts here rather than fine ones.
""")

code("""
fig, _ = plots.feature_panels(
    stats, ["sat_mean", "rb_diff", "hue_mean",
            "sat_mean_center", "rb_diff_center", "hue_mean_center"],
    titles={"sat_mean": "Saturation, whole crop",
            "rb_diff": "Red - blue, whole crop",
            "hue_mean": "Hue, whole crop",
            "sat_mean_center": "Saturation, tissue only",
            "rb_diff_center": "Red - blue (stain axis), tissue only",
            "hue_mean_center": "Hue, tissue only"},
    ncols=3)
plots.save(fig, "E_stain")

print("whole crop (network input) vs tissue only (the staining itself):")
print(stats.groupby("domain", observed=True)
      [["sat_mean", "sat_mean_center", "rb_diff", "rb_diff_center",
        "hue_mean", "hue_mean_center"]]
      .median().round(2).to_string())
""")

# ---------------------------------------------------------------- sharpness
md("""
## F. Sharpness, and a trap in measuring it

Laplacian variance at the network input size. Measuring it natively would make a
2592 px slide look sharper than a 55 px crop purely because of pixel count, so
everything is compared at 224 px.

There is a second and larger trap. Measured over the whole crop, NIH scores about
575 against 3 to 35 for every test domain, which reads as NIH being one to two
orders of magnitude sharper. It is not. A Laplacian responds to any strong
gradient, and the hard boundary between an NIH cell and its black background is
the strongest gradient in the image.

Removing it takes two things at once, and an earlier version of this analysis
got it wrong in both directions. A central disc alone leaves the padding inside
the window on the 7% of NIH crops whose cell is small enough for it to reach
there; those crops scored a median near 197 against about 6 for the rest, which
inflated NIH's spread enough to make Cohen's d report no sharpness difference
where one exists. A tissue mask alone fixes that but drops the disc, and the
window on an unmasked crop then becomes the whole rectangle, so a single
segmented NIH cell would be compared against a whole field of view and the
difference would be image content rather than imaging.

The window used here is the intersection: the central disc, with the tissue mask
eroded by three pixels so a neighbourhood filter never reads across the padding
boundary. Every crop is kept and the sampled region stays comparable.

Measured that way NIH sits at about 6, and the test domains fall on both sides of
it: MP-IDB Malariae at 34 and BBBC041 site_a at 24 are sharper, site_b at 2.5 is
blurrier, and Falciparum, Ovale and Vivax are close enough to be indistinguishable
from it. That split in direction, not an absence of difference, is what makes blur
a poor stand-in for the cross-dataset shift in RQ2.

site_b, the one JPEG domain, has the lowest tissue sharpness of all, and JPEG
quantisation smooths exactly the fine gradients a Laplacian measures, so part of
that floor may be compression rather than optics.

So the whole-crop figure measures the segmentation mask, not the optics. The
`_center` column is the one that answers the supervisor's question about blur
differences between datasets.
""")

code("""
fig, _ = plots.feature_panels(
    stats, ["lapvar224", "lapvar224_center", "lapvar_native"],
    titles={"lapvar224": "Sharpness, whole crop (mask edge dominates)",
            "lapvar224_center": "Sharpness, tissue only  <-- the real comparison",
            "lapvar_native": "Sharpness at native size (confounded by scale)"},
    log=("lapvar224", "lapvar224_center", "lapvar_native"), ncols=3)
plots.save(fig, "F_sharpness")

sharp = (stats.groupby("domain", observed=True)
         [["lapvar224", "lapvar224_center", "black_frac"]].median().round(2))
sharp["whole/tissue ratio"] = (sharp.lapvar224 / sharp.lapvar224_center).round(1)
print(sharp.to_string())
print()
print("The ratio tracks black_frac almost exactly: the inflation is the mask.")
""")

md("""
### Does parasite position bias the tissue-only comparison?

The `_center` window follows the tissue, but it does not know where within that
tissue a parasite sits.

That leaves a fair objection to the comparison above. Both test sets are cropped
centred on the parasite annotation, so parasite material sits inside the window by
construction. NIH crops are centred on the cell instead, so a parasite sits
wherever it happens to fall within that cell. A parasite is a small dark speck
against pale cytoplasm, which is exactly the gradient a Laplacian responds to. If
it lands inside the window more often on one side of the comparison, part of the
sharpness gap is an artefact of how the crops were cut rather than a property of
the imaging.

NIH is the only dataset carrying both classes, so it is the only place the size of
that effect can be measured directly.
""")

code("""
nih_lab = (stats[stats["dataset"] == "nih"]
           .groupby("label_binary", observed=True)
           [["lapvar224_center", "gray_std_center", "sat_mean_center"]]
           .median().round(2))
nih_lab.index = ["uninfected", "parasitised"]
print("NIH cells by label, measured inside the _center window:")
print(nih_lab.to_string())

uni = nih_lab.loc["uninfected", "lapvar224_center"]
par = nih_lab.loc["parasitised", "lapvar224_center"]
med = stats.groupby("domain", observed=True)["lapvar224_center"].median()
gap = med["bbbc041/site_a"] - med["nih"]

print()
print(f"parasite effect inside the window: {par - uni:+.2f} "
      f"({100 * (par - uni) / uni:+.0f}% over uninfected)")
print(f"NIH to bbbc041/site_a gap:         {gap:+.2f}")
print(f"the effect is {abs(gap) / (par - uni):.0f}x too small to explain the gap")
""")

md("""
The effect is real and it points the way the objection predicted, but it is an
order of magnitude too small to overturn the comparison. Parasitised NIH cells
score about 3 higher inside the window than uninfected ones, while the gap between
NIH and BBBC041 site_a is about 18. An NIH set in which every crop carried a
centred parasite would still sit far below site_a. The test domains really are
sharper on tissue; they are not merely parasite-centred.

The same split is worth noting for a second reason. Contrast and saturation inside
the window also rise with the label, which is the section B result reached from a
different direction: tissue colour alone predicts the NIH label at about AUC 0.81.
""")

# ---------------------------------------------------------------- format
md("""
## G. Crop format: the largest single difference

NIH cells are segmented onto a black background. The two test sets cannot be:
BBBC041 ships bounding boxes with no masks, so masking MP-IDB alone would have
made its Falciparum crops resemble the training data while Vivax crops did not,
confounding the RQ3 species comparison with crop format. Both test sets therefore
get plain rectangles.

The consequence is a genuine train/test difference that is a property of the
sources rather than of the imaging: roughly a quarter of every NIH image is pure
black padding, and essentially none of a test image is. `black_frac_outer`
measures it where it lives, in the corners.
""")

code("""
fig, _ = plots.feature_panels(
    stats, ["black_frac", "black_frac_outer", "black_frac_center"],
    titles={"black_frac": "Near-black pixels (whole crop)",
            "black_frac_outer": "Near-black pixels (outer ring)",
            "black_frac_center": "Near-black pixels (centre)"}, ncols=3)
plots.save(fig, "G_crop_format")

print(stats.groupby("domain", observed=True)
      [["black_frac", "black_frac_outer", "black_frac_center"]]
      .median().round(3).to_string())
""")

# ---------------------------------------------------------------- domain gap
md("""
## H. How far apart are the domains, and along which axis

Two measures on the feature table.

**Cohen's d** ranks the individual properties by how far each test domain sits
from NIH, in pooled standard deviations. This is what tells RQ2 which
degradations are worth sweeping.

**A domain classifier** asks whether the domains are separable at all from these
cheap features. It is grouped by source slide so it cannot pass by memorising a
slide. An AUC near 1.0 means a model does not need to be subtle to tell the
datasets apart, which is the premise of the whole project; it is reported twice,
once on all features and once with every size and mask-sensitive feature removed,
so the result is not just restating that the crops are different shapes.

Both measures are reported because they disagree, and the disagreement is worth
seeing. Cohen's d divides by a pooled standard deviation, which a skewed feature
inflates: `gray_p5` reaches |d| ~ 170 only because NIH's 5th percentile is exactly
0 on every masked crop, collapsing its variance. Separation is a rank measure, so
one extreme crop counts once. Where the two rank features differently, separation
is the one to trust on this data.

Read the ranking from the top and the largest differences between NIH and every
test set are crop-format artefacts: the black padding (`black_frac_outer`,
`gray_p5`), native resolution (`lapvar_native`) and whole-crop contrast.

Among the tissue-only features, which are the ones that describe imaging rather
than crop format, the red channel separates at 0.99, sharpness at 0.98, brightness
at 0.96, saturation at 0.84 and the stain axis at 0.80. So brightness and colour
do separate the domains, but sharpness is not the flat feature an earlier version
of this notebook reported. That reading came from a measuring-window bug described
in section F, which inflated NIH's spread and drove |d| down to 0.3. Corrected,
sharpness reaches |d| 5.8.

The two measures disagree in the other direction too, which is worth a sentence in
the write-up. MP-IDB Falciparum sits at |d| 1.8 on tissue sharpness but at
separation 0.29: its median is close to NIH's and the distributions overlap
heavily, and d is large only because the pooled standard deviation is dominated by
NIH's much larger sample. Reading d alone would put a difference there that a
reader could not act on.

The per-domain detail matters more than the ceiling. Sharpness separates NIH
cleanly from MP-IDB Malariae (0.98), BBBC041 site_a (0.93) and site_b (0.89), and
hardly at all from Falciparum (0.29), Ovale (0.03) or Vivax (0.03). The domains
that differ are split in direction: site_a and Malariae are sharper than NIH,
site_b is blurrier.
""")

code("""
gap = domain_gap_table(stats, reference="nih", group_col="domain")
print("Cohen's d vs NIH (positive = test domain is higher)")
print(gap.round(2).to_string())
gap.round(4).to_csv(paths.TABLES / "domain_gap_cohens_d.csv")
""")

code("""
sep = domain_gap_table(stats, reference="nih", group_col="domain",
                       measure=separation)
print("Rank-based separation vs NIH (0 = indistinguishable, 1 = always separable)")
print(sep.round(2).to_string())
sep.round(4).to_csv(paths.TABLES / "domain_gap_separation.csv")

both = pd.DataFrame({"separation": sep["max_abs"],
                     "cohens_d": gap["max_abs"]}).sort_values(
                         "separation", ascending=False)
print()
print("The two measures ranked side by side:")
print(both.round(2).head(14).to_string())
""")

code("""
fig, ax = plt.subplots(figsize=(9, 6))
# Tissue-only features. Every crop-format feature separates at 1.00 for every
# domain, which is the finding stated above but a flat block of full-width bars
# on a chart. What varies, and what the imaging comparison rests on, is here.
top = (sep.loc[[f for f in CENTER_FEATURES if f in sep.index]]
       .drop(columns="max_abs")
       .sort_values(list(sep.columns[:1]), ascending=False))
top = top.loc[sep.loc[top.index, "max_abs"].sort_values(ascending=False).index]
y = np.arange(len(top))
width = 0.8 / max(1, top.shape[1])
for i, col in enumerate(top.columns):
    ax.barh(y + i * width, top[col], height=width, label=col,
            color=plots.DOMAIN_COLORS.get(col))
ax.set_yticks(y + 0.4 - width / 2)
ax.set_yticklabels(top.index)
ax.invert_yaxis()
ax.axvline(0, color="k", lw=0.8)
# Separation is bounded 0-1, so this needs no log axis and no outlier caveat.
# That is the point of using it: gray_p5 reaches |d| ~ 170 under Cohen's d
# purely because NIH's 5th percentile is 0 on every masked crop, which
# collapses its variance. A rank measure cannot be distorted that way.
ax.set_xlim(0, 1)
ax.set_xlabel("Separation from NIH  (0 = indistinguishable, 1 = always separable)")
ax.set_title("Tissue-only features: which imaging properties differ most from NIH")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
plots.save(fig, "H_domain_gap")
""")

code("""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

# Everything that encodes crop geometry or the segmentation mask is removed, so
# the second AUC cannot be won just by noticing that NIH crops have black
# corners. That includes the whole-crop colour means: black pixels contribute
# zero saturation and zero red-minus-blue, so on masked crops those means
# encode the padding fraction almost as directly as black_frac itself. Only
# the tissue-only variants survive.
APPEARANCE_ONLY = [f for f in FEATURES if f not in
                   {"w", "h", "area", "aspect", "upscale_factor",
                    "black_frac", "black_frac_outer", "black_frac_center",
                    "lapvar_native", "gray_mean", "gray_std", "lapvar224",
                    "gray_p5", "gray_p50", "gray_p95",
                    "r_mean", "g_mean", "b_mean", "rb_diff",
                    "sat_mean", "hue_mean"}]
print("appearance-only features:", APPEARANCE_ONLY)

def domain_auc(target, features):
    sub = stats[stats.domain.astype(str).isin(["nih", target])]
    X = sub[features].to_numpy(float)
    y = (sub.domain.astype(str) == target).astype(int).to_numpy()
    groups = sub["source_image"].astype(str).to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                 n_jobs=4, random_state=0)  # bounded workers: keeps peak memory low, results identical
    p = cross_val_predict(clf, X, y, groups=groups,
                          cv=GroupKFold(n_splits=n_splits),
                          method="predict_proba")[:, 1]
    return roc_auc_score(y, p)

rows = []
for target in plots.domains_in(stats):
    if target == "nih":
        continue
    rows.append({"domain": target,
                 "AUC_all_features": round(domain_auc(target, FEATURES), 4),
                 "AUC_appearance_only": round(domain_auc(target, APPEARANCE_ONLY), 4)})
auc = pd.DataFrame(rows)
print("Can a cheap classifier tell this domain from NIH? (grouped by slide)")
print(auc.to_string(index=False))
auc.to_csv(paths.TABLES / "domain_classifier_auc.csv", index=False)
""")

# ---------------------------------------------------------------- contact
md("""
## I. What the network actually sees

Every crop rendered at 224 px, which is the only fair way to compare them.
""")

code("""
prim = stats[stats.eval_group == "primary"]
rows = [("NIH parasitised", prim[(prim.dataset == "nih") & (prim.label_binary == 1)]),
        ("NIH uninfected", prim[(prim.dataset == "nih") & (prim.label_binary == 0)])]
for d in plots.domains_in(prim):
    if d == "nih":
        continue
    sub = prim[(prim.domain.astype(str) == d) & (prim.label_binary == 1)]
    if len(sub):
        rows.append((f"{d} parasite", sub))
for d in ["bbbc041/site_a", "bbbc041/site_b"]:
    sub = prim[(prim.domain.astype(str) == d) & (prim.label_binary == 0)]
    if len(sub):
        rows.append((f"{d} uninfected", sub))

fig = plots.contact_sheet(rows, n=10,
                          title="Cells as the network sees them (224 px)")
plots.save(fig, "I_contact_sheet")
""")

# ---------------------------------------------------------------- RQ2
md("""
## J. Calibrating the RQ2 degradation levels

RQ2 sweeps blur, noise and contrast over the test images. Picking those severity
levels arbitrarily would make the results hard to interpret: a 15 px blur kernel
means nothing on its own. Instead the same degradations are applied to NIH crops
here and the resulting statistic is plotted against severity, with the real test
domains drawn as horizontal lines. Where a curve crosses a line is the severity
at which degraded NIH matches that dataset on that property, which turns the RQ2
levels into a statement about how much real-world shift each one represents.
""")

code("""
import albumentations as A
import cv2

SAMPLE = 400
nih_sample = stats[stats.dataset == "nih"].sample(SAMPLE, random_state=0)
imgs = [cv2.cvtColor(cv2.imread(str(paths.ROOT / p)), cv2.COLOR_BGR2RGB)
        for p in nih_sample["path"]]
imgs = [cv2.resize(i, (paths.MODEL_INPUT, paths.MODEL_INPUT),
                   interpolation=cv2.INTER_AREA) for i in imgs]

from malaria.imagestats import interior_mask

def measure(batch):
    g = [cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in batch]
    hsv = [cv2.cvtColor(i, cv2.COLOR_RGB2HSV) for i in batch]
    lap = [cv2.Laplacian(x, cv2.CV_64F) for x in g]
    # Same window as the per-cell statistics. The degraded curve is compared
    # against reference lines taken from `stats`, so it has to be measured the
    # same way or the crossings mean nothing.
    m = [interior_mask(x) for x in g]
    return {"lapvar224": float(np.median([x.var() for x in lap])),
            "lapvar224_center": float(np.median([x[mi].var() for x, mi in zip(lap, m)])),
            "gray_std": float(np.median([x.std() for x in g])),
            "gray_std_center": float(np.median([x[mi].std() for x, mi in zip(g, m)])),
            "gray_mean": float(np.median([x.mean() for x in g])),
            "sat_mean": float(np.median([h[..., 1].mean() for h in hsv]))}

sweeps = {
    # k=1 is the undegraded baseline, so each curve starts from where NIH sits
    "gaussian_blur": ([1, 3, 5, 7, 9, 11, 15, 21],
                      lambda k: A.NoOp(p=1.0) if k == 1
                      else A.GaussianBlur(blur_limit=(k, k), p=1.0)),
    "gauss_noise":   ([0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
                      lambda v: A.GaussNoise(std_range=(v, v), p=1.0)),
    "contrast":      ([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4],
                      lambda c: A.RandomBrightnessContrast(
                          brightness_limit=(0, 0), contrast_limit=(c, c), p=1.0)),
}

recs = []
for name, (levels, make) in sweeps.items():
    for lv in levels:
        t = make(lv)
        # Every transform carries its own random generator, seeded from entropy
        # at construction. Pinning it makes the calibration table identical on
        # every execution; the global random/numpy seeds do not reach it.
        t.set_random_seed(0)
        out = [t(image=i)["image"] for i in imgs]
        recs.append({"degradation": name, "severity": lv, **measure(out)})
sweep = pd.DataFrame(recs)
sweep.to_csv(paths.TABLES / "rq2_severity_calibration.csv", index=False)
print(sweep.round(2).to_string(index=False))
""")

code("""
# Matched on the tissue-only statistics: the whole-crop sharpness of an NIH cell
# is pinned by its mask edge, which no amount of blurring removes, so matching on
# it would be matching on the segmentation rather than on the optics.
targets = (stats[stats.dataset != "nih"]
           .groupby("domain", observed=True)
           [["lapvar224_center", "gray_std_center"]].median())

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
panels = [("gaussian_blur", "lapvar224_center", "blur kernel (px)", True),
          ("gauss_noise", "lapvar224_center", "noise std (fraction)", True),
          ("contrast", "gray_std_center", "contrast delta", False)]
for ax, (deg, stat, xlabel, logy) in zip(axes, panels):
    s = sweep[sweep.degradation == deg]
    ax.plot(s["severity"], s[stat], "o-", color=plots.DOMAIN_COLORS["nih"],
            label="NIH, degraded")
    for d, row in targets.iterrows():
        ax.axhline(row[stat], ls="--", lw=1.2,
                   color=plots.DOMAIN_COLORS.get(str(d)), label=str(d))
    if logy:
        ax.set_yscale("log")
    ax.set(xlabel=xlabel, ylabel=stat, title=f"{deg} -> {stat}")
axes[0].legend(fontsize=6, loc="upper right")
fig.tight_layout()
plots.save(fig, "J_rq2_calibration")
""")

md("""
### Reading the calibration

The severity at which the NIH curve crosses a dashed line is the degradation that
reproduces, on that one statistic, the gap to that real dataset.

The result is not the one the proposal assumed, and it is worth stating plainly.

**Blur is close to useless as a domain proxy here.** Measured on tissue, NIH is
already among the least detailed domains (about 6, against 24 for BBBC041 site_a
and 34 for MP-IDB Malariae). Most test domains are *sharper* than the training
data, so blurring NIH moves away from them. Only site_b, Ovale and Vivax sit below
NIH, and they are reached with a small kernel before the curve flattens.

**Contrast has to rise to reach most domains.** NIH tissue contrast is about 4.7,
lower than every test domain except site_b at 3.3; the other five run from 12.6 to
22.8. Reducing contrast reaches only site_b. Raising it stops improving past about
+0.2 because highlights clip, and never reaches site_a at 23.

**Noise moves the measure the wrong way.** It raises Laplacian variance rather
than lowering it, so it travels away from site_b, Ovale and Vivax, the three that
sit below NIH. It does pass through the values of site_a and Malariae on the way
up, and that crossing should not be read as a match: speckle at the pixel level is
not what makes those two datasets look different, and the curve keeps climbing
far past anything in the real data. Matching one statistic is not resembling a
dataset, which is the general caution this whole section carries.

So RQ2 should be reframed. Sweeping blur, noise and contrast still answers a
legitimate robustness question, namely how much degradation the model absorbs
before it breaks. It should not be presented as simulating the cross-dataset
shift, because the measured shift does not lie along those axes. The axes it does
lie along are stain colour, tissue brightness and crop format, and section H
ranks them.
""")

# ---------------------------------------------------------------- summary
md("""
## K. Summary table for the write-up

Brightness, contrast, sharpness, saturation and the stain axis are all tissue-only
measurements, for the reason given in sections E and F: the whole-crop versions
describe the NIH segmentation mask as much as the imaging, and reverse or dilute
the comparison.
""")

code("""
summary = (stats.groupby("domain", observed=True)
           .agg(cells=("cell_id", "size"),
                median_width=("w", "median"),
                upscale=("upscale_factor", "median"),
                brightness=("gray_mean_center", "median"),
                contrast=("gray_std_center", "median"),
                saturation=("sat_mean_center", "median"),
                stain_rb=("rb_diff_center", "median"),
                sharpness=("lapvar224_center", "median"),
                black_frac=("black_frac", "median"))
           .round(2))
print(summary.to_string())
summary.to_csv(paths.TABLES / "domain_summary.csv")
print()
print(f"figures -> {paths.FIGURES}")
print(f"tables  -> {paths.TABLES}")
""")


md("""
## L. Verification

Checks that the cached crops actually correspond to their manifest rows. These
run every time the notebook does, so a silent breakage in the crop pipeline shows
up here rather than in a model result three weeks later.
""")

code("""
import cv2
import json
from malaria.crops import mask_components, read_gray_mask
from malaria.manifests import _pair_img_gt, _shipped_stage_map

man = load_manifest()

# 1. every crop referenced by the manifest exists and decodes
sample = man[man.dataset != "nih"].sample(400, random_state=0)
bad = [q for q in sample["path"] if cv2.imread(str(paths.ROOT / q)) is None]
assert not bad, bad[:5]
print("[ok] 400 sampled crops all decode")

# 2. crop dimensions on disk match the recorded crop window
mismatch = [r["cell_id"] for _, r in sample.iterrows()
            if cv2.imread(str(paths.ROOT / r["path"])).shape[:2]
            != (r["r1"] - r["r0"], r["c1"] - r["c0"])]
assert not mismatch, mismatch[:5]
print("[ok] crop dimensions match the recorded crop windows")

# 3. BBBC041 row count equals the number of annotated objects
n_obj = sum(len(rec["objects"]) for jf in paths.BBBC041_JSON.values()
            for rec in json.loads(jf.read_text()))
assert int((man.dataset == "bbbc041").sum()) == n_obj
print(f"[ok] BBBC041 rows == {n_obj:,} annotated objects")

# 4. MP-IDB crops reproduce the dataset authors' own parasite decomposition
n_checked = 0
for sp in ["Falciparum", "Vivax"]:
    shipped = _shipped_stage_map(sp)
    for stem, _img, gt in _pair_img_gt(sp):
        if stem in shipped:
            assert len(mask_components(read_gray_mask(gt))) == len(shipped[stem]), stem
            n_checked += 1
print(f"[ok] MP-IDB component counts match the shipped crops on {n_checked} images")

# 5. no NIH slide spans two splits. Class balance is reported rather than
# asserted: the split balances cell counts per fold and balances labels only
# as a byproduct, so a different seed could legitimately widen the spread.
check_no_patient_leakage(nih_split)
rates = nih_split.groupby("split", observed=True)["label_binary"].mean()
spread = float(rates.max() - rates.min())
note = "" if spread < 0.02 else "  [warning: above 2 points, check before training]"
print(f"[ok] no slide leakage; parasitised-rate spread across splits {spread:.4f}{note}")
""")

md("""
## M. What this changes

**Section 4.1 of the proposal needs a correction.** MP-IDB ships `crops/` for only
Falciparum and Vivax, and the two use different conventions: Falciparum crops are
background-masked and Vivax crops are not. All MP-IDB cells are therefore regenerated
from the `gt/` masks under the same rule as BBBC041. The expert stage labels are
preserved by matching left to right against the shipped crops, validated against
the authors' own decomposition on all 144 images that have one.

**The largest train/test difference is crop format, not imaging.** NIH cells are
segmented onto black; about a quarter of every NIH crop is padding. That single
fact dominates the ranked domain gap, inflates whole-crop brightness, contrast
and sharpness enough to reverse all three comparisons, and dilutes every
whole-crop colour average the same way, which is why the colour numbers below are
tissue-only.

Colour is what genuinely separates the domains. Measured on tissue, the stain
red-minus-blue axis puts NIH at +42, indistinguishable from MP-IDB Falciparum at
+43, while BBBC041 site_a sits at -24: a sign reversal between the training set
and its largest test set. Saturation runs from 17 (site_b) to 77 (Falciparum)
with NIH at 65, and tissue brightness separates the domains cleanly. A cheap
classifier tells every test domain from NIH at AUC 0.998 or higher from tissue
colour and brightness alone.

Sharpness also differs, which an earlier version of this notebook denied. That
version measured the tissue window as a fixed central disc, and on the 7% of NIH
crops small enough for the padding to reach inside it the Laplacian returned the
mask edge instead of the optics. Those crops inflated NIH's standard deviation
from 4.6 to 77.7 and drove Cohen's d for sharpness to 0.3, which read as no
difference. With the window corrected (section F) sharpness separates NIH from
Malariae at 0.98, site_a at 0.93 and site_b at 0.89, at |d| up to 5.8.

Blur is still the wrong axis for RQ2, but for a reason that has to be stated
correctly. It is not that the domains share a sharpness; it is that they sit on
both sides of NIH. site_a and Malariae are sharper, so blurring the training data
moves away from them; only site_b is reached by blurring at all. The sub-question
survives as a robustness probe, and still cannot be described as simulating the
cross-dataset shift.

**BBBC041 is two datasets.** site_a and site_b differ from each other about as
much as either differs from NIH, and are reported separately throughout.

**RQ3 is sample-size limited.** Sixteen stage/domain combinations have n < 30 and
nine have n < 10 (seven if the two `unknown`-stage rows are left out). Per-class
recall needs confidence intervals, and the smallest classes cannot support a claim
in either direction.

Three further things constrain what RQ3 can claim. The cells cluster within
images: 888 of the 1,208 site_a images and 115 of the 120 site_b images contain a
parasite, at a median of two per image, and the three rarer MP-IDB species are
almost all one parasite per image, so per-class intervals should be clustered by
source image rather than computed as if every cell were an independent trial.
Species recall is also confounded with resolution, because Falciparum rings are
small: their crops have a median side of 78 px (2.9x upscale) against 126 to
147 px for the other three species, so a recall difference between species is
partly a difference in how much detail the network was given. And twenty-nine
crops have a shortest side under 30 px, including one Vivax component of 10 by
11 px, which is too little to score fairly. Report RQ3 with and without those, in
the same way `difficult` cells are held out of the BBBC041 primary metric.

**Accuracy is the wrong headline metric on BBBC041.** The primary set is 2.7%
parasitised on site_a (2,149 of 79,569) and 5.1% on site_b (303 of 5,917), so a
model that calls everything uninfected scores 95 to 97%. Report sensitivity and
specificity separately, with F1 or PR-AUC at a fixed threshold, and keep accuracy
for NIH, where the classes are balanced.

**The EDA points augmentation at colour.** The measured shift is stain colour and
tissue brightness (section H), so training-time hue and saturation jitter plus
brightness shifts are the augmentations the data argues for. Blur augmentation
would rehearse a difference that does not exist between these datasets. The
proposal's "minor brightness adjustments" should be widened to include colour.

On volume, the training side is fine. 19,203 training cells is in line with the
published NIH experiments that reach above 95% from ImageNet-initialised backbones
(Rajaraman et al., 2018), and all three networks are fine-tuned rather than
trained from scratch. The binding constraint sits on the test side, in the small
RQ3 classes.

The split is clean of duplicates too. No NIH file appears twice, and the
perceptual hash matches that do exist join different patients and often different
classes, which marks them as look-alike discs rather than repeats. The few
within-patient candidates cannot cross the patient-grouped split.

**NIH is a single-species dataset.** Its 200 patients are 150 infected with
P. falciparum and 50 uninfected (Rajaraman et al., 2018), which is exactly the
150 mixed and 50 uninfected-only slides found in section B. Every Malariae, Ovale
and Vivax result in RQ3 is therefore species-out-of-distribution, on top of the
imaging shift.

**Next step.** `data/manifests/nih_split.csv` fixes the slide-grouped split, so
MobileNetV2, ResNet-50 and VGG-16 can be fine-tuned on identical rows.
""")


md("""
## N. Answers to the Canvas EDA questions

The Canvas EDA activity asks for readiness to discuss a fixed list of questions at
the Week 3 supervision meeting. Each is answered here in a sentence or two, with
the section that supports it.

**Anomalies or unusual patterns.** The largest NIH-vs-test difference is crop
format, not imaging: a quarter of every NIH crop is black padding, which inflates
whole-crop brightness, contrast and sharpness enough to reverse all three
comparisons, and dilutes the whole-crop colour averages the same way (E, F, G). BBBC041 is two acquisition batches that differ as much from
each other as from NIH (A, H). Fifty NIH patients contributed uninfected cells
only, and those cells are visibly less saturated (B).

**Data quality issues.** No missing labels in the primary sets. A `Thumbs.db` sits
in each NIH class folder; MP-IDB `gt/` and `img/` disagree on three filenames and
its shipped crops use two conventions, so every crop was regenerated under one
rule; 11 MP-IDB cells have an unresolvable stage and are labelled `unknown` (A, L,
`mpidb_stage_audit.csv`). No exact duplicates and no near-duplicate leakage (B).
Twenty-nine MP-IDB crops are under 30 px on a side and should be held out of RQ3
or reported separately (C).

**Risks in the data.** Class imbalance: BBBC041 is 2.7% and 5.1% parasitised and
MP-IDB has no negatives at all (A). Sparsity: sixteen stage/domain cells have
n < 30 and nine have n < 10 (A). Representativeness: NIH is P. falciparum only,
from one hospital, so the three other species in RQ3 are out of distribution (M),
and Falciparum crops are about half the size of the other species' crops (C).

**Implications for preprocessing.** Resize to 224 px from crops stored at native
size; one padded-square cropping rule for both test sets; no background masking of
test crops, so the NIH mask stays a measured train/test difference rather than a
hidden one; BBBC041 `difficult` and `leukocyte` held out of the primary metric
(A, G, `malaria/crops.py`).

**Implications for methodological choices.** Split NIH by patient, not at random
(B). Report site_a and site_b separately (A, H). Treat RQ2 as a robustness probe,
not a simulation of the cross-dataset shift (J). Augment colour and brightness in
training, not blur (M). Cluster RQ3 intervals by source image, and report
sensitivity and specificity rather than accuracy on BBBC041 (M).

**Data leakage risks.** Patient identity: removed by the grouped split and checked
by assertion (B, L). Slide appearance as a label shortcut: tested and found small;
tissue colour alone reaches about AUC 0.81 with or without the uninfected-only
patients (B). Duplicates across splits: none (B).

**Overall quality and suitability.** Good for the main question and RQ1: three
clean, labelled sources, 27,558 balanced training cells, and a domain shift that
is real and measurable (H). Adequate for RQ2 as a robustness question, weak for
RQ2 as a simulation of the real shift (J). Adequate for RQ3 on Falciparum and on
BBBC041 stages, and too thin for a per-class claim on Malariae, Ovale or Vivax, or
on any gametocyte group outside site_a (A).

**Patterns in the target and features.** NIH is balanced by construction, and
every infected patient also contributes uninfected cells (B). In the test sets the
target is rare and clustered, at about two parasites per positive image (A). Among
the features, the domains separate on tissue brightness, colour and sharpness,
though sharpness only for three of the six test domains and in both directions
(D, E, F, H).

**Enough observations for the model?** Yes on the training side: 19,203 training
cells fine-tuning ImageNet-initialised backbones, in line with the published NIH
experiments. The limit is on the test side, in the small RQ3 classes (M).

**A representative sample, inspected.** Section I shows ten cells per group at
network input size; section C adds the twenty smallest MP-IDB crops.

**Duplicates or near-duplicates, including across splits.** None exact; the
perceptual-hash matches are look-alike discs from different patients, and the
within-patient candidates cannot cross the split (B).

**Noisy or inconsistent labels.** BBBC041 `difficult` cells are excluded and
counted; every MP-IDB stage label carries its provenance (shipped crop,
single-stage filename, or ambiguous) in `mpidb_stage_audit.csv` (A). NIH labels
are binary, come from a single expert slide reader (Rajaraman et al., 2018), and
are taken as given; there is no second annotator to measure their noise against.

**Spurious correlations.** The black background is the strongest gradient in an
NIH image, is absent from every test set, and dilutes whole-crop colour averages,
so colour is compared tissue-only (E, G, H); a cheap classifier separates every
test domain from NIH at AUC 0.998 or higher from tissue colour alone (H); tissue
colour alone predicts the NIH label at about AUC 0.81, so headline NIH accuracy
does not show that morphology was learned (B).

**Does imbalance or volume call for augmentation?** Not for volume. For the
measured shift, yes: hue, saturation and brightness augmentation in training is
the mitigation the data points to; blur augmentation is not (M).

**Systematic quality differences across classes or groups.** Uninfected-only NIH
patients are less saturated than the rest (B); site_b is darker, flatter and
JPEG-compressed where site_a is PNG (D, F); Falciparum crops are 78 px against
126 to 147 px for the other species (C).
""")


# ================================================================ week 4
# ---------------------------------------------------------------- O NIH photos
md("""
## O. Week 4: the NIH photographs behind cell_images

After the Week 3 meeting the supervisor pointed to NIH-NLM-ThinBloodSmearsPf
(Kassim et al., 2020): the full 5312x2988 Chittagong photographs, with an expert
outline (Polygon Set, 165 photographs) or centre point (Point Set, 800) for every
cell. Two questions: is it the same dataset as cell_images, and what do the NIH
cells look like when they are cut from those outlines?

`scripts/check_nih_overlap.py` matches the two releases photograph by photograph.
`scripts/build_crops.py --nihpoly` recuts every Polygon Set cell twice:
`polygon_raw`, the padded square with its background kept, which is the test-set
format, and `polygon_masked`, cut tight to the outline with everything outside it
set to 0, which is the cell_images format.
""")

code("""
overlap = pd.read_csv(paths.TABLES / "nih_source_overlap.csv")
print("photographs, by release:")
print(overlap["where"].value_counts().to_string())
per_patient = overlap.groupby("patient_id")["where"].agg(
    lambda s: "both" if (s == "both").any() else s.iloc[0])
print()
print("patients, by release:", per_patient.value_counts().to_dict())
print("patients only in cell_images:", sorted(per_patient[per_patient != "both"].index))

both = overlap[overlap["where"] == "both"].copy()
both["gap"] = both["gt_parasitized"] - both["local_parasitized"]
agreement = both.groupby("set").agg(
    photographs=("photo", "size"),
    parasitised_count_equal=("gap", lambda g: int((g == 0).sum())),
    within_one=("gap", lambda g: int((g.abs() <= 1).sum())),
    annotated_parasitised=("gt_parasitized", "sum"),
    kept_parasitised=("local_parasitized", "sum"),
    annotated_uninfected=("gt_uninfected", "sum"),
    kept_uninfected=("local_uninfected", "sum"))
print()
print(agreement.to_string())
""")

code("""
# The same cells in three formats, restricted to the photographs both releases
# share: cell_images as shipped, and the Polygon Set outlines cut both ways.
poly_stats = pd.read_csv(paths.TABLES / "cell_stats_nihpoly.csv", low_memory=False)
shared_poly = both[both["set"] == "polygon"]
shared_keys = set(zip(shared_poly["patient_id"], shared_poly["photo"]))
nih_rows = stats[stats["dataset"] == "nih"].copy()
nih_rows["photo"] = (nih_rows["source_image"].astype(str)
                     .str.extract(r"(IMG_[0-9]{8}_[0-9]{6}a?)")[0])
on_shared = [k in shared_keys for k in zip(nih_rows["patient_id"].astype(str),
                                           nih_rows["photo"])]
recut = plots.add_domain(pd.concat(
    [nih_rows[on_shared].drop(columns="domain"),
     poly_stats[poly_stats["eval_group"] == "primary"]], ignore_index=True))

recut_cols = ["w", "black_frac", "gray_mean_center", "gray_std_center",
              "rb_diff_center", "sat_mean_center", "lapvar224_center"]
print(recut.groupby(["domain", "label_binary"], observed=True)[recut_cols]
      .median().round(2).to_string())

raw_poly = poly_stats[(poly_stats["source_split"] == "polygon_raw")
                      & (poly_stats["eval_group"] == "primary")]
print()
print(f"parasitised share of annotated red cells on the Polygon Set photographs: "
      f"{raw_poly['label_binary'].mean():.1%} "
      f"(cell_images is 50% by construction; BBBC041 site_a is 2.7%)")
""")

code("""
fig = plots.contact_sheet([
    ("cell_images, parasitised", recut[(recut["domain"] == "nih") & (recut["label_binary"] == 1)]),
    ("polygon masked, parasitised", recut[(recut["domain"] == "nihpoly/polygon_masked") & (recut["label_binary"] == 1)]),
    ("polygon raw, parasitised", recut[(recut["domain"] == "nihpoly/polygon_raw") & (recut["label_binary"] == 1)]),
    ("cell_images, uninfected", recut[(recut["domain"] == "nih") & (recut["label_binary"] == 0)]),
    ("polygon masked, uninfected", recut[(recut["domain"] == "nihpoly/polygon_masked") & (recut["label_binary"] == 0)]),
    ("polygon raw, uninfected", recut[(recut["domain"] == "nihpoly/polygon_raw") & (recut["label_binary"] == 0)]),
], n=10, title="NIH cells from the shared Polygon Set photographs, three formats")
plots.save(fig, "O_nih_recut")
""")

md("""
**Findings.** It is the same acquisition, not a new dataset. 192 of the 200
cell_images patients and 960 of the 965 photographs appear in both releases. The
eight patients only cell_images has are six uninfected-only patients (C1, C203,
C204, C206, C207, C209) and two infected ones whose photographs carry an `a`
suffix (C33, C37). On shared photographs the parasitised counts agree exactly on
137 of 165 Polygon Set photographs (within one on 152) and on 690 of 795 Point Set
photographs. cell_images kept 1,087 of the 1,142 annotated parasitised cells on
the Polygon Set photographs, but only 1,814 of the 33,071 uninfected ones, about
eleven per photograph: that subsampling is how it was balanced to 50/50.

Two consequences follow. The release can never serve as an external test set,
and every recut cell inherits its patient's side of `nih_split.csv`, which
`check_no_patient_leakage` asserts when the manifest is built. And its uninfected
cells restore the real prevalence: 3.3% of the annotated red cells on the Polygon
Set photographs are parasitised, close to BBBC041 site_a's 2.7%, where
cell_images is 50% by construction.

The recut masked cells reproduce the cell_images format. On the shared
photographs, tissue brightness, contrast, stain colour, saturation, hue and
sharpness are indistinguishable from the cell_images crops of the same
photographs (separation 0.00 to 0.10). Only the outline differs: the hand-drawn
polygons (median 17 vertices) sit slightly outside the automatic segmentation, so
the recut crops are about 7 px wider (137 vs 130 px uninfected, 146 vs 139
parasitised) and carry a little more black (0.30 against 0.26 to 0.29). The raw
variant is the same cells with their plasma and neighbours kept, which is the
format the test sets are in.
""")

# ---------------------------------------------------------------- P background
md("""
## P. Background removal on the test crops

Section G found crop format to be the largest train/test difference: every NIH
cell sits on black, while every test crop keeps its plasma and neighbouring
cells. The supervisor asked for background-removal algorithms to be tried on the
test crops.

`malaria/background.py` offers two families. rembg's general-purpose models
(U^2-Net, IS-Net, BiRefNet) were trained on everyday photographs, not blood
smears; Otsu thresholds on brightness and on saturation are the transparent
baseline. Whatever the method, only the component belonging to the cell of
interest is kept (under the crop centre, or touching the annotated parasite for
MP-IDB), its holes are filled, and the result is cut to the square tight around
it, which is how cell_images crops are cut.

`scripts/validate_background.py` scores every method before one is used. On the
recut NIH cells from section O, the hand-drawn outline is a true answer, so
Dice and IoU measure the cell edge. On MP-IDB, parasite retention measures
whether the method keeps the thing being classified. BBBC041 has no outlines,
so only the failure statuses and the contact sheets speak for it.
""")

code("""
BACKGROUND_METHOD = "otsu-gray"   # chosen from the validation table below
# every parasitised test cell plus 3000 uninfected per split (build_masked_crops.py)
MASKED_NAME = BACKGROUND_METHOD.replace(":", "_") + "_sample"

bg = pd.read_csv(paths.TABLES / "background_method_summary.csv")
print("Dice against the NIH hand-drawn outlines (median, 10th percentile):")
print(bg[bg["domain"] == "nihpoly/polygon_raw"]
      .set_index("method")[["n", "dice_median", "dice_p10", "iou_median",
                            "status_nearest", "sec_per_1000"]].to_string())
print()
thr_path = paths.TABLES / "background_method_summary_u2net_threshold.csv"
if thr_path.exists():
    thr = pd.read_csv(thr_path)
    print("U^2-Net with a lower mask cut-off, against the NIH outlines:")
    print(thr[thr["domain"] == "nihpoly/polygon_raw"].set_index("method")
          [["dice_median", "dice_p10", "status_empty", "status_nearest"]].to_string())
    print()
print("MP-IDB parasite retention (median; share of crops keeping >= 99%):")
mp_bg = bg[bg["domain"].str.startswith("mpidb")]
print(mp_bg.pivot(index="method", columns="domain",
                  values="retention_median").round(3).to_string())
print(mp_bg.pivot(index="method", columns="domain",
                  values="retention_ge_0.99").round(2).to_string())
print()
print("BBBC041, no ground truth: failure statuses and kept area:")
print(bg[bg["domain"].str.startswith("bbbc041")]
      .pivot(index="method", columns="domain",
             values=["status_empty", "status_nearest", "kept_frac_median"])
      .round(3).to_string())
""")

code("""
# The methods side by side on the same cells, for the write-up; the per-method
# sheets (P_background_<method>.png) cover every domain.
from malaria.background import apply_mask, cell_mask, foreground

val = pd.read_csv(paths.TABLES / "background_method_validation.csv")
by_id = pd.concat([load_manifest("nihpoly_cells.csv"),
                   load_manifest("bbbc041_cells.csv")]).set_index("cell_id")
labels = {"nihpoly/polygon_raw": "NIH (outlined)", "bbbc041/site_a": "BBBC041 site_a",
          "bbbc041/site_b": "BBBC041 site_b"}
picked = []
for dom in labels:
    ids = val[(val["domain"] == dom) & (val["method"] == "otsu-gray")]["cell_id"].head(2)
    picked += [(labels[dom], by_id.loc[i, "path"]) for i in ids]
compare = ["otsu-gray", "rembg:u2net", "rembg:isnet-general-use"]
fig, axes = plt.subplots(len(compare) + 1, len(picked),
                         figsize=(1.9 * len(picked), 1.9 * (len(compare) + 1)))
for c, (label, path) in enumerate(picked):
    rgb = plots.load_rgb(path)
    axes[0, c].imshow(rgb)
    axes[0, c].set_title(label, fontsize=8)
    for r, method in enumerate(compare, start=1):
        kept, _ = cell_mask(foreground(rgb, method))
        axes[r, c].imshow(apply_mask(rgb, kept))
for r, label in enumerate(["original"] + compare):
    axes[r, 0].set_ylabel(label, fontsize=8)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.tight_layout()
plots.save(fig, "P_background_compare")
""")

code("""
# The chosen method applied to every test crop (scripts/build_masked_crops.py),
# raw and masked side by side for the same cells.
masked_stats = plots.add_domain(pd.read_csv(
    paths.TABLES / f"cell_stats_masked_{MASKED_NAME}.csv", low_memory=False))
# MP-IDB whole-cell crops keep the parasite-framed cell_id, so the raw crop for a
# masked MP-IDB cell has to come from the whole-cell manifest, not from `stats`.
raw_by_id = (pd.concat([stats.loc[stats["dataset"] != "mpidb", ["cell_id", "path"]],
                        load_manifest("mpidb_wholecell_cells.csv")[["cell_id", "path"]]])
             .drop_duplicates("cell_id").set_index("cell_id"))
pairs = []
for dom in plots.domains_in(masked_stats):
    sub = masked_stats[(masked_stats["domain"] == dom) & (masked_stats["label_binary"] == 1)]
    pick = sub.sample(min(8, len(sub)), random_state=0)
    raw = raw_by_id.loc[pick["cell_id"]].reset_index()
    pairs += [(dom.replace("_masked", "") + " (raw)", raw), (dom, pick)]
fig = plots.contact_sheet(pairs, n=8, sample=False,
                          title=f"Test crops before and after {BACKGROUND_METHOD}")
plots.save(fig, "P_masked_test_crops")
""")

md("""
**MP-IDB framed on its host cell.** MP-IDB annotates parasites, not the red cells
they sit in, so its crops were cut around the parasite: close-ups that leave the
cell edge outside the crop, unlike the whole-cell crops of NIH and BBBC041, and
that no background method can put into the NIH format. `background.host_cell_box`
reframes every parasite on its host cell: a window three cell widths across
(`paths.MPIDB_CELL_SIDE`, 140 px, measured on MP-IDB) is thresholded with Otsu,
touching cells are cut apart with a distance-transform watershed, and the crop is
the padded square around the cell region(s) holding the parasite. The parasites,
their order, stage labels and `cell_id`s are unchanged
(`scripts/build_crops.py --mpidb-wholecell` asserts it).
""")

code("""
framing = pd.read_csv(paths.TABLES / "mpidb_wholecell_framing.csv")
print("host-cell framing outcomes:")
print(pd.crosstab(framing["species"], framing["framing"], margins=True).to_string())
print()
print("median sizes, px (crop side includes the 10% padding):")
print(framing.groupby("species")[["parasite_side", "host_side", "crop_side"]]
      .median().to_string())
print()
print("largest crops, for inspection:")
print(framing.nlargest(10, "crop_side")[["cell_id", "stage", "framing",
                                         "parasite_side", "host_side", "crop_side"]]
      .to_string(index=False))

mp_parasite = load_manifest("mpidb_cells.csv")
mp_wholecell = load_manifest("mpidb_wholecell_cells.csv").set_index("cell_id")
framing_pairs = []
for species in paths.MPIDB_SPECIES:
    pick = mp_parasite[mp_parasite["species"] == species].sample(8, random_state=1)
    framing_pairs.append((f"{species}, parasite", pick))
    framing_pairs.append((f"{species}, host cell",
                          mp_wholecell.loc[pick["cell_id"]].reset_index()))
fig = plots.contact_sheet(framing_pairs, n=8, sample=False,
                          title="MP-IDB: framed on the parasite vs on its host red cell")
plots.save(fig, "P_mpidb_framing")
""")

code("""
# Does the format gap close? Whole-crop features, as the network is fed them,
# for NIH, the raw test crops and the masked test crops.
status = pd.read_csv(paths.TABLES / f"masked_{MASKED_NAME}_status.csv")
print("background-removal status on the full test sets:")
print(pd.crosstab([status["dataset"], status["source_split"]], status["status"]).to_string())
print(status.groupby(["dataset", "source_split"])[["kept_frac", "parasite_retention"]]
      .median().round(3).to_string())

fmt = plots.add_domain(pd.concat([stats.drop(columns="domain"),
                                  masked_stats.drop(columns="domain")],
                                 ignore_index=True))
fmt = fmt[fmt["eval_group"] == "primary"]
fmt_cols = ["w", "black_frac", "black_frac_outer", "gray_mean", "gray_std", "lapvar224"]
print()
print(fmt.groupby("domain", observed=True)[fmt_cols].median().round(2).to_string())
print()
print("separation from NIH on the crop-format features (0 = same, 1 = always separable):")
print(domain_gap_table(fmt, features=["black_frac", "black_frac_outer", "black_frac_center",
                                      "gray_mean", "gray_std", "lapvar224"],
                       measure=separation).round(2).to_string())
""")

md("""
**Findings.** General-purpose background removal does not work on these cells, and
a threshold does. Against the hand-drawn NIH outlines, Otsu on brightness reaches a
median Dice of 0.90 (10th percentile 0.75) and never returns an empty mask. rembg's
U^2-Net reaches 0.03: it finds no foreground on 21% of crops and keeps a blob away
from the centre on another 51%; IS-Net finds none on 49%. Lowering U^2-Net's mask
cut-off from 0.5 to 0.05 lifts its median Dice only to 0.15, and 0.15 and 0.3 do
worse, so this is not a threshold setting: a pale red cell on pale plasma is not a
salient object to a model trained on everyday photographs. BiRefNet needs about 31 s
per crop on this CPU, some 750 CPU-hours for the test sets, and was not run. Otsu on
brightness is the method used.

Otsu's own weakness is touching cells, which it merges into one component, so on
BBBC041 the kept region is the cell under the crop centre together with anything
touching it. For MP-IDB, framing the host cell adds a distance-transform watershed
that cuts touching cells apart. A region wider than 1.4 cell widths (2.0 for
P. vivax, P. ovale and gametocytes, which enlarge or elongate their host) is taken
for an unsplit clump and falls back to a cell-sized square centred on the parasite.
The host cell was found for 74% of Falciparum parasites, where smears are densely
packed, 86% of Malariae, and every Ovale and Vivax parasite. The median Falciparum
crop went from 78 to 168 px, alongside the NIH cells.

The whole-cell framing is also what makes background removal work on MP-IDB. On the
parasite close-ups Otsu kept a median 68% of the Falciparum parasite's pixels; on
the host-cell crops it keeps 98 to 100% for every species, before the annotated
pixels are added back.
""")

# ---------------------------------------------------------------- Q colour
md("""
## Q. The cells in other colour spaces

The supervisor suggested converting RGB to other colour spaces to see what the
cells look like there. RGB mixes three things the comparison needs apart: which
colour a pixel is, how strong that colour is, and how bright it is.
`malaria/colour.py` measures every cell in:

- **HSV**: hue (which colour), saturation (how strong), value (how bright);
- **CIE Lab**: lightness apart from a green-red axis `a` and a blue-yellow axis `b`;
- **YCbCr**: luma apart from blue- and red-difference chroma;
- **Giemsa deconvolution** with scikit-image's `bex_from_rgb` matrix (methyl blue
  and eosin): how much of each stain a pixel absorbed. This is the Giemsa
  counterpart of the H&E deconvolution that the stain-standardisation literature
  builds on;
- **H&E deconvolution** (`rgb2hed`), for reference only, since these slides are
  not H&E-stained.

All means are over tissue pixels, on the test crops in the NIH format from
section P. Hue is weighted by saturation, because a pale pixel's hue is close to
arbitrary.
""")

code("""
from malaria import colour

cstats = plots.add_domain(pd.read_csv(paths.TABLES / "colour_channel_stats.csv"))

# Hue wraps at 360, and NIH cells are either pink (near 330-360) or lavender (near
# 240-280), so hue is compared as a signed angle from the NIH circular mean rather
# than as a raw number of degrees; 350 and 10 are close, not 340 apart.
nih_hue = np.deg2rad(cstats.loc[cstats["domain"] == "nih", "H"].to_numpy())
nih_hue_mean = np.rad2deg(np.arctan2(np.sin(nih_hue).mean(), np.cos(nih_hue).mean())) % 360
cstats["H_from_nih"] = ((cstats["H"] - nih_hue_mean + 180) % 360) - 180
gap_channels = ["H_from_nih"] + [c for c in colour.CHANNELS if c != "H"]
print(f"NIH circular mean hue: {nih_hue_mean:.1f} degrees")
print(cstats.groupby(["domain", "label_binary"], observed=True)[gap_channels]
      .median().round(3).to_string())

colour_gap = domain_gap_table(cstats, features=gap_channels, measure=separation)
colour_gap = colour_gap.drop(columns="max_abs")
fig, ax = plt.subplots(figsize=(0.9 * colour_gap.shape[1] + 2.5, 0.33 * len(colour_gap) + 1.6))
grid = colour_gap.to_numpy(dtype=float)
im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
ax.set_xticks(range(colour_gap.shape[1]))
ax.set_xticklabels(colour_gap.columns, rotation=40, ha="right")
ax.set_yticks(range(len(colour_gap)))
ax.set_yticklabels(colour_gap.index)
for (i, j), v in np.ndenumerate(grid):
    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
            color="white" if v < 0.6 else "black")
ax.grid(False)
fig.colorbar(im, ax=ax, label="separation from NIH")
ax.set_title("Which colour channel tells each domain apart from NIH")
plots.save(fig, "Q_colour_separation")
""")

code("""
# One parasitised cell per domain, decomposed. Each channel uses one colour scale
# across all rows, so a darker or brighter panel means a real difference.
show = ["H", "S", "V", "L", "a", "b", "giemsa_blue", "giemsa_eosin"]
examples = []
for dom in plots.domains_in(cstats):
    sub = cstats[(cstats["domain"] == dom) & (cstats["label_binary"] == 1)]
    if len(sub):
        rgb = plots.load_rgb(sub.sample(1, random_state=3).iloc[0]["path"], paths.MODEL_INPUT)
        examples.append((dom, rgb, colour.channel_maps(rgb), colour.tissue_mask(rgb)))
limits = {ch: np.percentile(np.concatenate([m[ch][t] for _, _, m, t in examples]), [2, 98])
          for ch in show}
fig, axes = plt.subplots(len(examples), len(show) + 1,
                         figsize=(1.35 * (len(show) + 1), 1.4 * len(examples)))
for r, (dom, rgb, maps, tissue) in enumerate(examples):
    axes[r, 0].imshow(rgb)
    axes[r, 0].set_ylabel(dom, rotation=0, ha="right", va="center", fontsize=7)
    for c, ch in enumerate(show, start=1):
        axes[r, c].imshow(np.where(tissue, maps[ch], np.nan),
                          cmap="twilight" if ch == "H" else "magma",
                          vmin=limits[ch][0], vmax=limits[ch][1])
        if r == 0:
            axes[r, c].set_title(ch, fontsize=8)
    if r == 0:
        axes[r, 0].set_title("RGB", fontsize=8)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.tight_layout()
plots.save(fig, "Q_channel_maps")
""")

code("""
# Colour transforms toward NIH. Each is applied to masked test cells, the channel
# means are measured again, and the separation from NIH is recomputed: a transform
# that works should pull the colour channels toward 0.
rng = np.random.default_rng(0)
nih_ref = cstats[cstats["domain"] == "nih"]
ref_imgs = [plots.load_rgb(p, paths.MODEL_INPUT) for p in nih_ref["path"].head(200)]
lab_mu, lab_sd = colour.lab_moments(ref_imgs)
ref_q = colour.reference_quantiles(ref_imgs)
nih_h, nih_s = nih_ref["H"].median(), nih_ref["S"].median()

def transforms_for(dom_rows):
    dh = ((nih_h - dom_rows["H"].median() + 180) % 360) - 180
    ds = nih_s / max(dom_rows["S"].median(), 1e-6)
    return {"original": lambda im: im,
            "HSV shift to NIH": lambda im: colour.hsv_adjust(im, dh, ds),
            "Reinhard to NIH": lambda im: colour.reinhard(im, lab_mu, lab_sd),
            "histogram match to NIH": lambda im: colour.match_to_reference(im, ref_q)}

key_channels = ["H_from_nih", "S", "a", "b", "giemsa_blue", "giemsa_eosin"]
after, gallery = [], {}
for dom in [d for d in plots.domains_in(cstats) if "_masked/" in d]:
    rows = cstats[cstats["domain"] == dom]
    tf = transforms_for(rows)
    for p in rows["path"].sample(min(40, len(rows)), random_state=0):
        rgb = plots.load_rgb(p, paths.MODEL_INPUT)
        for name, fn in tf.items():
            out = fn(rgb)
            means = colour.tissue_channel_means(out)
            means["H_from_nih"] = ((means["H"] - nih_hue_mean + 180) % 360) - 180
            after.append({"domain": dom, "transform": name, **means})
            gallery.setdefault(dom, {}).setdefault(name, out)
after = pd.DataFrame(after)

nih_vals = nih_ref[key_channels]
effect = (after.groupby(["transform", "domain"])
          .apply(lambda g: pd.Series({ch: separation(g[ch].to_numpy(), nih_vals[ch].to_numpy())
                                      for ch in key_channels}))
          .round(2))
effect.reset_index().to_csv(paths.TABLES / "colour_transform_effect.csv", index=False)
print("separation from NIH after each transform (lower is closer to NIH):")
print(effect.to_string())
print()
print(effect.groupby("transform").median().round(2).to_string())

doms = list(gallery)
names = list(next(iter(gallery.values())))
fig, axes = plt.subplots(len(doms), len(names), figsize=(2.3 * len(names), 1.9 * len(doms)))
axes = np.asarray(axes).reshape(len(doms), len(names))
for r, dom in enumerate(doms):
    for c, name in enumerate(names):
        axes[r, c].imshow(gallery[dom][name])
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([]); axes[r, c].grid(False)
        if r == 0:
            axes[r, c].set_title(name, fontsize=8)
    axes[r, 0].set_ylabel(dom, rotation=0, ha="right", va="center", fontsize=7)
fig.tight_layout()
plots.save(fig, "Q_colour_transforms")
""")


md("""
**Findings.** Measured on tissue pixels of the NIH-format crops, the channel that
best tells the datasets apart is the Giemsa methyl-blue deconvolution. BBBC041 cells
absorbed three to four times as much methyl blue as NIH cells (median 0.065 to 0.090
against 0.022; separation 0.97 and 1.00), and MP-IDB cells up to about twice as much
(0.033 to 0.054; separation 0.48 to 0.70). The recut NIH cells from section O stay
close to the NIH crops on every channel (separation 0.17 or less), so these are
shifts between datasets, not artefacts of how the cells were cut.

The shifts do not all point the same way. site_a is darker and bluer (Lab L 37
against 66, b -24 against +4); site_b is darker and nearly grey (saturation 0.09
against 0.26) with little blue-yellow shift; Falciparum leans yellow-brown (b +13),
while Vivax and Ovale lean blue (b -20 and -15) at close to NIH brightness. No single
hue, saturation or brightness adjustment can reach every test set.

The transforms bear this out (median over the six test sets, 40 cells each).
Shifting hue and saturation toward NIH matches saturation but moves methyl blue
further from NIH than it was untransformed (separation 0.88 against 0.72). Matching
the Lab mean and spread (Reinhard) or the per-channel histograms brings the
blue-yellow axis and both stain channels close to NIH (b 0.12, methyl blue 0.17 to
0.22, eosin about 0.26). For RQ2 this points to a Lab- or stain-based normalisation
rather than an HSV adjustment, with a check that it does not wash out the parasite.
""")


nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": sys.version.split()[0]},
})
# Every code cell has to parse before the notebook is written. The cells are
# built from triple-quoted strings, so a backslash escape meant for the generated
# code is expanded here instead and silently emits a broken cell. Without this
# check the failure only surfaces minutes later, when papermill executes it.
# emits a broken cell. Without this check the failure only surfaces minutes later
# when papermill executes it.
broken = []
for i, c in enumerate(cells):
    if c.cell_type != "code":
        continue
    try:
        ast.parse(c.source)
    except SyntaxError as exc:
        broken.append(f"  cell {i}: {exc.msg} (line {exc.lineno})")
if broken:
    msg = "generated notebook has cells that do not parse:"
    raise SystemExit(chr(10).join([msg] + broken))

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT} ({len(cells)} cells, all parse)")
