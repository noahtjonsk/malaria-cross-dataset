"""How NIH-NLM-ThinBloodSmearsPf relates to the cell_images training set.

The supervisor asked whether the full-photograph NIH release is the same dataset
as cell_images. This answers it per photograph: which photographs and patients
appear in both, and whether the expert annotation counts on a photograph match
the number of cells cell_images kept from it.

    python scripts/check_nih_overlap.py

Writes outputs/tables/nih_source_overlap.csv, one row per photograph.

Matching is on (patient, photograph). The patient is the C### prefix, the same
key malaria/manifests.py uses for the split. Two naming quirks are handled:
cell_images has 20 photographs whose timestamp carries a trailing "a"
(C33P1, C37BP2), and the online release has patient C47 in two folders
(146C47P8thin_Original_Motic and 148C47P8thinOriginalOlympusCX21).
"""
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from malaria import paths  # noqa: E402

LOCAL_NAME = re.compile(r"^(?P<prefix>.+?)_?(?P<photo>IMG_\d{8}_\d{6}a?)"
                        r"_cell_\d+\.png$")
FOLDER_PATIENT = re.compile(r"^\d+(C\d+)")


def local_photos() -> pd.DataFrame:
    counts = collections.defaultdict(collections.Counter)
    for folder, label in ((paths.NIH_PARASITIZED, "Parasitized"),
                          (paths.NIH_UNINFECTED, "Uninfected")):
        for f in folder.iterdir():
            if f.suffix.lower() != ".png":
                continue
            m = LOCAL_NAME.match(f.name)
            if m is None:
                raise ValueError(f"unexpected cell_images filename: {f.name}")
            patient = re.match(r"(C\d+)", m.group("prefix")).group(1)
            counts[(patient, m.group("photo"))][label] += 1
    return pd.DataFrame([{"patient_id": p, "photo": ph,
                          "local_parasitized": c["Parasitized"],
                          "local_uninfected": c["Uninfected"]}
                         for (p, ph), c in counts.items()])


def online_photos() -> pd.DataFrame:
    rows = []
    for set_name, set_dir in paths.NIHPOLY_SETS.items():
        for folder in sorted(set_dir.iterdir()):
            if not folder.is_dir():
                continue
            m = FOLDER_PATIENT.match(folder.name)
            if m is None:
                raise ValueError(f"unexpected patient folder: {folder.name}")
            for gt in sorted((folder / "GT").glob("*.txt")):
                lines = [l for l in gt.read_text(errors="ignore").splitlines()
                         if l.strip()]
                labels = collections.Counter(l.split(",")[1].strip()
                                             for l in lines[1:])
                rows.append({"patient_id": m.group(1), "photo": gt.stem,
                             "set": set_name, "folder": folder.name,
                             "gt_parasitized": labels["Parasitized"],
                             "gt_uninfected": labels["Uninfected"],
                             "gt_wbc": labels["White_Blood_Cell"]})
    return pd.DataFrame(rows)


def main() -> None:
    local, online = local_photos(), online_photos()
    table = local.merge(online, on=["patient_id", "photo"], how="outer",
                        indicator=True)
    table["where"] = table["_merge"].map({"both": "both",
                                          "left_only": "cell_images_only",
                                          "right_only": "thinblood_only"})
    table = table.drop(columns="_merge").sort_values(["patient_id", "photo"])

    out = paths.TABLES / "nih_source_overlap.csv"
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    pat_local, pat_online = set(local.patient_id), set(online.patient_id)
    print(f"cell_images:  {len(local):,} photographs, {len(pat_local)} patients")
    print(f"ThinBloodPf:  {len(online):,} photographs, {len(pat_online)} patients "
          f"({online.folder.nunique()} folders)")
    print(f"patients in both {len(pat_local & pat_online)}, "
          f"cell_images only {sorted(pat_local - pat_online)}, "
          f"ThinBloodPf only {sorted(pat_online - pat_local)}")
    print("photographs:", table["where"].value_counts().to_dict())

    both = table[table["where"] == "both"]
    for set_name, g in both.groupby("set"):
        diff = (g.gt_parasitized - g.local_parasitized).abs()
        print(f"{set_name:8s} shared photographs {len(g):4d}: parasitised count "
              f"equal on {int((diff == 0).sum())}, within 1 on "
              f"{int((diff <= 1).sum())}; annotated {int(g.gt_parasitized.sum()):,} "
              f"vs kept {int(g.local_parasitized.sum()):,} parasitised, "
              f"{int(g.gt_uninfected.sum()):,} vs {int(g.local_uninfected.sum()):,} "
              f"uninfected")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
