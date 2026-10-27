#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recomputes the accuracy of the cumulative Hierarchical Divergence sets
Phi^(h) = first 7h columns of hd_full.npy, for h = 1..hmax, with the protocol of the
ablation analysis:  StandardScaler fitted on the training folds, RBF SVM with default
parameters, stratified 10-fold cross-validation repeated 5 times (same folds as
ablation_r32.py).
 
It produces
  * the LaTeX rows of Table 3 (levels 1..8), mean (std), best value in bold;
  * cumulative_accuracy.csv  (all levels 1..hmax, e.g. for the H_max sensitivity);
  * a short sensitivity report: best level and gain over H_max = 8.
 
Input: every folder under --root that contains hd_full.npy and y.npy
       (as written by build_inputs.py). The dataset name is the folder name, or the
       name of its parent when the folder is called 'features'.
 
Usage
  python cumulative_table.py --root /home/jvmerenda/TUDA/
"""
import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

DISPLAY = {"actinobacteria": "Actinobacteria", "animals": "Animals",
           "firmicutes-bacillis": "Firmicutes", "firmicutes": "Firmicutes",
           "fungi": "Fungi", "kingdom": "Kingdom", "plant": "Plant", "protist": "Protist",
           "enzymes": "Enzymes", "proteins": "Proteins", "mutag": "Mutag",
           "collab": "Collab", "imdb-multi": "IMDB-multi",
           "reddit-binary": "Reddit-binary", "reddit-multi": "Reddit-multi"}
ORDER = ["Actinobacteria", "Animals", "Firmicutes", "Fungi", "Kingdom", "Plant", "Protist",
         "Enzymes", "Proteins", "Mutag", "Collab", "IMDB-multi", "Reddit-binary", "Reddit-multi"]

# Caminho padrão onde o CSV será salvo automaticamente
DEFAULT_OUT_FILE = Path("/home/jvmerenda/hierarchical_divergence/ablation_study/summary.csv")


def rnd(x):                       # round half up, as in the tables of the manuscript
    return int(math.floor(x + 0.5))


def dataset_name(d: Path):
    return d.parent.name if d.name == "features" else d.name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--hmax", type=int, default=13)
    ap.add_argument("--table-levels", type=int, default=8)
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out-file", default=str(DEFAULT_OUT_FILE),
                    help="Caminho completo do arquivo de saída")
    ap.add_argument("--check-summary", default=None,
                    help="summary.csv of ablation_r32.py: compares the level-8 result with it")
    args = ap.parse_args()

    rows = []
    for p in sorted(Path(args.root).rglob("hd_full.npy")):
        d = p.parent
        if not (d / "y.npy").exists():
            continue
        name = dataset_name(d)
        X = np.load(p)
        y = LabelEncoder().fit_transform(np.load(d / "y.npy", allow_pickle=False).ravel())
        hmax = min(args.hmax, X.shape[1] // 7)
        
        # Define 5 splits para o dataset protista/protist e mantém o padrão para os demais
        n_splits = 5 if name.lower() in ["protist", "protista"] else args.n_splits

        splits = list(RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=args.n_repeats,
                                              random_state=args.seed).split(np.zeros(len(y)), y))
        print(f"{name}: {len(y)} samples, levels 1..{hmax} (n_splits={n_splits})")
        for h in range(1, hmax + 1):
            acc = cross_validate(make_pipeline(StandardScaler(), SVC()), X[:, :7 * h], y, cv=splits,
                                 scoring="accuracy", n_jobs=args.n_jobs)["test_score"]
            rows.append(dict(dataset=name, folder=d.name, h=h, acc_mean=acc.mean(), acc_std=acc.std()))
    if not rows:
        raise SystemExit("no folder with hd_full.npy and y.npy found")

    df = pd.DataFrame(rows)
    
    # Salva automaticamente no caminho especificado
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)

    # ---- LaTeX rows of Table 3 -------------------------------------------------
    L = args.table_levels
    print("\n%% ---- Table 3 rows (levels 1..%d); paste in place of the old rows ----" % L)
    names = [DISPLAY.get(n.lower(), n) for n in df.dataset.unique()]
    for disp in sorted(set(names), key=lambda n: ORDER.index(n) if n in ORDER else 99):
        key = [n for n in df.dataset.unique() if DISPLAY.get(n.lower(), n) == disp][0]
        sub = df[(df.dataset == key) & (df.h <= L)].sort_values("h")
        means = [rnd(100 * m) for m in sub.acc_mean]
        best = max(means)
        cells = []
        for m_, s_ in zip(means, sub.acc_std):
            txt = f"{m_} ({rnd(100 * s_)})"
            cells.append(r"\textbf{" + txt + "}" if m_ == best else txt)
        print(f"{disp:<15} & " + " & ".join(cells) + r" \\")

    # ---- sensitivity to H_max ----------------------------------------------------
    print("\n%% ---- H_max sensitivity (levels 1..%d) ----" % args.hmax)
    for key, sub in df.groupby("dataset"):
        sub = sub.sort_values("h")
        a8 = sub[sub.h == min(8, sub.h.max())].acc_mean.iloc[0]
        j = sub.acc_mean.idxmax()
        print(f"{DISPLAY.get(key.lower(), key):<15} best h={int(sub.loc[j, 'h']):>2} "
              f"({100 * sub.loc[j, 'acc_mean']:.1f}%)   h=8: {100 * a8:.1f}%   "
              f"gain over h=8: {100 * (sub.loc[j, 'acc_mean'] - a8):+.1f} pts")

    # ---- consistency check with the ablation summary -----------------------------
    if args.check_summary:
        s = pd.read_csv(args.check_summary)
        s = s[(s.feature_set == "hd") & (s.classifier == "rbf_svm")].set_index("dataset")
        print("\n%% ---- consistency with summary.csv (level 8 vs hd / rbf_svm) ----")
        for _, r_ in df[df.h == 8].iterrows():
            ref = r_.dataset if r_.dataset in s.index else (r_.folder if r_.folder in s.index else None)
            if ref is None:
                print(f"{r_.dataset}: not in summary.csv")
                continue
            print(f"{r_.dataset:<22} cumulative {100 * r_.acc_mean:.2f}  summary {100 * s.loc[ref, 'acc_mean']:.2f}  "
                  f"diff {100 * (r_.acc_mean - s.loc[ref, 'acc_mean']):+.3f}")
    print(f"\nWritten: {out_file}")


if __name__ == "__main__":
    main()