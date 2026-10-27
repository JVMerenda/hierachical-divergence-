#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation for Reviewer 3, comment 3.2:
is Hierarchical Divergence discriminative by itself, or only because of the RBF-SVM?

Blocks
  A  linear classifiers (logistic regression, linear SVM, LDA) + RBF-SVM (reference)
     + majority-class baseline.
  B  controls: label permutation, independent column shuffling, Gaussian noise.
  C  separability without a classifier: silhouette (+ permutation null) and ANOVA F.
  D  the baseline feature sets (B1-B4, A1-A6, ...) evaluated with the SAME
     classifiers, folds and controls; paired tests HD vs. each baseline.

Expected content of the dataset directory (one dataset per directory)
  y.npy | y.csv                labels, one per sample (any dtype)
  X_hd.npy | X_hd.csv          Hierarchical Divergence features (n_samples x 7h);
                               columns ordered [v_1, v_2, ..., v_h], each v_n =
                               [mean, std, entropy, q1, q2, q3, q4]
  X_<name>.npy | X_<name>.csv  any other feature set, e.g. X_B1, X_B2, X_B3, X_B4
  (a lone X.npy is treated as the 'hd' set)
CSV files: numeric, with or without a header row.

Protocol (same as the manuscript): StandardScaler fitted on the training folds only,
stratified 10-fold cross-validation, default SVM parameters (C=1, gamma='scale').
All feature sets of a dataset use exactly the same folds.

Usage
  python ablation_r32.py --data-dir /home/jvmerenda/directory/
  python ablation_r32.py --data-dir /home/jvmerenda/all_datasets/ --all-subdirs
  python ablation_r32.py --data-dir /home/jvmerenda/directory/ --quick     # smoke test
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_validate, permutation_test_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Decision criteria: DECLARE THEM BEFORE LOOKING AT THE RESULTS
# ---------------------------------------------------------------------------
ALPHA = 0.05            # significance level for permutation tests
CHANCE_MARGIN = 0.05    # null controls must stay within majority accuracy + margin
STATS = ["mean", "std", "entropy", "q1", "q2", "q3", "q4"]   # order of extract_vH


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _read_table(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    df = pd.read_csv(path, header=None)
    if df.iloc[0].map(lambda v: isinstance(v, str)).any():
        df = pd.read_csv(path)
    return df.to_numpy()


def load_dataset(d: Path, hd_name: str):
    files = [p for p in sorted(d.iterdir()) if p.suffix in (".npy", ".csv")]
    ycands = [p for p in files if p.stem in ("y", "labels", "target")]
    if not ycands:
        raise FileNotFoundError(f"{d}: no y.npy / y.csv found")
    y = _read_table(ycands[0]).ravel()
    y = LabelEncoder().fit_transform(y)

    sets = {}
    for p in files:
        if p.stem == "X":
            sets[hd_name] = _read_table(p).astype(float)
        elif p.stem.startswith("X_"):
            sets[p.stem[2:]] = _read_table(p).astype(float)
    if hd_name not in sets:
        raise FileNotFoundError(f"{d}: feature set '{hd_name}' (X_{hd_name}.npy) not found")
    for k, X in sets.items():
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[0] != len(y):
            raise ValueError(f"{d}: X_{k} has {X.shape[0]} rows but y has {len(y)}")
        if not np.isfinite(X).all():
            print(f"  [warn] {k}: non-finite values replaced by 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        sets[k] = X
    return sets, y


def feature_names(name, n_features, hd_name):
    if name == hd_name and n_features % len(STATS) == 0:
        return [f"L{j // len(STATS) + 1}_{STATS[j % len(STATS)]}" for j in range(n_features)]
    return [f"f{j}" for j in range(n_features)]


# ---------------------------------------------------------------------------
# Classifiers (block A) and helpers
# ---------------------------------------------------------------------------
def make_classifiers():
    return {
        "rbf_svm": SVC(kernel="rbf"),                                  # as in the manuscript
        "linear_svm": LinearSVC(C=1.0, dual=False, max_iter=10000),
        "logreg": LogisticRegression(max_iter=5000),
        "lda": LinearDiscriminantAnalysis(),
    }


def pipe(clf):
    return make_pipeline(StandardScaler(), clone(clf))


def make_splits(y, n_splits, n_repeats, seed):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(rskf.split(np.zeros(len(y)), y))


def cv_scores(estimator, X, y, splits, n_jobs):
    r = cross_validate(estimator, X, y, cv=splits, n_jobs=n_jobs,
                       scoring={"acc": "accuracy", "bal": "balanced_accuracy"})
    return r["test_acc"], r["test_bal"]


# ---------------------------------------------------------------------------
# Paired tests (block D)
# ---------------------------------------------------------------------------
def corrected_resampled_ttest(d, test_frac):
    """Nadeau & Bengio (2003) corrected resampled t-test on paired fold differences."""
    n = len(d)
    var = np.var(d, ddof=1)
    if n < 2 or var == 0:
        return np.nan, np.nan
    t = np.mean(d) / np.sqrt((1.0 / n + test_frac / (1.0 - test_frac)) * var)
    return t, 2 * stats.t.sf(abs(t), n - 1)


def wilcoxon_p(d):
    try:
        return stats.wilcoxon(d, zero_method="zsplit").pvalue
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Controls (block B)
# ---------------------------------------------------------------------------
def control_label_permutation(X, y, clf, n_perm, n_splits, seed, n_jobs):
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    score, perm, p = permutation_test_score(
        pipe(clf), X, y, scoring="accuracy", cv=cv,
        n_permutations=n_perm, n_jobs=n_jobs, random_state=seed)
    return score, perm.mean(), perm.std(), p


def control_null(X, y, clf, kind, n_null, n_splits, seed, n_jobs):
    """Mean CV accuracy of n_null replicates of a destroyed feature matrix."""
    rng = np.random.default_rng(seed)
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    accs = []
    for _ in range(n_null):
        if kind == "shuffle_columns":      # each column permuted independently
            Xn = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        elif kind == "gaussian_noise":     # same dimensionality, N(0,1)
            Xn = rng.normal(size=X.shape)
        else:
            raise ValueError(kind)
        s = cross_validate(pipe(clf), Xn, y, cv=cv, scoring="accuracy",
                           n_jobs=n_jobs)["test_score"]
        accs.append(s.mean())
    real = cross_validate(pipe(clf), X, y, cv=cv, scoring="accuracy",
                          n_jobs=n_jobs)["test_score"].mean()
    accs = np.array(accs)
    p = (1 + np.sum(accs >= real)) / (len(accs) + 1)
    return real, accs.mean(), accs.std(), p


# ---------------------------------------------------------------------------
# Separability without a classifier (block C): silhouette and ANOVA F
# ---------------------------------------------------------------------------
def bh_fdr(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    return out


def separability(X, y, seed, n_perm, max_n):
    rng = np.random.default_rng(seed)
    n = len(y)
    if n > max_n:
        idx, _ = train_test_split(np.arange(n), train_size=max_n, stratify=y,
                                  random_state=seed)
    else:
        idx = np.arange(n)
    Xs = StandardScaler().fit_transform(X[idx])
    ys = y[idx]
    D = pairwise_distances(Xs)
    sil = silhouette_score(D, ys, metric="precomputed")
    null = np.array([silhouette_score(D, rng.permutation(ys), metric="precomputed")
                     for _ in range(n_perm)])
    p_sil = (1 + np.sum(null >= sil)) / (n_perm + 1)

    F, pv = f_classif(X, y)
    F = np.nan_to_num(F, nan=0.0)
    pv = np.where(np.isnan(pv), 1.0, pv)
    q = bh_fdr(pv)
    summary = dict(silhouette=sil, sil_null_mean=null.mean(), sil_null_std=null.std(),
                   sil_p=p_sil, n_samples_used=len(idx), n_features=X.shape[1],
                   anova_median_F=float(np.median(F)), anova_max_F=float(F.max()),
                   anova_n_sig_bonf=int(np.sum(pv < ALPHA / X.shape[1])),
                   anova_n_sig_bh=int(np.sum(q < ALPHA)))
    return summary, F, pv, q


# ---------------------------------------------------------------------------
# One dataset
# ---------------------------------------------------------------------------
def run_dataset(path: Path, args):
    ds = path.name
    print(f"\n=== {ds} ===")
    sets, y = load_dataset(path, args.hd_name)
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    print(f"  samples={n}  classes={len(classes)}  majority={counts.max() / n:.3f}  "
          f"sets={sorted(sets)}")
    n_perm, n_null = args.n_perm, args.n_null
    if n > 2000:
        n_perm, n_null = min(n_perm, 30), min(n_null, 10)
        print(f"  [note] large dataset: permutations reduced to {n_perm}, nulls to {n_null}")

    clfs = make_classifiers()
    splits = make_splits(y, args.n_splits, args.n_repeats, args.seed)

    # ---- Blocks A and D: all sets x all classifiers, identical folds ---------
    fold_rows, sum_rows, fold_scores = [], [], {}
    maj_acc, maj_bal = cv_scores(DummyClassifier(strategy="most_frequent"),
                                 sets[args.hd_name], y, splits, args.n_jobs)
    for i, (a, b) in enumerate(zip(maj_acc, maj_bal)):
        fold_rows.append(dict(dataset=ds, feature_set="-", classifier="majority", fold=i, acc=a, bal_acc=b))
    sum_rows.append(dict(dataset=ds, feature_set="-", classifier="majority",
                         acc_mean=maj_acc.mean(), acc_std=maj_acc.std(),
                         bal_mean=maj_bal.mean(), bal_std=maj_bal.std()))
    majority_acc = float(counts.max() / n)

    for fs, X in sets.items():
        for cn, clf in clfs.items():
            acc, bal = cv_scores(pipe(clf), X, y, splits, args.n_jobs)
            fold_scores[(fs, cn)] = acc
            for i, (a, b) in enumerate(zip(acc, bal)):
                fold_rows.append(dict(dataset=ds, feature_set=fs, classifier=cn, fold=i, acc=a, bal_acc=b))
            sum_rows.append(dict(dataset=ds, feature_set=fs, classifier=cn,
                                 acc_mean=acc.mean(), acc_std=acc.std(),
                                 bal_mean=bal.mean(), bal_std=bal.std()))
            print(f"  [A/D] {fs:>8s} {cn:>10s}  acc={acc.mean():.3f}+-{acc.std():.3f}")

    # ---- paired tests: HD vs every other set, per classifier ------------------
    test_rows = []
    test_frac = 1.0 / args.n_splits
    for cn in clfs:
        for fs in sets:
            if fs == args.hd_name:
                continue
            d = fold_scores[(args.hd_name, cn)] - fold_scores[(fs, cn)]
            t, p_t = corrected_resampled_ttest(d, test_frac)
            test_rows.append(dict(dataset=ds, classifier=cn, hd=args.hd_name, other=fs,
                                  mean_diff=d.mean(), wilcoxon_p=wilcoxon_p(d),
                                  corrected_t=t, corrected_t_p=p_t))

    # ---- Block B: controls (on the sets/classifiers requested) ----------------
    ctrl_rows = []
    for fs in args.control_sets:
        if fs not in sets:
            continue
        X = sets[fs]
        for cn in args.control_classifiers:
            clf = clfs[cn]
            real, pm, ps, p = control_label_permutation(X, y, clf, n_perm, args.n_splits,
                                                        args.seed, args.n_jobs)
            ctrl_rows.append(dict(dataset=ds, feature_set=fs, classifier=cn,
                                  control="label_permutation", real_acc=real,
                                  null_mean=pm, null_std=ps, p_value=p))
            for kind in ("shuffle_columns", "gaussian_noise"):
                real, nm, ns, p = control_null(X, y, clf, kind, n_null, args.n_splits,
                                               args.seed, args.n_jobs)
                ctrl_rows.append(dict(dataset=ds, feature_set=fs, classifier=cn,
                                      control=kind, real_acc=real, null_mean=nm,
                                      null_std=ns, p_value=p))
            print(f"  [B]   {fs:>8s} {cn:>10s}  controls done")

    # ---- Block C: silhouette and ANOVA F --------------------------------------
    sep_rows, anova_rows = [], []
    for fs, X in sets.items():
        summ, F, pv, q = separability(X, y, args.seed, args.n_perm_sil, args.max_sil_samples)
        sep_rows.append(dict(dataset=ds, feature_set=fs, **summ))
        names = feature_names(fs, X.shape[1], args.hd_name)
        for j in range(X.shape[1]):
            anova_rows.append(dict(dataset=ds, feature_set=fs, feature=names[j],
                                   F=F[j], p=pv[j], q_bh=q[j]))
        print(f"  [C]   {fs:>8s}  silhouette={summ['silhouette']:.3f} "
              f"(null {summ['sil_null_mean']:.3f}, p={summ['sil_p']:.3f})  "
              f"ANOVA sig(BH)={summ['anova_n_sig_bh']}/{summ['n_features']}")

    # ---- Decision criteria (declared at the top of this file) ------------------
    crit = dict(dataset=ds, majority_acc=majority_acc)
    ctrl = pd.DataFrame(ctrl_rows, columns=["dataset", "feature_set", "classifier", "control",
                                            "real_acc", "null_mean", "null_std", "p_value"])
    # (a) the linear model beats the label-permutation null
    a = ctrl[(ctrl.feature_set == args.hd_name) & (ctrl.classifier == "logreg")
             & (ctrl.control == "label_permutation")]
    crit["a_linear_above_chance"] = bool(a.p_value.iloc[0] < ALPHA) if len(a) else np.nan
    # (b) ordering HD vs best baseline has the same sign for all classifiers
    others = [fs for fs in sets if fs != args.hd_name]
    if others:
        signs = []
        for cn in clfs:
            best = max(fold_scores[(fs, cn)].mean() for fs in others)
            signs.append(np.sign(fold_scores[(args.hd_name, cn)].mean() - best))
        crit["b_order_preserved"] = bool(len(set(signs)) == 1)
        crit["b_delta_by_clf"] = ";".join(f"{cn}:{s:+.0f}" for cn, s in zip(clfs, signs))
    else:
        crit["b_order_preserved"] = np.nan
    # (c) shuffled-column and noise controls stay near the majority accuracy
    c = ctrl[(ctrl.feature_set == args.hd_name) & (ctrl.control.isin(["shuffle_columns", "gaussian_noise"]))]
    crit["c_nulls_near_chance"] = bool((c.null_mean <= majority_acc + CHANCE_MARGIN).all()) if len(c) else np.nan
    # (d) separability beats the silhouette permutation null
    s = [r for r in sep_rows if r["feature_set"] == args.hd_name][0]
    crit["d_silhouette_above_null"] = bool(s["sil_p"] < ALPHA)
    print("  [criteria]", {k: v for k, v in crit.items() if k != "dataset"})

    return dict(folds=pd.DataFrame(fold_rows), summary=pd.DataFrame(sum_rows),
                tests=pd.DataFrame(test_rows), controls=ctrl,
                separability=pd.DataFrame(sep_rows), anova=pd.DataFrame(anova_rows),
                criteria=pd.DataFrame([crit]))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="/home/jvmerenda/directory/")
    ap.add_argument("--all-subdirs", action="store_true",
                    help="treat each subdirectory of --data-dir as one dataset")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--hd-name", default="hd")
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=100, help="label permutations (block B)")
    ap.add_argument("--n-null", type=int, default=20, help="replicates of shuffled/noise controls")
    ap.add_argument("--n-perm-sil", type=int, default=200, help="label permutations for the silhouette null")
    ap.add_argument("--max-sil-samples", type=int, default=2000)
    ap.add_argument("--control-sets", default="hd", help="comma-separated, e.g. hd,B4")
    ap.add_argument("--control-classifiers", default="logreg,rbf_svm")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    args = ap.parse_args()
    if args.quick:
        args.n_repeats, args.n_perm, args.n_null, args.n_perm_sil = 1, 10, 3, 20
    args.control_sets = [s for s in args.control_sets.split(",") if s]
    args.control_classifiers = [s for s in args.control_classifiers.split(",") if s]

    root = Path(args.data_dir)
    paths = sorted(p for p in root.iterdir() if p.is_dir()) if args.all_subdirs else [root]
    out = Path(args.out_dir) if args.out_dir else root / "ablation_r32_results"
    out.mkdir(parents=True, exist_ok=True)

    parts = {}
    for p in paths:
        if p.name == out.name:
            continue
        try:
            res = run_dataset(p, args)
        except Exception as e:                       # keep going on a bad dataset
            print(f"  [skip] {p.name}: {e}")
            continue
        for k, v in res.items():
            parts.setdefault(k, []).append(v)

    for k, v in parts.items():
        pd.concat(v, ignore_index=True).to_csv(out / f"{k}.csv", index=False)
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()