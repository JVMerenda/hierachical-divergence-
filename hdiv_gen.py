#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds the input files of ablation_r32.py from a folder of networks.
 
Input
  --data-dir   folder with one adjacency-list text file per network
               (networkx format: "u v1 v2 ...", '#' starts a comment; a line with a
               single token declares an isolated node).
               The class of each network is the part of the file name before the
               first '_'   (e.g. "3_graph_0017.txt" -> class "3").
 
Output (--out-dir, default <data-dir>/features)
  y.npy                  class label of each network (strings)
  graph_names.txt        file name of each row, in the order used everywhere
  hd_full.npy            (n_graphs x 7*13) Hierarchical Divergence descriptor for
                         levels 1..13:  [v_1, v_2, ..., v_13],
                         v_n = [mean, std, entropy, q1, q2, q3, q4]   (as extract_vH)
  columns.txt            name of each column of hd_full.npy  (L<n>_<stat>)
  X_hd.npy               first 7*--main-levels columns of hd_full (default 8 levels,
                         the configuration of the manuscript) -> ready for ablation_r32.py
  cumulative/H01..H13/   (with --emit-cumulative) one dataset folder per H_max in 1..13,
                         each with y.npy and X_hd.npy = Phi^(H_max) = first 7*H_max columns.
                         Run:  ablation_r32.py --data-dir <out-dir>/cumulative --all-subdirs
  X_B1.npy ... X_B4.npy  (with --baselines) structural comparison sets of the manuscript.
                         APPROXIMATION: the exact definitions of modularity/diameter on
                         disconnected graphs are not in the text; if you have the code
                         used in the paper, save its outputs with the same row order
                         (graph_names.txt) instead.
 
Usage
  python build_inputs.py --data-dir /home/jvmerenda/TUDA/ENZYMES/
  python build_inputs.py --data-dir /home/jvmerenda/TUDA/ENZYMES/ --emit-cumulative --baselines
"""
import argparse
import re
import shutil
from pathlib import Path
 
import networkx as nx
import numpy as np
from scipy.stats import entropy
 
STATS = ["mean", "std", "entropy", "q1", "q2", "q3", "q4"]
HMAX = 13
 
 
# ---------------------------------------------------------------------------
# Same statistics as extract_vH of the experiments
# ---------------------------------------------------------------------------
def extract_vH(vals):
    """[mean, std, entropy(10-bin histogram), Q1, Q2 (median), Q3, Q4 (max)]."""
    vals = list(vals)
    if len(vals) == 0:
        return [0.0] * 7
    mean_val = np.mean(vals)
    std_val = np.std(vals)
    counts, _ = np.histogram(vals, bins=10, density=False)
    probs = counts / np.sum(counts) if np.sum(counts) > 0 else counts
    ent = entropy(probs)
    q1, q2, q3, q4 = np.percentile(vals, [25, 50, 75, 100])
    return [mean_val, std_val, ent, q1, q2, q3, q4]
 
 
# ---------------------------------------------------------------------------
# Hierarchical Divergence
# ---------------------------------------------------------------------------
def divergence_levels(G, hmax):
    """div(i, H) for all nodes and H=1..hmax with ONE truncated BFS per node.
    Empty shell (isolated node, or H beyond the eccentricity) -> 0 (Algorithm 1)."""
    deg = dict(G.degree())
    nodes = list(G.nodes())
    out = np.zeros((len(nodes), hmax))
    for r, i in enumerate(nodes):
        sums = np.zeros(hmax + 1)
        cnt = np.zeros(hmax + 1)
        for j, d in nx.single_source_shortest_path_length(G, i, cutoff=hmax).items():
            if d >= 1:
                sums[d] += deg[j] - deg[i]
                cnt[d] += 1
        nz = cnt[1:] > 0
        out[r, nz] = sums[1:][nz] / cnt[1:][nz]
    return out
 
 
def div_reference(G, H):
    """Literal transcription of Algorithm 1 (one level), used as a self-test."""
    D = {}
    for i in G.nodes():
        deg_i = G.degree(i)
        at_h, queue, visited = [], [(i, 0)], {i}
        while queue:
            cur, dist = queue.pop(0)
            if dist == H:
                at_h.append(cur)
                continue
            for nb in G.neighbors(cur):
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        if not at_h:
            D[i] = 0.0
            continue
        D[i] = sum(G.degree(j) - deg_i for j in at_h) / len(at_h)
    return D
 
 
def hd_descriptor(G, hmax):
    div = divergence_levels(G, hmax)
    feats = []
    for h in range(hmax):
        feats.extend(extract_vH(div[:, h]))
    return np.array(feats, dtype=float), div
 
 
# ---------------------------------------------------------------------------
# Optional baselines B1-B4 (approximation, see the docstring)
# ---------------------------------------------------------------------------
def _components(G):
    return [G.subgraph(c) for c in nx.connected_components(G)]
 
 
def baseline_B1(G):
    deg = np.array([d for _, d in G.degree()], float)
    clu = np.array(list(nx.clustering(G).values()), float)
    lens = []
    for comp in _components(G):
        for _, dist in nx.all_pairs_shortest_path_length(comp):
            lens.extend(v for v in dist.values() if v > 0)
    lens = np.array(lens, float) if lens else np.zeros(1)
    return [deg.mean(), deg.std(), clu.mean(), clu.std(), lens.mean(), lens.std()]
 
 
def baseline_B2(G):
    try:
        comms = nx.community.louvain_communities(G, seed=0)
        mod = nx.community.modularity(G, comms)
    except Exception:
        mod = 0.0
    diam, ecc = 0, []
    for comp in _components(G):
        e = nx.eccentricity(comp)
        ecc.extend(e.values())
        diam = max(diam, max(e.values()) if e else 0)
    return [mod, nx.transitivity(G), float(diam), float(np.mean(ecc)) if ecc else 0.0]
 
 
def baseline_B3(G):
    return [nx.number_connected_components(G), G.number_of_nodes(),
            G.number_of_edges(), nx.density(G)]
 
 
# ---------------------------------------------------------------------------
def natural_key(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]
 
 
def read_graph(path, comments):
    try:
        return nx.read_adjlist(path, comments=comments, nodetype=int, create_using=nx.Graph)
    except (TypeError, ValueError):
        return nx.read_adjlist(path, comments=comments, create_using=nx.Graph)
 
 
def column_names(nlevels):
    return [f"L{n}_{s}" for n in range(1, nlevels + 1) for s in STATS]
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="/home/jvmerenda/TUDA/ENZYMES/")
    ap.add_argument("--out-dir", default="/home/jvmerenda/hierarchical_divergence/feature-vectors/synthetic-model/")
    ap.add_argument("--pattern", default="*.txt")
    ap.add_argument("--comments", default="#")
    ap.add_argument("--hmax", type=int, default=HMAX, help="largest level computed (default 13)")
    ap.add_argument("--main-levels", type=int, default=8,
                    help="levels kept in X_hd.npy (default 8, as in the manuscript)")
    ap.add_argument("--emit-cumulative", action="store_true",
                    help="write one dataset folder per H_max=1..hmax")
    ap.add_argument("--baselines", action="store_true", help="also write X_B1..X_B4 (approximation)")
    args = ap.parse_args()
 
    root = Path(args.data_dir)
    out = Path(args.out_dir) if args.out_dir else root / "features"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(root.glob(args.pattern), key=natural_key)
    if not files:
        raise SystemExit(f"No files '{args.pattern}' in {root}")
    print(f"{len(files)} networks found in {root}")
 
    X, y, names, B = [], [], [], {k: [] for k in ("B1", "B2", "B3")}
    for n, path in enumerate(files):
        G = read_graph(path, args.comments)
        label = path.name.split("_")[0] if "_" in path.name else path.stem
        feats, div = hd_descriptor(G, args.hmax)
        if n == 0:                       # self-test against the literal Algorithm 1
            for h in range(1, min(args.hmax, 6) + 1):
                ref = div_reference(G, h)
                got = {v: div[r, h - 1] for r, v in enumerate(G.nodes())}
                assert all(abs(ref[v] - got[v]) < 1e-12 for v in ref), f"self-test failed at H={h}"
            print("  self-test against Algorithm 1: OK")
        X.append(feats)
        y.append(label)
        names.append(path.name)
        if args.baselines:
            B["B1"].append(baseline_B1(G))
            B["B2"].append(baseline_B2(G))
            B["B3"].append(baseline_B3(G))
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(files)}")
 
    X = np.vstack(X)
    y = np.array(y)
    np.save(out / "y.npy", y)
    np.save(out / "hd_full.npy", X)
    (out / "graph_names.txt").write_text("\n".join(names) + "\n")
    (out / "columns.txt").write_text("\n".join(column_names(args.hmax)) + "\n")
    k = 7 * args.main_levels
    np.save(out / "X_hd.npy", X[:, :k])
    print(f"y.npy: {len(y)} samples, classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"hd_full.npy: {X.shape}   X_hd.npy: {X[:, :k].shape} ({args.main_levels} levels)")
 
    if args.baselines:
        b1, b2, b3 = (np.array(B[k_], float) for k_ in ("B1", "B2", "B3"))
        for name, M in (("B1", b1), ("B2", b2), ("B3", b3), ("B4", np.hstack([b1, b2, b3]))):
            np.save(out / f"X_{name}.npy", M)
        print("baselines written: X_B1..X_B4  (approximation of the manuscript sets)")
 
    if args.emit_cumulative:
        cum = out / "cumulative"
        for h in range(1, args.hmax + 1):
            d = cum / f"H{h:02d}"
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "y.npy", y)
            np.save(d / "X_hd.npy", X[:, :7 * h])
            if args.baselines:
                for name in ("B1", "B2", "B3", "B4"):
                    shutil.copy(out / f"X_{name}.npy", d / f"X_{name}.npy")
        print(f"cumulative sets H=1..{args.hmax} written to {cum}")
 
    print("\nNext step:")
    print(f"  python ablation_r32.py --data-dir {out}")
    if args.emit_cumulative:
        print(f"  python ablation_r32.py --data-dir {out / 'cumulative'} --all-subdirs "
              f"--control-sets '' --n-repeats 3     # H_max sensitivity")
 
 
if __name__ == "__main__":
    main()
 