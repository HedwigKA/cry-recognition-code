"""Stage 2: cry reason on Donate-a-Cry, grouped stratified 5-fold CV repeated over seeds.

Label schemes x {class-weighted, unweighted}. Clip prediction = argmax of mean window
probabilities over non-silent windows. Baselines: always-"hungry" and Fahmi-style k-NN.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold

from baseline import clip_features, knn
from features import SILENCE_DBFS
from train_lib import Scaler, fit, predict_int8, to_int8

REASONS = ["hungry", "belly_pain", "burping", "discomfort", "tired"]
SCHEMES = {
    "five": {r: i for i, r in enumerate(REASONS)},
    "binary": {"hungry": 0, "belly_pain": 1, "burping": 1, "discomfort": 1, "tired": 1},
    "three": {"hungry": 0, "belly_pain": 1, "discomfort": 1, "burping": 2, "tired": 2},
}
SCHEME_NAMES = {"five": REASONS, "binary": ["hungry", "other"],
                "three": ["hungry", "pain/discomfort", "tired/burping"]}
SEEDS = [0, 1, 2]


def main():
    d = np.load("data.npz")
    df = pd.read_csv("data_clips.csv", keep_default_na=False)
    dac = df[df["source"] == "dac"].reset_index(drop=True)
    clip_max = np.full(d["clip_id"].max() + 1, -np.inf)
    np.maximum.at(clip_max, d["clip_id"], d["dbfs"])
    # drop silent windows, but keep all windows of very quiet clips (5 clips peak below -50 dBFS)
    keep = np.isin(d["clip_id"], dac["clip_id"]) & (
        (d["dbfs"] >= SILENCE_DBFS) | (clip_max[d["clip_id"]] < SILENCE_DBFS))
    X, cid = d["X"][keep], d["clip_id"][keep]
    pos = {c: i for i, c in enumerate(dac["clip_id"])}
    cidx = np.array([pos[c] for c in cid])          # window -> row in dac
    groups = dac["group"].values
    B = np.stack([clip_features(p) for p in dac["path"]])

    results, confusions = {}, {}
    for scheme, mapping in SCHEMES.items():
        yc = dac["reason"].map(mapping).values
        K = len(SCHEME_NAMES[scheme])
        for weighted in (True, False):
            key = f"{scheme}_{'weighted' if weighted else 'unweighted'}"
            f1s, bals, per_class, cm = [], [], [], np.zeros((K, K), int)
            f1_int8 = []
            for seed in SEEDS:
                pred = np.zeros(len(dac), int)
                pred8 = np.zeros(len(dac), int)
                outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
                for f, (tr, te) in enumerate(outer.split(np.zeros(len(dac)), yc, groups)):
                    trw, tew = np.isin(cidx, tr), np.isin(cidx, te)
                    sc = Scaler().fit(X[trw])
                    m = fit(sc(X[trw]), yc[cidx[trw]], K, groups[cidx[trw]],
                            class_weight=weighted, epochs=30, seed=seed * 10 + f)
                    P = m.predict(sc(X[tew]), batch_size=1024, verbose=0)
                    do8 = key == "five_weighted"
                    P8 = predict_int8(to_int8(m, sc(X[trw])), sc(X[tew])) if do8 else None
                    for c in te:
                        mw = cidx[tew] == c
                        pred[c] = P[mw].mean(0).argmax()
                        if do8:
                            pred8[c] = P8[mw].mean(0).argmax()
                f1s.append(f1_score(yc, pred, average="macro"))
                bals.append(balanced_accuracy_score(yc, pred))
                per_class.append(precision_recall_fscore_support(yc, pred, labels=range(K),
                                                                 zero_division=0)[:3])
                cm += confusion_matrix(yc, pred, labels=range(K))
                if key == "five_weighted":
                    f1_int8.append(f1_score(yc, pred8, average="macro"))
                print(f"{key} seed {seed}: macro-F1 {f1s[-1]:.3f}", flush=True)
            pc = np.mean(per_class, axis=0)
            results[key] = dict(
                macro_f1_mean=float(np.mean(f1s)), macro_f1_sd=float(np.std(f1s)),
                balanced_acc_mean=float(np.mean(bals)),
                per_class={n: dict(precision=float(pc[0][i]), recall=float(pc[1][i]),
                                   f1=float(pc[2][i]), clips=int((yc == i).sum()))
                           for i, n in enumerate(SCHEME_NAMES[scheme])})
            if f1_int8:
                results[key]["int8_macro_f1_mean"] = float(np.mean(f1_int8))
            confusions[key] = cm.tolist()

        # baselines on the same folds
        maj = np.zeros(len(dac), int)
        base_f1 = []
        for seed in SEEDS:
            pb = np.zeros(len(dac), int)
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
                    np.zeros(len(dac)), yc, groups):
                pb[te] = knn().fit(B[tr], yc[tr]).predict(B[te])
            base_f1.append(f1_score(yc, pb, average="macro"))
        results[f"{scheme}_baseline_knn"] = dict(macro_f1_mean=float(np.mean(base_f1)),
                                                 macro_f1_sd=float(np.std(base_f1)))
        results[f"{scheme}_always_hungry"] = dict(macro_f1_mean=float(f1_score(yc, maj, average="macro")))

    json.dump(dict(results=results, confusions=confusions, labels=SCHEME_NAMES),
              open("stage2_results.json", "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
