"""Stage 1: cry vs non-cry vs silence, grouped stratified 5-fold CV.

Windows quieter than -50 dBFS are labelled silence. Clip decision = mean P(cry) over
non-silent windows (>= 0.5), and the paper's device rule (any 3 consecutive windows with
mean P(cry) >= 0.80) is reported as well.

Variant "dac_only": ESC-50 crying_baby clips are removed from training, so their recall in
the test folds measures cross-source generalisation (shortcut check).
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedGroupKFold

from baseline import clip_features, knn
from features import SILENCE_DBFS
from train_lib import Scaler, fit, predict_int8, to_int8

SEED = 42
NONCRY, CRY, SIL = 0, 1, 2


def clip_scores(p_cry, clip_id, loud, clips):
    mean_s, dev_s = [], []
    for c in clips:
        m = clip_id == c
        pc, lo = p_cry[m], loud[m]
        mean_s.append(pc[lo].mean() if lo.any() else pc.mean())  # very quiet clip: use all windows
        run = np.convolve(pc, np.ones(3) / 3, mode="valid") if len(pc) >= 3 else pc
        dev_s.append(float(run.max()))
    return np.array(mean_s), np.array(dev_s)


def binary_metrics(y, yhat):
    return dict(acc=accuracy_score(y, yhat), macro_f1=f1_score(y, yhat, average="macro"),
                cry_precision=precision_score(y, yhat, zero_division=0),
                cry_recall=recall_score(y, yhat, zero_division=0),
                noncry_fpr=float(((yhat == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)))


def main():
    d = np.load("data.npz")
    X, clip_id, db = d["X"], d["clip_id"], d["dbfs"]
    df = pd.read_csv("data_clips.csv", keep_default_na=False)
    loud = db >= SILENCE_DBFS
    yw = np.where(~loud, SIL, df["cry"].values[clip_id])
    # clips with no window above the silence threshold keep their clip label (quiet recording, not silence)
    clip_max = np.full(len(df), -np.inf)
    np.maximum.at(clip_max, clip_id, db)
    quiet_clip = clip_max[clip_id] < SILENCE_DBFS
    yw[quiet_clip] = df["cry"].values[clip_id[quiet_clip]]
    y_clip, g_clip = df["cry"].values, df["group"].values

    print("baseline features ...", flush=True)
    B = np.stack([clip_features(p) for p in df["path"]])

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, preds = [], {k: np.zeros(len(df)) for k in
                       ["cnn_mean", "cnn_dev", "int8_mean", "base", "dac_only_mean"]}
    win_true, win_pred = [], []
    kb = None
    for f, (tr_c, te_c) in enumerate(outer.split(np.zeros(len(df)), y_clip, g_clip)):
        tr_w, te_w = np.isin(clip_id, tr_c), np.isin(clip_id, te_c)
        sc = Scaler().fit(X[tr_w])
        Xtr, Xte = sc(X[tr_w]), sc(X[te_w])
        model = fit(Xtr, yw[tr_w], 3, df["group"].values[clip_id[tr_w]], seed=SEED + f)
        P = model.predict(Xte, batch_size=1024, verbose=0)
        win_true.append(yw[te_w]); win_pred.append(P.argmax(1))
        ms, ds = clip_scores(P[:, CRY], clip_id[te_w], loud[te_w], te_c)
        preds["cnn_mean"][te_c], preds["cnn_dev"][te_c] = ms, ds

        tfl = to_int8(model, Xtr)
        kb = len(tfl) / 1024
        P8 = predict_int8(tfl, Xte)
        preds["int8_mean"][te_c] = clip_scores(P8[:, CRY], clip_id[te_w], loud[te_w], te_c)[0]

        # cross-source check: drop ESC crying_baby from training
        keep = tr_w & ~((df["esc_category"].values[clip_id] == "crying_baby"))
        sc2 = Scaler().fit(X[keep])
        m2 = fit(sc2(X[keep]), yw[keep], 3, df["group"].values[clip_id[keep]], seed=SEED + f)
        P2 = m2.predict(sc2(X[te_w]), batch_size=1024, verbose=0)
        preds["dac_only_mean"][te_c] = clip_scores(P2[:, CRY], clip_id[te_w], loud[te_w], te_c)[0]

        base = knn().fit(B[tr_c], y_clip[tr_c])
        preds["base"][te_c] = base.predict(B[te_c])
        print(f"fold {f}: done", flush=True)

    thr = {"cnn_mean": 0.5, "cnn_dev": 0.8, "int8_mean": 0.5, "base": 0.5, "dac_only_mean": 0.5}
    esc_cb = (df["esc_category"] == "crying_baby").values
    dac = (df["source"] == "dac").values
    res = {}
    for k, s in preds.items():
        yhat = (s >= thr[k]).astype(int)
        r = binary_metrics(y_clip, yhat)
        r["recall_dac_cry"] = float(yhat[dac].mean())
        r["recall_esc_crying_baby"] = float(yhat[esc_cb].mean())
        res[k] = r
    yt, yp = np.concatenate(win_true), np.concatenate(win_pred)
    res["window_3class"] = dict(macro_f1=f1_score(yt, yp, average="macro"),
                                confusion=confusion_matrix(yt, yp).tolist(),
                                labels=["non-cry", "cry", "silence"])
    res["int8_kb"] = kb
    yhat = preds["cnn_mean"] >= 0.5
    fp = (pd.DataFrame({"cat": df["esc_category"], "fp": yhat & (y_clip == 0)})
          .query("cat != '' and cat != 'crying_baby'").groupby("cat")["fp"].mean()
          .sort_values(ascending=False))
    res["top_false_positive_categories"] = fp.head(10).round(3).to_dict()
    df.assign(**{f"score_{k}": v for k, v in preds.items()}).to_csv("stage1_predictions.csv", index=False)
    json.dump(res, open("stage1_results.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
