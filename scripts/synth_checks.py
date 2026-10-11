"""Controls 7(a) and 7(c) of docs/synthetic_onset_design.md on condition B3, seed 11.

(c) detector sanity: the same features and folds, target = attack in progress at the anchor window.
(a) host identity: can the last-window features identify the host (60 classes, split by time)?
Writes reports/synthetic_checks_B3_s11.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import synth_onset_study as s  # noqa: E402

df, runs = s.lab.generate("B3", 11)
nh = df.host_id.nunique(); nw = len(df) // nh
df = df.sort_values(["host_id", "window_start"])
X = np.log1p(df[s.lab.F].to_numpy(np.float32)).reshape(nh, nw, -1)
att = df.attack_present.to_numpy(np.int8).reshape(nh, nw)

# (c) detector sanity: host-disjoint folds, every 10th window, current window features
rng = np.random.default_rng(42)
order = rng.permutation(nh); fold = np.empty(nh, int); fold[order] = np.arange(nh) % s.N_FOLDS
idx = np.arange(s.L, nw, 10)
aucs = []
for k in range(s.N_FOLDS):
    tr = [h for h in range(nh) if fold[h] != k]; te = [h for h in range(nh) if fold[h] == k]
    Xtr = np.concatenate([X[h, idx] for h in tr]); ytr = np.concatenate([att[h, idx] for h in tr])
    Xte = np.concatenate([X[h, idx] for h in te]); yte = np.concatenate([att[h, idx] for h in te])
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=300).fit(sc.transform(Xtr), ytr)
    aucs.append(s.auc(yte, m.decision_function(sc.transform(Xte))))

# (a) host identity: train on first 60% of time, test on last 40%
cut = int(nw * 0.6)
tr_idx = np.arange(0, cut, 20); te_idx = np.arange(cut, nw, 20)
Xtr = np.concatenate([X[h, tr_idx] for h in range(nh)]); ytr = np.repeat(np.arange(nh), len(tr_idx))
Xte = np.concatenate([X[h, te_idx] for h in range(nh)]); yte = np.repeat(np.arange(nh), len(te_idx))
sc = StandardScaler().fit(Xtr)
m = LogisticRegression(max_iter=300).fit(sc.transform(Xtr), ytr)
acc = float((m.predict(sc.transform(Xte)) == yte).mean())

out = dict(detector_sanity_fold_auc=aucs, detector_sanity_median=float(np.median(aucs)),
           host_identity_accuracy=acc, host_identity_chance=1 / nh)
(ROOT / "reports/synthetic_checks_B3_s11.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
