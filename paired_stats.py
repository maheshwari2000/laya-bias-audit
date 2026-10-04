"""
Paired significance tests for the calibration comparisons (Table 4 of the paper).

Reads the cached raw outputs in results/ (produced by laya_audit_full.py) and, for every
(task, checkpoint, framing, method) pair, compares the method against the raw model output
ON THE SAME TEST ITEMS using:
  * an exact McNemar test on per-item correctness (two-sided),
  * a paired bootstrap 95% CI for the accuracy difference (method - raw),
  * a paired bootstrap 95% CI for the ECE difference (method - raw),
then applies Holm-Bonferroni correction across all McNemar tests in the family.
No model calls are needed.  Usage:  python paired_stats.py [results_dir]
"""
import json, sys
import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
SEED, N_CALIB, N_BOOT, EPS = 0, 200, 2000, 1e-15


def load(name):
    with open(f"{RES}/{name}.jsonl") as f:
        return [json.loads(l) for l in f]


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def ece(conf, correct, n_bins=15):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return tot


def ece_bin(p, y):
    return ece(np.maximum(p, 1 - p), (p > 0.5) == y)


def split(n):
    perm = np.random.default_rng(SEED).permutation(n)
    return perm[:N_CALIB], perm[N_CALIB:]


def methods(p, y, cal, test, prior_global, prior_item=None):
    v = {"raw": p[test], "contextual": sigmoid(logit(p[test]) - logit(prior_global))}
    if prior_item is not None:
        v["contextual_per_item"] = sigmoid(logit(p[test]) - logit(prior_item[test]))
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(logit(p[cal]).reshape(-1, 1), y[cal])
    v["platt"] = lr.predict_proba(logit(p[test]).reshape(-1, 1))[:, 1]
    return v


def compare(raw, new, y):
    c0, c1 = (raw > 0.5) == y, (new > 0.5) == y
    b, c = int((c0 & ~c1).sum()), int((~c0 & c1).sum())          # raw-only correct, method-only correct
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
    rng = np.random.default_rng(SEED)
    d_acc, d_ece = [], []
    for _ in range(N_BOOT):
        s = rng.integers(0, len(y), len(y))
        d_acc.append(c1[s].mean() - c0[s].mean())
        d_ece.append(ece_bin(new[s], y[s]) - ece_bin(raw[s], y[s]))
    return {"acc_raw": c0.mean(), "acc_new": c1.mean(), "d_acc": c1.mean() - c0.mean(),
            "d_acc_ci": np.percentile(d_acc, [2.5, 97.5]), "b": b, "c": c, "p": p,
            "d_ece": ece_bin(new, y) - ece_bin(raw, y), "d_ece_ci": np.percentile(d_ece, [2.5, 97.5])}


rows = []
for ck in ["english", "multilingual"]:
    # ---------------- SST-2 ----------------
    R, CFR = load(f"{ck}_sst2"), load(f"{ck}_sst2_cf")
    y = np.array([r["pos"] for r in R])
    F = {"noul 'positive?'": np.array([r["noul_pos"] for r in R]),
         "noul 'negative?'": 1 - np.array([r["noul_neg"] for r in R]),
         "choice yes=A": np.array([r["choice_yesA"] for r in R]),
         "choice yes=B": np.array([r["choice_yesB"] for r in R])}
    F["choice averaged"] = (F["choice yes=A"] + F["choice yes=B"]) / 2
    cfk = {"noul 'positive?'": "noul_pos", "noul 'negative?'": "noul_neg_inv",
           "choice yes=A": "choice_yesA", "choice yes=B": "choice_yesB"}
    CF = {k: float(np.mean([c[v] for c in CFR])) for k, v in cfk.items()}
    CF["choice averaged"] = (CF["choice yes=A"] + CF["choice yes=B"]) / 2
    cal, test = split(len(y))
    for name, p in F.items():
        v = methods(p, y, cal, test, CF[name])
        for m in ["contextual", "platt"]:
            rows.append(("SST-2", ck, name, m, compare(v["raw"], v[m], y[test])))
    # ---------------- BoolQ ----------------
    R = load(f"{ck}_boolq")
    y = np.array([r["gold"] for r in R])
    cal, test = split(len(y))
    for name, key, ck_cf in [("noul", "noul", "cf_noul"), ("choice yes=A", "choice_yesA", "cf_yesA"),
                             ("choice yes=B", "choice_yesB", "cf_yesB")]:
        p, cf = np.array([r[key] for r in R]), np.array([r[ck_cf] for r in R])
        v = methods(p, y, cal, test, float(cf[cal].mean()), prior_item=cf)
        for m in ["contextual", "contextual_per_item", "platt"]:
            rows.append(("BoolQ", ck, name, m, compare(v["raw"], v[m], y[test])))

# ---------------- Holm-Bonferroni across the whole family ----------------
pv = np.array([r[4]["p"] for r in rows])
order = np.argsort(pv)
m = len(pv)
adj = np.empty(m)
running = 0.0
for rank, i in enumerate(order):
    running = max(running, min(1.0, (m - rank) * pv[i]))
    adj[i] = running

print(f"Paired comparisons vs raw (n_test: SST-2 672, BoolQ 600); {m} tests, Holm-corrected\n")
print("| task | ckpt | framing | method | acc raw | acc method | diff [95% CI] | raw-only / method-only correct | McNemar p | Holm p | sig | dECE [95% CI] |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
out = []
for (task, ck, name, meth, r), pa in zip(rows, adj):
    sig = ("+" if r["d_acc"] > 0 else "-") if pa < 0.05 else ""
    print(f"| {task} | {ck} | {name} | {meth} | {r['acc_raw']:.3f} | {r['acc_new']:.3f} | "
          f"{r['d_acc']:+.3f} [{r['d_acc_ci'][0]:+.3f}, {r['d_acc_ci'][1]:+.3f}] | {r['b']} / {r['c']} | "
          f"{r['p']:.2e} | {pa:.2e} | {sig} | {r['d_ece']:+.3f} [{r['d_ece_ci'][0]:+.3f}, {r['d_ece_ci'][1]:+.3f}] |")
    out.append({"task": task, "checkpoint": ck, "framing": name, "method": meth, "holm_p": float(pa),
                **{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()}})
with open(f"{RES}/paired_stats.json", "w") as f:
    json.dump(out, f, indent=1, default=float)

# ---------------- Key schemes vs semantic keys (Table 2), paired McNemar + Holm ----------------
krows = []
for ck in ["english", "multilingual"]:
    for ds, label in [("ag", "AG News"), ("emo", "Emotion")]:
        R = load(f"{ck}_{ds}_keys")
        by = {}
        for r in R:
            by.setdefault(r["scheme"], {})[r["id"]] = r["chosen"] == r["gold"]
        ids = sorted(by["semantic"])
        s = np.array([by["semantic"][i] for i in ids])
        for sc in ["letters", "numbers", "random"]:
            t = np.array([by[sc][i] for i in ids])
            b, c = int((s & ~t).sum()), int((~s & t).sum())
            p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
            krows.append((label, ck, sc, t.mean() - s.mean(), b, c, p))
kp = np.array([r[-1] for r in krows])
kadj, running = np.empty(len(kp)), 0.0
for rank, i in enumerate(np.argsort(kp)):
    running = max(running, min(1.0, (len(kp) - rank) * kp[i]))
    kadj[i] = running
print(f"\nKey schemes vs semantic keys ({len(kp)} tests, Holm-corrected)\n")
print("| dataset | ckpt | scheme | acc diff | semantic-only / scheme-only correct | McNemar p | Holm p |")
print("|---|---|---|---|---|---|---|")
for (label, ck, sc, d, b, c, p), pa in zip(krows, kadj):
    print(f"| {label} | {ck} | {sc} | {d:+.3f} | {b} / {c} | {p:.3f} | {pa:.3f} |")
