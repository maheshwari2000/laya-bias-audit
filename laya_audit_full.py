# =============================================================================
# Laya bias audit: full end-to-end run (both checkpoints, all experiments)
# =============================================================================
# How to run on Kaggle:
#   1. Settings: Accelerator = GPU T4 x2, Internet = On.
#   2. Cell 1:  !pip install -q laya datasets
#   3. Cell 2:  paste this whole file and run it.
#   Recommended: "Save Version" -> "Save & Run All" so it runs in the background
#   (~30-45 minutes, ~35,000 model calls).
#
#   Every experiment caches its raw results in /kaggle/working/laya_audit/.
#   If the session dies, just run again: finished experiments load from disk.
#   Delete that folder to force a completely fresh run.
#
#   When it finishes, copy EVERYTHING printed after "BEGIN RESULTS" and paste it back.
#   Figures are saved in /kaggle/working/laya_audit/figures/.
# =============================================================================

import os
os.environ["USE_TF"] = "0"          # Kaggle ships TensorFlow; this avoids a model-loading deadlock

import builtins, json, time, itertools
import importlib.metadata as im
import numpy as np
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import chisquare
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Disable Laya's 4-decimal output rounding (research use; integer rounding kept intact) ----
def _no_round(x, n=None):
    return builtins.round(x) if n is None else x

import laya.agent as _la
_la.round = _no_round
try:
    import laya.onnx_agent as _lo
    _lo.round = _no_round
except Exception:
    pass
from laya import Router

# ============================== configuration ==============================
SEED = 0
N_BOOT = 1000
EPS = 1e-15
CHECKPOINTS = ["english", "multilingual"]
OUT = "/kaggle/working/laya_audit"
FIG = f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True)

N_AG, N_EMO, N_BOOLQ = 200, 300, 800   # SST-2 uses its full validation set (872)
N_EMO_PERMS = 6
N_CALIB = 200
RANDOM_KEYS = ["qzv", "lmt", "wkr", "pfy", "dxo", "bnu"]
CF_INPUTS = ["N/A", "[MASK]", "..."]

# ============================== environment ==============================
def _ver(pkg):
    try:
        return im.version(pkg)
    except Exception:
        return "unknown"

ENV = {
    "laya": _ver("laya"), "transformers": _ver("transformers"), "torch": torch.__version__,
    "datasets": _ver("datasets"), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
}

router = Router(preload=True, device="cuda" if torch.cuda.is_available() else "cpu")

# ============================== model helpers ==============================
def ask_choice(text, criteria, instructions, ckpt):
    q = {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}}
    a = router.predict(text, q, model=ckpt)["answers"]["q"]
    return a["choice"], {k: float(v) for k, v in a["probabilities"].items()}

def ask_noul(text, instructions, ckpt):
    q = {"q": {"type": "noul", "instructions": instructions}}
    return float(router.predict(text, q, model=ckpt)["answers"]["q"]["noul"])

_, _probe = ask_choice("The update broke everything.", {"positive": "happy", "negative": "angry"},
                       "What is the sentiment?", "english")
PATCH_OK = max(len(repr(v).split(".")[-1]) for v in _probe.values()) > 4

def cached(name, fn):
    """Run fn() once and store its records as JSONL; reload on later runs."""
    path = f"{OUT}/{name}.jsonl"
    if os.path.exists(path):
        with open(path) as f:
            return [json.loads(line) for line in f]
    t0 = time.time()
    recs = fn()
    with open(path + ".tmp", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    os.replace(path + ".tmp", path)
    print(f"  [ran {name}: {len(recs)} records in {time.time() - t0:.0f}s]")
    return recs

# ============================== statistics ==============================
def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def ece(conf, correct, n_bins=15):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)

def boot(stat, y, p, n=N_BOOT):
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        if len(np.unique(y[s])) < 2:
            continue
        vals.append(stat(y[s], p[s]))
    return tuple(float(v) for v in np.percentile(vals, [2.5, 97.5]))

def acc_stat(y, p):
    return float(((p > 0.5) == y).mean())

def binary_metrics(p, y):
    p, y = np.asarray(p, float), np.asarray(y, bool)
    correct = (p > 0.5) == y
    return {"acc": float(correct.mean()), "acc_ci": boot(acc_stat, y, p),
            "ece": ece(np.maximum(p, 1 - p), correct),
            "auroc": float(roc_auc_score(y, p)), "auroc_ci": boot(roc_auc_score, y, p),
            "recall_pos": float((p[y] > 0.5).mean()), "recall_neg": float((p[~y] <= 0.5).mean())}

def boot_acc_ci(correct, n=N_BOOT):
    correct = np.asarray(correct, float)
    rng = np.random.default_rng(SEED)
    vals = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n)]
    return tuple(float(v) for v in np.percentile(vals, [2.5, 97.5]))

def fmt(x):
    if isinstance(x, tuple) and len(x) == 2:
        return f"[{x[0]:.3f}, {x[1]:.3f}]"
    if isinstance(x, (float, np.floating)):
        return f"{x:.3f}"
    return str(x)

def md(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("|" + "---|" * len(headers))
    for r in rows:
        print("| " + " | ".join(fmt(v) for v in r) + " |")
    print()

# ============================== data ==============================
AG_LABELS = ["world", "sports", "business", "scitech"]
AG_DESC = {"world": "world news, politics, international affairs",
           "sports": "sports, games, athletes, teams",
           "business": "business, companies, markets, economy",
           "scitech": "science and technology, computers, research"}
AG_INSTR = "What is the topic of this news article?"

EMO_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]    # dataset label order
EMO_DESC = {"sadness": "sad, unhappy, grieving, lonely",
            "joy": "happy, cheerful, pleased, excited",
            "love": "love, affection, caring, romantic feelings",
            "anger": "angry, annoyed, irritated, furious",
            "fear": "afraid, scared, anxious, nervous",
            "surprise": "surprised, amazed, astonished, shocked"}
EMO_INSTR = "Which emotion does the writer express?"

SST_Q = "Is this review positive?"
SST_YES_A = {"A": "yes, the review is positive", "B": "no, the review is negative"}
SST_YES_B = {"A": "no, the review is negative", "B": "yes, the review is positive"}
BQ_YES_A = {"A": "yes", "B": "no"}
BQ_YES_B = {"A": "no", "B": "yes"}

try:
    ag = load_dataset("fancyzhx/ag_news", split="test")
except Exception:
    ag = load_dataset("ag_news", split="test")
emo = load_dataset("dair-ai/emotion", split="test")
sst = load_dataset("stanfordnlp/sst2", split="validation")
bq = load_dataset("google/boolq", split="validation")

def sample(n_total, n):
    return [int(i) for i in np.random.default_rng(SEED).choice(n_total, n, replace=False)]

AG_ITEMS = [(i, ag[i]["text"], AG_LABELS[ag[i]["label"]]) for i in sample(len(ag), N_AG)]
EMO_ITEMS = [(i, emo[i]["text"], EMO_LABELS[emo[i]["label"]]) for i in sample(len(emo), N_EMO)]
SST_ITEMS = [(i, sst[i]["sentence"], bool(sst[i]["label"] == 1)) for i in range(len(sst))]
BQ_ITEMS = [(i, bq[i]["passage"], bq[i]["question"].strip().capitalize() + "?", bool(bq[i]["answer"]))
            for i in sample(len(bq), N_BOOLQ)]

def key_schemes(labels):
    n = len(labels)
    return {"semantic": {c: c for c in labels},
            "letters": {c: "ABCDEFGHIJ"[i] for i, c in enumerate(labels)},
            "numbers": {c: str(i + 1) for i, c in enumerate(labels)},
            "random": dict(zip(labels, RANDOM_KEYS)),
            "misleading": {c: labels[(i + 1) % n] for i, c in enumerate(labels)}}

# ============================== experiment runners ==============================
def run_position(items, labels, desc, instr, ckpt, all_perms, tag):
    recs = []
    for iid, text, gold in tqdm(items, desc=f"{ckpt} {tag} position"):
        if all_perms:
            orders = [list(o) for o in itertools.permutations(labels)]
        else:
            rng = np.random.default_rng(SEED * 100003 + iid)
            orders = [[str(x) for x in rng.permutation(labels)] for _ in range(N_EMO_PERMS)]
        for order in orders:
            ch, pr = ask_choice(text, {c: desc[c] for c in order}, instr, ckpt)
            recs.append({"id": iid, "gold": gold, "order": order, "choice": ch, "probs": pr})
    return recs

def run_keys(items, labels, desc, instr, ckpt, tag):
    recs = []
    for iid, text, gold in tqdm(items, desc=f"{ckpt} {tag} keys"):
        for scheme, km in key_schemes(labels).items():
            inv = {v: c for c, v in km.items()}
            ck, pr = ask_choice(text, {km[c]: desc[c] for c in labels}, instr, ckpt)
            recs.append({"id": iid, "scheme": scheme, "gold": gold, "chosen": inv[ck], "chosen_key": ck,
                         "probs": {inv[k]: v for k, v in pr.items()}})
    return recs

def run_sst2(ckpt):
    recs = []
    for iid, text, pos in tqdm(SST_ITEMS, desc=f"{ckpt} sst2"):
        recs.append({"id": iid, "pos": pos,
                     "noul_pos": ask_noul(text, "Is this review positive?", ckpt),
                     "noul_neg": ask_noul(text, "Is this review negative?", ckpt),
                     "choice_yesA": ask_choice(text, SST_YES_A, SST_Q, ckpt)[1]["A"],
                     "choice_yesB": ask_choice(text, SST_YES_B, SST_Q, ckpt)[1]["B"]})
    return recs

def run_sst2_cf(ckpt):
    # every value is expressed as P(positive)
    return [{"input": t,
             "noul_pos": ask_noul(t, "Is this review positive?", ckpt),
             "noul_neg_inv": 1 - ask_noul(t, "Is this review negative?", ckpt),
             "choice_yesA": ask_choice(t, SST_YES_A, SST_Q, ckpt)[1]["A"],
             "choice_yesB": ask_choice(t, SST_YES_B, SST_Q, ckpt)[1]["B"]} for t in CF_INPUTS]

def run_boolq(ckpt):
    recs = []
    for iid, passage, q, ans in tqdm(BQ_ITEMS, desc=f"{ckpt} boolq"):
        recs.append({"id": iid, "gold": ans,
                     "noul": ask_noul(passage, q, ckpt),
                     "choice_yesA": ask_choice(passage, BQ_YES_A, q, ckpt)[1]["A"],
                     "choice_yesB": ask_choice(passage, BQ_YES_B, q, ckpt)[1]["B"],
                     "cf_noul": ask_noul("N/A", q, ckpt),
                     "cf_yesA": ask_choice("N/A", BQ_YES_A, q, ckpt)[1]["A"],
                     "cf_yesB": ask_choice("N/A", BQ_YES_B, q, ckpt)[1]["B"]})
    return recs

# ============================== analyses ==============================
FIGDATA = {"position": {}, "sst2_noulpos": {}, "mitigation": []}

def analyze_position(recs, labels, title, ckpt, ds):
    k = len(labels)
    by = {}
    for r in recs:
        by.setdefault(r["id"], []).append(r)
    ids = list(by)
    chosen = np.array([[sum(r["order"].index(r["choice"]) == s for r in by[i]) for s in range(k)] for i in ids], float)
    goldc = np.array([[sum(r["order"].index(r["gold"]) == s for r in by[i]) for s in range(k)] for i in ids], float)
    rate, grate = chosen.sum(0) / chosen.sum(), goldc.sum(0) / goldc.sum()
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(N_BOOT):
        s = rng.integers(0, len(ids), len(ids))
        bs.append(chosen[s].sum(0) / chosen[s].sum())
    lo, hi = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
    chi = chisquare(chosen.sum(0), f_exp=grate * chosen.sum())

    consistency = float(np.mean([len({r["choice"] for r in by[i]}) == 1 for i in ids]))
    distinct = float(np.mean([len({r["choice"] for r in by[i]}) for i in ids]))
    pg_std = float(np.mean([np.std([r["probs"][r["gold"]] for r in by[i]]) for i in ids]))
    swing = float(np.mean([max(r["probs"][r["gold"]] for r in by[i]) - min(r["probs"][r["gold"]] for r in by[i]) > 0.2
                           for i in ids]))
    single_acc = float(np.mean([r["choice"] == r["gold"] for r in recs]))
    avg_correct = []
    for i in ids:
        mp = {c: np.mean([r["probs"][c] for r in by[i]]) for c in labels}
        avg_correct.append(max(mp, key=mp.get) == by[i][0]["gold"])
    avg_acc = float(np.mean(avg_correct))

    print(f"### Position: {title}\n")
    print(f"items={len(ids)}, orderings per item={len(by[ids[0]])}, calls={len(recs)}")
    print(f"permutation consistency={consistency:.3f}, mean distinct answers per item={distinct:.2f}")
    print(f"mean std of p(gold) across orderings={pg_std:.4f}, items with p(gold) swing>0.2={swing:.3f}")
    print(f"accuracy single ordering={single_acc:.3f}, accuracy averaged over orderings={avg_acc:.3f}")
    print(f"chi-square chosen-slot vs gold-slot distribution: stat={chi.statistic:.2f}, p={chi.pvalue:.2e} "
          f"(assumes independent calls; see cluster-bootstrap CIs)\n")
    rows = []
    for s in range(k):
        sub = [r for r in recs if r["order"].index(r["gold"]) == s]
        c = np.array([r["choice"] == r["gold"] for r in sub])
        conf = [max(r["probs"].values()) for r in sub]
        rows.append([s + 1, float(rate[s]), (float(lo[s]), float(hi[s])), float(grate[s]),
                     float(c.mean()), ece(conf, c), float(np.mean([r["probs"][r["gold"]] for r in sub]))])
    md(["slot", "chosen share", "95% CI (item bootstrap)", "gold share", "acc when gold here",
        "ECE when gold here", "mean p(gold)"], rows)
    FIGDATA["position"][(ckpt, ds)] = {"rate": rate, "lo": lo, "hi": hi, "gold_rate": grate}
    return {"consistency": consistency, "distinct": distinct, "pgold_std": pg_std, "swing_gt_0.2": swing,
            "acc_single": single_acc, "acc_averaged": avg_acc, "chi2": float(chi.statistic),
            "chi2_p": float(chi.pvalue), "slot_rate": rate.tolist(), "slot_ci_lo": lo.tolist(),
            "slot_ci_hi": hi.tolist(), "gold_rate": grate.tolist(), "slot_table": rows}

def analyze_keys(recs, labels, title):
    schemes = list(key_schemes(labels))
    sem = {r["id"]: r["chosen"] for r in recs if r["scheme"] == "semantic"}
    rows, out = [], {}
    for s in schemes:
        rs = [r for r in recs if r["scheme"] == s]
        c = np.array([r["chosen"] == r["gold"] for r in rs])
        conf = [max(r["probs"].values()) for r in rs]
        agree = float(np.mean([r["chosen"] == sem[r["id"]] for r in rs]))
        row = [s, float(c.mean()), boot_acc_ci(c), ece(conf, c),
               float(np.mean([r["probs"][r["gold"]] for r in rs])), agree]
        rows.append(row)
        out[s] = row[1:]
    mis = [r for r in recs if r["scheme"] == "misleading"]
    follow_desc = float(np.mean([r["chosen"] == r["gold"] for r in mis]))
    follow_key = float(np.mean([r["chosen_key"] == r["gold"] for r in mis]))
    print(f"### Keys: {title}\n")
    md(["key scheme", "acc", "95% CI", "ECE", "mean p(gold)", "agreement w/ semantic"], rows)
    print(f"misleading keys: followed description={follow_desc:.3f}, followed key name={follow_key:.3f}, "
          f"other={1 - follow_desc - follow_key:.3f}\n")
    return {"schemes": out, "follow_desc": follow_desc, "follow_key": follow_key}

def mitigation(p, y, cal, test, prior_global, prior_item=None):
    v = {"raw": p[test], "contextual": sigmoid(logit(p[test]) - logit(prior_global))}
    if prior_item is not None:
        v["contextual_per_item"] = sigmoid(logit(p[test]) - logit(prior_item[test]))
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(logit(p[cal]).reshape(-1, 1), y[cal])
    v["platt"] = lr.predict_proba(logit(p[test]).reshape(-1, 1))[:, 1]
    return v, float(lr.coef_[0, 0])

def split(n):
    perm = np.random.default_rng(SEED).permutation(n)
    return perm[:N_CALIB], perm[N_CALIB:]

def analyze_sst2(rows, cfrecs, ckpt):
    y = np.array([r["pos"] for r in rows])
    F = {"noul 'positive?'": np.array([r["noul_pos"] for r in rows]),
         "noul 'negative?' (1-p)": 1 - np.array([r["noul_neg"] for r in rows]),
         "choice yes=A": np.array([r["choice_yesA"] for r in rows]),
         "choice yes=B": np.array([r["choice_yesB"] for r in rows])}
    F["choice ensemble"] = (F["choice yes=A"] + F["choice yes=B"]) / 2
    cfkeys = {"noul 'positive?'": "noul_pos", "noul 'negative?' (1-p)": "noul_neg_inv",
              "choice yes=A": "choice_yesA", "choice yes=B": "choice_yesB"}
    CF = {name: float(np.mean([c[k] for c in cfrecs])) for name, k in cfkeys.items()}
    CF["choice ensemble"] = (CF["choice yes=A"] + CF["choice yes=B"]) / 2

    print(f"### SST-2 yes/no framings (full validation set, n={len(y)}, positive rate={y.mean():.3f})\n")
    rows_t, out = [], {"raw": {}, "mitigation": {}, "content_free": CF}
    for name, p in F.items():
        m = binary_metrics(p, y)
        out["raw"][name] = m
        rows_t.append([name, m["acc"], m["acc_ci"], m["ece"], m["auroc"], m["auroc_ci"],
                       m["recall_pos"], m["recall_neg"], f"{p.min():.2e} / {np.median(p):.2e} / {p.max():.2e}",
                       CF[name]])
    md(["framing (as P(positive))", "acc", "acc 95% CI", "ECE", "AUROC", "AUROC 95% CI", "recall pos",
        "recall neg", "min / median / max", "content-free P(pos)"], rows_t)
    pp, pn = F["noul 'positive?'"], F["noul 'negative?' (1-p)"]
    sym = float(np.abs(pp - pn).mean())
    contra = float(((pp > 0.5) != (pn > 0.5)).mean())
    stuck = float(((np.array([r["noul_pos"] for r in rows]) < 0.2) & (np.array([r["noul_neg"] for r in rows]) < 0.2)).mean())
    print(f"noul symmetry error={sym:.3f}, contradictions={contra:.3f}, stuck (both 'no' with P<0.2)={stuck:.3f}\n")
    out.update({"symmetry_error": sym, "contradictions": contra, "stuck": stuck})

    cal, test = split(len(y))
    print(f"### SST-2 mitigation (test n={len(test)}, Platt fit on n={len(cal)} labels; contextual uses {len(CF_INPUTS)} content-free inputs, no labels)\n")
    rows_m = []
    for name, p in F.items():
        v, slope = mitigation(p, y, cal, test, CF[name])
        out["mitigation"][name] = {"platt_slope": slope}
        for meth, pv in v.items():
            m = binary_metrics(pv, y[test])
            out["mitigation"][name][meth] = m
            rows_m.append([name, meth, m["acc"], m["acc_ci"], m["ece"], m["auroc"], m["auroc_ci"]])
        rows_m.append([name, "(Platt slope)", f"{slope:+.3f}", "", "", "", ""])
        if name in ("noul 'negative?' (1-p)", "choice yes=A"):
            FIGDATA["mitigation"].append((f"SST-2 {name}\n{ckpt}",
                                          {mm: out["mitigation"][name][mm]["acc"] for mm in v}))
    md(["framing", "method", "acc", "acc 95% CI", "ECE", "AUROC", "AUROC 95% CI"], rows_m)
    FIGDATA["sst2_noulpos"][ckpt] = (F["noul 'positive?'"], y)
    return out

def analyze_boolq(rows, ckpt):
    y = np.array([r["gold"] for r in rows])
    print(f"### BoolQ (n={len(y)}, majority-class baseline 'always yes'={y.mean():.3f})\n")
    probe_rows, out = [], {"majority": float(y.mean()), "probes": {}, "mitigation": {}}
    for k in ["cf_noul", "cf_yesA", "cf_yesB"]:
        pc = np.array([r[k] for r in rows])
        au, ci = float(roc_auc_score(y, pc)), boot(roc_auc_score, y, pc)
        out["probes"][k] = {"mean": float(pc.mean()), "auroc": au, "auroc_ci": ci}
        probe_rows.append([k, float(pc.mean()), au, ci])
    print("Content-free probes ('N/A' as passage, real question). AUROC > 0.5 = probe carries answer knowledge:\n")
    md(["probe", "mean P(yes)", "AUROC vs gold", "95% CI"], probe_rows)

    cal, test = split(len(y))
    rows_m = []
    for name, key, cfk in [("noul", "noul", "cf_noul"), ("choice yes=A", "choice_yesA", "cf_yesA"),
                           ("choice yes=B", "choice_yesB", "cf_yesB")]:
        p = np.array([r[key] for r in rows])
        cf = np.array([r[cfk] for r in rows])
        v, slope = mitigation(p, y, cal, test, float(cf[cal].mean()), prior_item=cf)
        out["mitigation"][name] = {"platt_slope": slope}
        for meth, pv in v.items():
            m = binary_metrics(pv, y[test])
            out["mitigation"][name][meth] = m
            rows_m.append([name, meth, m["acc"], m["acc_ci"], m["ece"], m["auroc"], m["auroc_ci"]])
        rows_m.append([name, "(Platt slope)", f"{slope:+.3f}", "", "", "", ""])
        if name in ("noul", "choice yes=A"):
            FIGDATA["mitigation"].append((f"BoolQ {name}\n{ckpt}",
                                          {mm: out["mitigation"][name][mm]["acc"] for mm in v}))
    print(f"### BoolQ mitigation (test n={len(test)}; 'contextual' = one averaged prior from the calibration "
          f"split, 'contextual_per_item' = each question's own probe; Platt fit on n={len(cal)})\n")
    md(["framing", "method", "acc", "acc 95% CI", "ECE", "AUROC", "AUROC 95% CI"], rows_m)
    print(f"test-split majority baseline={y[test].mean():.3f}\n")
    return out

# ============================== run everything ==============================
S = {"env": ENV, "rounding_patch_active": PATCH_OK}
results_text = {}
for ckpt in CHECKPOINTS:
    print(f"\n>>> running checkpoint: {ckpt}")
    S[ckpt] = {"raw": {
        "ag_pos": cached(f"{ckpt}_ag_position",
                         lambda: run_position(AG_ITEMS, AG_LABELS, AG_DESC, AG_INSTR, ckpt, True, "ag")),
        "emo_pos": cached(f"{ckpt}_emo_position",
                          lambda: run_position(EMO_ITEMS, EMO_LABELS, EMO_DESC, EMO_INSTR, ckpt, False, "emo")),
        "ag_keys": cached(f"{ckpt}_ag_keys",
                          lambda: run_keys(AG_ITEMS, AG_LABELS, AG_DESC, AG_INSTR, ckpt, "ag")),
        "emo_keys": cached(f"{ckpt}_emo_keys",
                           lambda: run_keys(EMO_ITEMS, EMO_LABELS, EMO_DESC, EMO_INSTR, ckpt, "emo")),
        "sst2": cached(f"{ckpt}_sst2", lambda: run_sst2(ckpt)),
        "sst2_cf": cached(f"{ckpt}_sst2_cf", lambda: run_sst2_cf(ckpt)),
        "boolq": cached(f"{ckpt}_boolq", lambda: run_boolq(ckpt)),
    }}

print("\n\n" + "=" * 30 + " BEGIN RESULTS " + "=" * 30 + "\n")
print(f"Environment: {ENV}")
print(f"Rounding patch active: {PATCH_OK}")
print(f"Samples: AG News {N_AG} (all 24 orderings), Emotion {N_EMO} ({N_EMO_PERMS} random orderings), "
      f"SST-2 {len(SST_ITEMS)}, BoolQ {N_BOOLQ}; seed={SEED}, bootstrap={N_BOOT}\n")

for ckpt in CHECKPOINTS:
    raw = S[ckpt].pop("raw")
    print(f"\n## CHECKPOINT: {ckpt}\n")
    S[ckpt]["ag_position"] = analyze_position(raw["ag_pos"], AG_LABELS, f"AG News ({ckpt})", ckpt, "AG News")
    S[ckpt]["emo_position"] = analyze_position(raw["emo_pos"], EMO_LABELS, f"Emotion ({ckpt})", ckpt, "Emotion")
    S[ckpt]["ag_keys"] = analyze_keys(raw["ag_keys"], AG_LABELS, f"AG News ({ckpt})")
    S[ckpt]["emo_keys"] = analyze_keys(raw["emo_keys"], EMO_LABELS, f"Emotion ({ckpt})")
    S[ckpt]["sst2"] = analyze_sst2(raw["sst2"], raw["sst2_cf"], ckpt)
    S[ckpt]["boolq"] = analyze_boolq(raw["boolq"], ckpt)

print("=" * 30 + " END RESULTS " + "=" * 30)

# ============================== figures ==============================
# Fig 1: chosen share per option slot, with item-level bootstrap CIs
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for a, (ckpt, ds) in zip(axes.flat, itertools.product(CHECKPOINTS, ["AG News", "Emotion"])):
    d = FIGDATA["position"][(ckpt, ds)]
    x = np.arange(len(d["rate"]))
    a.bar(x, d["rate"], yerr=[d["rate"] - d["lo"], d["hi"] - d["rate"]], capsize=3, color="#4C72B0",
          label="share of choices")
    a.plot(x, d["gold_rate"], "k_", markersize=22, mew=2, label="share of correct answers")
    a.set_xticks(x)
    a.set_xticklabels([str(i + 1) for i in x])
    a.set_title(f"{ds} ({ckpt})")
    a.set_xlabel("option slot")
    a.set_ylabel("share")
axes[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/fig1_position.png", dpi=200)

# Fig 2: hidden inversion, log-scale noul probabilities split by true class
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for a, ckpt in zip(axes, CHECKPOINTS):
    p, y = FIGDATA["sst2_noulpos"][ckpt]
    lp = np.log10(np.clip(p, 1e-12, 1))
    lo_, hi_ = lp.min(), lp.max() if lp.max() > lp.min() else lp.min() + 1
    bins = np.linspace(lo_, hi_, 40)
    a.hist(lp[y], bins=bins, alpha=0.6, label="positive reviews")
    a.hist(lp[~y], bins=bins, alpha=0.6, label="negative reviews")
    a.set_title(f"noul 'Is this review positive?' ({ckpt})")
    a.set_xlabel("log10 P(yes)")
axes[0].set_ylabel("count")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_noul_inversion.png", dpi=200)

# Fig 3: accuracy by mitigation method across settings
methods = ["raw", "contextual", "contextual_per_item", "platt"]
settings = FIGDATA["mitigation"]
fig, a = plt.subplots(figsize=(max(8, 1.6 * len(settings)), 4))
w = 0.2
for j, m in enumerate(methods):
    vals = [d.get(m, np.nan) for _, d in settings]
    a.bar(np.arange(len(settings)) + (j - 1.5) * w, vals, w, label=m)
a.set_xticks(np.arange(len(settings)))
a.set_xticklabels([s for s, _ in settings], fontsize=7)
a.set_ylabel("test accuracy")
a.set_ylim(0.4, 1.0)
a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/fig3_mitigation.png", dpi=200)

with open(f"{OUT}/results_summary.json", "w") as f:
    json.dump(S, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
print(f"\nSaved: {OUT}/results_summary.json and figures in {FIG}/")
