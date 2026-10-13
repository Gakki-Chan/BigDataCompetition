# -*- coding: utf-8 -*-
"""Probe: can hidden edges (task B/C) be reconstructed by a generic learned pairwise model?
Also: chain recovery from the *predicted* graph."""
import json, os, re, sys, collections, io, datetime, random
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TRAIN = os.path.join(ROOT, "赛题4", "数据集", "训练集")


def load(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def edge_pair(c):
    a = c.get("cause_event_id", c.get("cause", c.get("cause_id")))
    b = (c.get("result_event_id") or c.get("result") or c.get("effect_event_id")
         or c.get("result_id"))
    return a, b


def parse_time(s):
    if not s:
        return None
    s = str(s)
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.datetime.strptime(s[:10], fmt)
        except Exception:
            continue
    m = re.match(r"(\d{4})", s)
    if m:
        return datetime.datetime(int(m.group(1)), 1, 1)
    return None


def build(pk):
    d = os.path.join(TRAIN, pk)
    evs = load(os.path.join(d, "事件列表.json"))
    ces = load(os.path.join(d, "事件因果关系列表.json"))
    gold = {g["sample_id"]: g for g in load(os.path.join(d, "gold", "问答对_答案.json"))}
    qs = load(os.path.join(d, "问题.json"))
    return evs, ces, qs, gold


packs = sorted(d for d in os.listdir(TRAIN) if os.path.isdir(os.path.join(TRAIN, d)))
random.seed(0)
random.shuffle(packs)
n_val = 200
val_packs, tr_packs = packs[:n_val], packs[n_val:]

# ---- type-pair priors from training packs (smoothed) ----
pair_cnt = collections.Counter()
cause_cnt = collections.Counter()
for pk in tr_packs:
    evs, ces, _, _ = build(pk)
    tof = {e["event_id"]: e["event_type"] for e in evs}
    for c in ces:
        a, b = edge_pair(c)
        pair_cnt[(tof.get(a, "UNK"), tof.get(b, "UNK"))] += 1
        cause_cnt[tof.get(a, "UNK")] += 1
types = sorted({t for p in pair_cnt for t in p})
ntypes = len(types)
tidx = {t: i for i, t in enumerate(types)}


def features(pk):
    evs, ces, qs, gold = build(pk)
    n = len(evs)
    ids = [e["event_id"] for e in evs]
    pos = {x: i for i, x in enumerate(ids)}
    tof = {e["event_id"]: e["event_type"] for e in evs}
    tt = {e["event_id"]: parse_time(e.get("occur_time")) for e in evs}
    truth = set()
    for c in ces:
        a, b = edge_pair(c)
        truth.add((a, b))
    X, y, meta = [], [], []
    for a in ids:
        for b in ids:
            if a == b:
                continue
            ta, tb = tof.get(a, "UNK"), tof.get(b, "UNK")
            pa = pair_cnt.get((ta, tb), 0)
            denom = cause_cnt.get(ta, 0) + ntypes
            prior = (pa + 1.0) / denom
            d1, d2 = pos.get(b, 0) - pos.get(a, 0), 0
            dt = None
            if tt[a] and tt[b]:
                dt = (tt[b] - tt[a]).days
            f = [
                d1, abs(d1), 1.0 if d1 == 1 else 0.0, 1.0 if d1 > 0 else 0.0,
                1.0 if pos[a] == 0 else 0.0, 1.0 if pos[b] == n - 1 else 0.0,
                prior, np.log1p(pa), 1.0 if ta == tb else 0.0,
                dt if dt is not None else 0.0,
                1.0 if dt is not None and dt >= 0 else 0.0,
            ]
            X.append(f)
            y.append(1 if (a, b) in truth else 0)
            meta.append((a, b))
    return np.array(X, dtype=np.float64), np.array(y), meta, evs, qs, gold, truth, tof


Xtr, ytr = [], []
for pk in tr_packs:
    X, y, *_ = features(pk)
    Xtr.append(X)
    ytr.append(y)
Xtr = np.vstack(Xtr)
ytr = np.concatenate(ytr)
print("训练 pair 样本:", Xtr.shape, "正例比例 %.4f" % ytr.mean())

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.15)
clf.fit(Xtr, ytr)

tp = fp = fn = 0
chain_scores = collections.defaultdict(list)
for pk in val_packs:
    X, y, meta, evs, qs, gold, truth, tof = features(pk)
    p = clf.predict_proba(X)[:, 1]
    pred = {meta[i] for i in range(len(meta)) if p[i] >= 0.5}
    tp += len(pred & truth)
    fp += len(pred - truth)
    fn += len(truth - pred)
    # chain recovery on predicted graph: shortest path between mentioned nodes
    adj = collections.defaultdict(list)
    for a, b in pred:
        adj[a].append(b)

    def bfs(s):
        dist, par = {s: 0}, {s: None}
        q = [s]
        while q:
            n0 = q.pop(0)
            for m in adj.get(n0, []):
                if m not in dist:
                    dist[m] = dist[n0] + 1
                    par[m] = n0
                    q.append(m)
        return dist, par

    def to(par, t):
        o = []
        while t is not None:
            o.append(t)
            t = par[t]
        return o[::-1]

    for q in qs:
        g = gold.get(q["sample_id"])
        if not g:
            continue
        chains = g.get("evidence_chains") or []
        if not chains or q.get("reasoning_type") not in ("temporal_causal", "counterfactual"):
            continue
        m = list(dict.fromkeys(re.findall(r"D\d{3}", q["question"])))
        if len(m) != 2:
            continue
        best = 0.0
        for cand in chains:
            for s, t in (m, m[::-1]):
                dist, par = bfs(s)
                if t in dist:
                    pp = to(par, t)
                    sa, sb = set(pp), set(cand)
                    inter = len(sa & sb)
                    f1v = 2.0 * inter / (len(sa) + len(sb)) if (sa and sb) else 0.0
                    best = max(best, f1v)
        chain_scores[q.get("reasoning_type")].append(best)

prec = tp / max(1, tp + fp)
rec = tp / max(1, tp + fn)
print("边预测(held-out): P=%.4f R=%.4f F1=%.4f" % (prec, rec, 2 * prec * rec / max(1e-9, prec + rec)))
for k, v in chain_scores.items():
    v = np.array(v)
    print("链恢复 %-16s n=%4d  mean nodeF1=%.4f  >=0.8: %.3f  exact: %.3f"
          % (k, len(v), v.mean(), (v >= 0.8).mean(), (v >= 0.999).mean()))

importances = sorted(zip(clf.feature_importances_,
                         ["di", "abs di", "adjacent", "forward", "i_first", "j_last",
                          "typePrior", "logPairCnt", "sameType", "dt", "dt_ge0"]),
                     reverse=True)
print("\n特征重要度:")
for v, n in importances:
    print("   %-12s %.4f" % (n, v))
