# -*- coding: utf-8 -*-
"""Oracle ceiling: best achievable evidence-chain F1 by searching the given graph."""
import json, os, re, sys, collections, io

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


def all_paths(adj, maxlen=6):
    out = []
    nodes = list(adj.keys())
    for s in nodes:
        q = [(s, [s])]
        while q:
            n, p = q.pop(0)
            for m in adj.get(n, []):
                if m in p:
                    continue
                np_ = p + [m]
                out.append(np_)
                if len(np_) < maxlen:
                    q.append((m, np_))
    return out


def f1(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    p, r = inter / len(sa), inter / len(sb)
    return 2 * p * r / (p + r) if p + r else 0.0


packs = sorted(d for d in os.listdir(TRAIN) if os.path.isdir(os.path.join(TRAIN, d)))
res = collections.defaultdict(collections.Counter)
ratio = collections.defaultdict(float)
for pk in packs:
    d = os.path.join(TRAIN, pk)
    qs = load(os.path.join(d, "问题.json"))
    gold = {g["sample_id"]: g for g in load(os.path.join(d, "gold", "问答对_答案.json"))}
    ces = load(os.path.join(d, "事件因果关系列表.json"))
    adj = collections.defaultdict(list)
    for c in ces:
        a, b = edge_pair(c)
        adj[a].append(b)
    cands = all_paths(adj)
    cands1 = [[n] for n in adj]
    cands_all = cands + cands1
    for q in qs:
        g = gold.get(q["sample_id"])
        if not g:
            continue
        rt = q.get("reasoning_type")
        chains = g.get("evidence_chains") or []
        if not chains:
            continue
        best = 0.0
        for ch in chains:
            for c in cands_all:
                best = max(best, f1(c, ch))
        res[rt]["n"] += 1
        ratio[rt] += best
        if best >= 0.999:
            res[rt]["exact"] += 1
        elif best >= 0.8:
            res[rt]["ge08"] += 1
        elif best >= 0.5:
            res[rt]["ge05"] += 1
        else:
            res[rt]["lt05"] += 1

print("%-22s %6s %7s %7s %7s %7s %8s" % ("reasoning_type", "n", "exact", ">=0.8", ">=0.5", "<0.5", "meanF1"))
for rt in sorted(res):
    n = res[rt]["n"]
    print("%-22s %6d %7.3f %7.3f %7.3f %7.3f %8.3f" % (
        rt, n, res[rt]["exact"] / n, res[rt]["ge08"] / n, res[rt]["ge05"] / n,
        res[rt]["lt05"] / n, ratio[rt] / n))
