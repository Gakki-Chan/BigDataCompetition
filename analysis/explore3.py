# -*- coding: utf-8 -*-
"""Oracle probe 2: mentioned nodes vs gold chain; shortest-path baselines."""
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


def bfs(adj, s, maxdepth=6):
    """dist and shortest path from s"""
    dist = {s: 0}
    par = {s: None}
    q = [s]
    while q:
        n = q.pop(0)
        if dist[n] >= maxdepth:
            continue
        for m in adj.get(n, []):
            if m not in dist:
                dist[m] = dist[n] + 1
                par[m] = n
                q.append(m)
    return dist, par


def path_to(par, t):
    out = []
    while t is not None:
        out.append(t)
        t = par[t]
    return out[::-1]


def _p(ch, adj):
    return all(b in adj.get(a, []) for a, b in zip(ch, ch[1:]))


packs = sorted(d for d in os.listdir(TRAIN) if os.path.isdir(os.path.join(TRAIN, d)))
cnt = collections.Counter()
exact_fw = collections.Counter()
recall = collections.Counter()

for pk in packs:
    d = os.path.join(TRAIN, pk)
    qs = load(os.path.join(d, "问题.json"))
    gold = {g["sample_id"]: g for g in load(os.path.join(d, "gold", "问答对_答案.json"))}
    evs = load(os.path.join(d, "事件列表.json"))
    ces = load(os.path.join(d, "事件因果关系列表.json"))
    adj = collections.defaultdict(list)
    for c in ces:
        a, b = edge_pair(c)
        adj[a].append(b)
    for q in qs:
        g = gold.get(q["sample_id"])
        if not g:
            continue
        chains = g.get("evidence_chains") or []
        if not chains:
            continue
        rt = q.get("reasoning_type")
        cnt[(rt, "n")] += 1
        # best-acceptable chain is a path?
        if any(_p(ch, adj) for ch in chains):
            cnt[(rt, "some_chain_is_path")] += 1
        m = re.findall(r"D\d{3}", q["question"])
        m = list(dict.fromkeys(m))
        if len(m) == 2:
            cnt[(rt, "two_mentions")] += 1
            a, b = m
            dist_a, par_a = bfs(adj, a)
            dist_b, par_b = bfs(adj, b)
            gold_set = set(chains[0])
            best = 0.0
            for src, tgt, dist, par in ((a, b, dist_a, par_a), (b, a, dist_b, par_b)):
                if tgt in dist:
                    p = path_to(par, tgt)
                    inter = len(set(p) & gold_set)
                    f1 = 2.0 * inter / (len(p) + len(gold_set))
                    best = max(best, f1)
                    if list(chains[0]) == p:
                        exact_fw[(rt, "exact")] += 1
            recall[(rt, "sum")] += best
            cnt[(rt, "bfs_f1_ge_08")] += 1 if best >= 0.8 else 0


for rt in sorted({k[0] for k in cnt}):
    n = cnt[(rt, "n")]
    print("%-22s n=%-5d somePath=%.3f  twoMentions=%-5d exactShortestPath=%.3f  bfsF1>=0.8=%.3f  meanBFStoGoldF1=%.3f"
          % (rt, n, cnt[(rt, "some_chain_is_path")] / n,
             cnt[(rt, "two_mentions")],
             exact_fw[(rt, "exact")] / n,
             cnt[(rt, "bfs_f1_ge_08")] / n,
             recall[(rt, "sum")] / n))
