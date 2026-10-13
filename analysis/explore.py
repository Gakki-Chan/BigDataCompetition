# -*- coding: utf-8 -*-
"""Exploratory statistics over the competition dataset."""
import json, os, re, sys, collections, glob, io

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TRAIN = os.path.join(ROOT, "赛题4", "数据集", "训练集")


def load(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


packs = sorted(d for d in os.listdir(TRAIN) if os.path.isdir(os.path.join(TRAIN, d)))
print("训练集 pack 数:", len(packs))

qt = collections.Counter()
rt = collections.Counter()
hop = collections.Counter()
chain_len = collections.Counter()
conf = collections.Counter()
nq = 0
ev_per_pack = []
edge_per_pack = []
edge_forward = 0
edge_total = 0
self_edge_types = collections.Counter()
event_type_by_domain = collections.defaultdict(collections.Counter)
edge_type_pairs = collections.Counter()
edges_adjacent = 0
edges_dist = collections.Counter()
answer_prefix = collections.Counter()
missing = []
edge_keys = collections.Counter()


def edge_pair(c):
    edge_keys[tuple(sorted(c.keys()))] += 1
    a = c.get("cause_event_id", c.get("cause", c.get("cause_id")))
    b = c.get("result_event_id", c.get("result", c.get("effect_event_id", c.get("result_id"))))
    return a, b

for pk in packs:
    d = os.path.join(TRAIN, pk)
    domain = re.sub(r"_\d+$", "", pk)
    qf = os.path.join(d, "问题.json")
    gf = os.path.join(d, "gold", "问答对_答案.json")
    ef = os.path.join(d, "事件列表.json")
    cf = os.path.join(d, "事件因果关系列表.json")
    for f in (qf, gf, ef, cf):
        if not os.path.exists(f):
            missing.append(f)
    if missing:
        continue
    qs = load(qf)
    gold = {g["sample_id"]: g for g in load(gf)}
    evs = load(ef)
    ces = load(cf)
    nq += len(qs)
    ev_per_pack.append(len(evs))
    edge_per_pack.append(len(ces))
    ids = [e["event_id"] for e in evs]
    idx = {e: i for i, e in enumerate(ids)}
    type_of = {e["event_id"]: e["event_type"] for e in evs}
    for e in evs:
        event_type_by_domain[domain][e["event_type"]] += 1
    for c in ces:
        a, b = edge_pair(c)
        edge_total += 1
        if a in idx and b in idx:
            if idx[b] > idx[a]:
                edge_forward += 1
            edges_dist[idx[b] - idx[a]] += 1
            if idx[b] == idx[a] + 1:
                edges_adjacent += 1
            edge_type_pairs[(type_of[a], type_of[b])] += 1
        else:
            self_edge_types["OOB"] += 1
    for q in qs:
        qt[q["question_type"]] += 1
        rt[q.get("reasoning_type", "NA")] += 1
        g = gold.get(q["sample_id"])
        if not g:
            continue
        if g.get("confidence_level"):
            conf[g["confidence_level"]] += 1
        chs = g.get("evidence_chains") or []
        ann = g.get("annotations") or {}
        if "hops" in ann:
            hop[ann["hops"]] += 1
        for ch in chs:
            chain_len[len(ch)] += 1
        a = (g.get("answers") or "")
        answer_prefix[a[:12]] += 1

print("\n问题总数:", nq)
print("question_type:", dict(qt))
print("reasoning_type:", dict(rt))
print("confidence_level:", dict(conf))
print("annotations.hops:", dict(hop))
print("证据链长度分布(所有候选链):", dict(sorted(chain_len.items())))
print("每 pack 事件数: min %d max %d avg %.1f" % (min(ev_per_pack), max(ev_per_pack), sum(ev_per_pack) / len(ev_per_pack)))
print("每 pack 因果边数: min %d max %d avg %.1f" % (min(edge_per_pack), max(edge_per_pack), sum(edge_per_pack) / len(edge_per_pack)))
print("边中 事件序 j>i 占比: %.4f (n=%d)" % (edge_forward / edge_total, edge_total))
print("边中相邻事件(i,i+1)占比: %.4f" % (edges_adjacent / edge_total))
print("边跨度分布 top:", edges_dist.most_common(8))
print("\n答案文本前12字 top20:")
for k, v in answer_prefix.most_common(20):
    print("   ", repr(k), v)
print("\n缺文件:", missing[:5], "count", len(missing))

print("\n各领域 event_type 集合（前3个领域）:")
for dom in list(event_type_by_domain)[:3]:
    print("  ", dom, dict(event_type_by_domain[dom].most_common(30)))

print("\n边类型对 top25:")
for k, v in edge_type_pairs.most_common(25):
    print("   ", k, v)

print("\n因果边条目的键组合:", edge_keys.most_common())
