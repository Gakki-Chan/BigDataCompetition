# -*- coding: utf-8 -*-
"""Oracle probes: how far can a generic graph/generation procedure get on the given data?"""
import json, os, re, sys, collections, io, itertools

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


def shortest_paths(adj, s, t, maxlen=6):
    """all simple paths s->t up to maxlen nodes, shortest-first"""
    out, q = [], [(s, [s])]
    while q:
        node, path = q.pop(0)
        if len(path) > maxlen:
            continue
        for nxt in adj.get(node, []):
            if nxt in path:
                continue
            np = path + [nxt]
            if nxt == t:
                out.append(np)
            else:
                q.append((nxt, np))
    best = min((len(p) for p in out), default=None)
    return [p for p in out if best is not None and len(p) <= best + 1]


def descendants(adj, s, maxlen=6):
    out, q = [], [(s, [s])]
    while q:
        node, path = q.pop(0)
        if len(path) > maxlen:
            continue
        for nxt in adj.get(node, []):
            if nxt in path:
                continue
            np = path + [nxt]
            out.append(np)
            q.append((nxt, np))
    return out


def _path_ok(chain, adj, radj):
    for a, b in zip(chain, chain[1:]):
        if b not in adj.get(a, []):
            return False
    return True


packs = sorted(d for d in os.listdir(TRAIN) if os.path.isdir(os.path.join(TRAIN, d)))
one_to_one = 0
multi_event_docs = collections.Counter()
missing_doc_event = 0
n_ev = 0

stats = collections.Counter()
chain_hit = collections.Counter()
by_reason = collections.defaultdict(collections.Counter)
answer_gap = collections.Counter()
n_ans_match_chain = 0
n_ans_total = 0

for pk in packs:
    d = os.path.join(TRAIN, pk)
    qs = load(os.path.join(d, "问题.json"))
    gold = {g["sample_id"]: g for g in load(os.path.join(d, "gold", "问答对_答案.json"))}
    evs = load(os.path.join(d, "事件列表.json"))
    ces = load(os.path.join(d, "事件因果关系列表.json"))
    ids = [e["event_id"] for e in evs]
    docs = {e["doc_id"] for e in evs}
    if len(ids) == len(set(ids)):
        one_to_one += 1
    n_ev += len(evs)
    perdoc = collections.Counter(e["doc_id"] for e in evs)
    for k, v in perdoc.items():
        multi_event_docs[v] += 1
    adj = collections.defaultdict(list)
    radj = collections.defaultdict(list)
    for c in ces:
        a, b = edge_pair(c)
        adj[a].append(b)
        radj[b].append(a)
    type_of = {e["event_id"]: e["event_type"] for e in evs}
    for q in qs:
        g = gold.get(q["sample_id"])
        if not g:
            continue
        qtext = q["question"]
        mentioned = re.findall(r"D\d{3}", qtext)
        chains = g.get("evidence_chains") or []
        if not chains:
            continue
        gold_chain = chains[0]
        # does the gold chain appear as a graph path?
        ok = any(len(c) == len(gold_chain) and all(x == y for x, y in zip(c, gold_chain))
                 for c in chains)
        ispath = _path_ok(gold_chain, adj, radj)
        stats["chains"] += 1
        if ok:
            stats["chain_in_list"] += 1
        if ispath:
            stats["chain_is_path"] += 1
        if len(gold_chain) >= 2:
            stats["chain_ge2"] += 1
            if ispath:
                stats["chain_ge2_path"] += 1
        # answer text mentions the chain ids in order?
        ans = g.get("answers") or ""
        ans_ids = re.findall(r"D\d{3}", ans)
        n_ans_total += 1
        if ans_ids[:len(gold_chain)] == list(gold_chain):
            n_ans_match_chain += 1
        by_reason[q.get("reasoning_type")]["n"] += 1
        if ispath:
            by_reason[q.get("reasoning_type")]["path"] += 1


print("事件总数:", n_ev)
print("pack 中 event_id 唯一:", one_to_one, "/", len(packs))
print("每个 doc 的事件数分布 (doc->count):", dict(multi_event_docs))
print()
print("证据链是图中路径的比例:", stats["chain_is_path"], "/", stats["chains"],
      "= %.4f" % (stats["chain_is_path"] / max(1, stats["chains"])))
print("长度>=2 的链是图中路径的比例:", stats["chain_ge2_path"], "/", stats["chain_ge2"],
      "= %.4f" % (stats["chain_ge2_path"] / max(1, stats["chain_ge2"])))
print("答案文本前 n 个 D-id 等于 gold 链:", n_ans_match_chain, "/", n_ans_total,
      "= %.4f" % (n_ans_match_chain / max(1, n_ans_total)))
print()
for k, v in by_reason.items():
    print("   ", k, dict(v), "path率=%.3f" % (v["path"] / max(1, v["n"])))
