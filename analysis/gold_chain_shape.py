# -*- coding: utf-8 -*-
"""统计各意图下"锚点 vs 金标链"的形状，用来修正候选生成策略。"""
import collections
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from schema import iter_pack_dirs, load_pack, mentions  # noqa: E402

TR = os.path.join(ROOT, "赛题4", "数据集", "训练集")
stat = collections.defaultdict(collections.Counter)
lens = collections.defaultdict(collections.Counter)

for d in iter_pack_dirs(TR):
    pack = load_pack(d)
    ids = set(pack.event_ids)
    for q in pack.questions:
        g = pack.gold.get(q.sample_id)
        if g is None or not g.evidence_chains:
            continue
        intent = q.reasoning_type
        chain = g.evidence_chains[0]
        anchors = [m for m in mentions(q.text) if m in ids]
        s = stat[intent]
        s["n"] += 1
        s["n_anchor_%d" % min(len(anchors), 3)] += 1
        lens[intent][len(chain)] += 1
        if len(anchors) == 1:
            a = anchors[0]
            s["a_is_head"] += int(chain[0] == a)
            s["a_is_tail"] += int(chain[-1] == a)
            s["a_in_chain"] += int(a in chain)
        elif len(anchors) >= 2:
            a, b = anchors[0], anchors[1]
            s["path_a_to_b"] += int(chain[0] == a and chain[-1] == b)
            s["path_b_to_a"] += int(chain[0] == b and chain[-1] == a)
            s["edge_a_b"] += int(list(chain) == [a, b])
            s["both_in"] += int(a in chain and b in chain)
        else:
            s["no_anchor"] += 1

print("%-22s %5s %6s %6s %6s | 单锚点: head tail in | 双锚点: a->b b->a edge both"
      % ("intent", "n", "0锚", "1锚", "2锚+"))
for k, v in sorted(stat.items(), key=lambda kv: -kv[1]["n"]):
    n = v["n"]
    print("%-22s %5d %6.2f %6.2f %6.2f | %8.2f %5.2f %5.2f | %8.2f %5.2f %5.2f %5.2f"
          % (k, n, v["n_anchor_0"] / n, v["n_anchor_1"] / n, v["n_anchor_2"] / n,
             v["a_is_head"] / max(1, v["n_anchor_1"]),
             v["a_is_tail"] / max(1, v["n_anchor_1"]),
             v["a_in_chain"] / max(1, v["n_anchor_1"]),
             v["path_a_to_b"] / max(1, v["n_anchor_2"]),
             v["path_b_to_a"] / max(1, v["n_anchor_2"]),
             v["edge_a_b"] / max(1, v["n_anchor_2"]),
             v["both_in"] / max(1, v["n_anchor_2"])))

print("\n链长度分布：")
for k in sorted(lens, key=lambda x: -sum(lens[x].values())):
    tot = sum(lens[k].values())
    top = ", ".join("%d:%d" % (l, c) for l, c in sorted(lens[k].items())[:6])
    print("   %-22s n=%-5d  %s" % (k, tot, top))
