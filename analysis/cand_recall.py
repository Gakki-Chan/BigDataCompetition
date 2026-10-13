# -*- coding: utf-8 -*-
"""候选召回诊断：金标链是否出现在候选集里（区分"生成不足"与"排序不好"）。"""
import collections
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from chain import build_graph, generate_candidates  # noqa: E402
from schema import iter_pack_dirs, load_pack, mentions  # noqa: E402

TR = os.path.join(ROOT, "赛题4", "数据集", "训练集")
dirs = iter_pack_dirs(TR)
random.Random(0).shuffle(dirs)

stat = collections.defaultdict(collections.Counter)
sizes = collections.defaultdict(list)
pred_examples = []

for d in dirs[:300]:
    pack = load_pack(d)
    graph = build_graph(pack)
    ids = set(pack.event_ids)
    for q in pack.questions:
        g = pack.gold.get(q.sample_id)
        if g is None:
            continue
        intent = q.reasoning_type
        golds = [tuple(c) for c in g.evidence_chains] or [()]
        cands = generate_candidates(pack, q, graph, intent)
        stat[intent]["n"] += 1
        stat[intent]["cand_recall"] += int(any(tuple(c) in golds for c in cands))
        sizes[intent].append(len(cands))
        if intent == "grounded_pred" and len(pred_examples) < 6:
            anchors = [m for m in mentions(q.text) if m in ids]
            pred_examples.append((q.text[:110], anchors, [list(c) for c in golds][:2],
                                  [list(c) for c in cands[:4]]))

print("%-22s %5s %10s %10s" % ("intent", "n", "候选召回", "平均候选数"))
for k, v in sorted(stat.items(), key=lambda kv: -kv[1]["n"]):
    print("%-22s %5d %10.3f %10.1f"
          % (k, v["n"], v["cand_recall"] / max(1, v["n"]),
             sum(sizes[k]) / max(1, len(sizes[k]))))

print("\ngrounded_pred 样例（问题 / 锚点 / 金标链 / 候选前几条）：")
for t, a, g, c in pred_examples:
    print("  Q: %s" % t)
    print("     锚点=%s 金标=%s" % (a, g))
    print("     候选=%s" % c)
