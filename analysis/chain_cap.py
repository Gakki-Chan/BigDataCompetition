# -*- coding: utf-8 -*-
"""检验"证据链长度上限"对分数的影响（赛题规定 1~5 跳，但我们生成了更长的主干链）。"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from schema import iter_pack_dirs, load_pack  # noqa: E402
from scorer import Scorer  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "artifacts", "l0_soft", "pred_dev_A.json")
preds = {p["sample_id"]: p for p in json.load(open(path, encoding="utf-8"))["preds"]}
scorer = Scorer()

items = []
for d in iter_pack_dirs(os.path.join(ROOT, "赛题4", "数据集", "训练集")):
    pack = load_pack(d)
    for q in pack.questions:
        g = pack.gold.get(q.sample_id)
        p = preds.get(q.sample_id)
        if g is None or p is None:
            continue
        items.append((g, pack.tier, p))

print("%-8s %8s %8s %8s" % ("链长上限", "证据链", "答案", "总分"))
for cap in [4, 5, 6, 8, 10, 999]:
    recs = []
    for g, tier, p in items:
        chain = p["evidence_chain"][:cap] if cap < 999 else p["evidence_chain"]
        pred = {"answer": p["answer"], "evidence_chain": chain,
                "confidence": p["confidence"]}
        recs.append(dict({"tier": tier}, **scorer.score_one(pred, g)))
    agg = scorer.aggregate(recs)
    print("%-8s %8.2f %8.2f %8.2f"
          % (cap if cap < 999 else "无", agg["ALL_chain"], agg["ALL_answer"],
             agg["ALL_total"]))
