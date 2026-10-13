# -*- coding: utf-8 -*-
"""按领域拆解分数：大领域（250/234/232/228 个 pack）vs 小领域（各 8 个）。

小领域的模板与大领域不同，可以当作"数据分布偏移"的近似观测。
"""
import collections
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from schema import iter_pack_dirs, load_pack  # noqa: E402
from scorer import Scorer, REFUSE_TEXT  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "artifacts", "l0", "pred_dev_final.json")
preds = {p["sample_id"]: p for p in json.load(open(path, encoding="utf-8"))["preds"]}

BIG = {"公共安全", "经济贸易", "政治外交", "自然灾害"}
scorer = Scorer()
per_domain = collections.defaultdict(list)
for d in iter_pack_dirs(os.path.join(ROOT, "赛题4", "数据集", "训练集")):
    pack = load_pack(d)
    for q in pack.questions:
        g = pack.gold.get(q.sample_id)
        p = preds.get(q.sample_id)
        if g is None or p is None:
            continue
        per_domain[pack.domain].append(dict({"tier": pack.tier}, **scorer.score_one(p, g)))

print("%-12s %6s %8s %8s %8s %8s" % ("领域", "n", "答案", "证据链", "置信度", "总分"))
groups = {"大领域(4个)": [], "小领域(7个)": []}
for dom, rows in sorted(per_domain.items(), key=lambda kv: -len(kv[1])):
    agg = scorer.aggregate(rows)
    print("%-12s %6d %8.2f %8.2f %8.2f %8.2f"
          % (dom, len(rows), agg["ALL_answer"], agg["ALL_chain"],
             agg["ALL_confidence"], agg["ALL_total"]))
    groups["大领域(4个)" if dom in BIG else "小领域(7个)"].extend(rows)

print()
for name, rows in groups.items():
    agg = scorer.aggregate(rows)
    print("%-12s %6d %8.2f %8.2f %8.2f %8.2f"
          % (name, len(rows), agg["ALL_answer"], agg["ALL_chain"],
             agg["ALL_confidence"], agg["ALL_total"]))
