# -*- coding: utf-8 -*-
"""在 dev 缓存上做离线调参：拒答阈值 × 置信度映射。

前提：缓存是用 --abstain-threshold 1.01（永不拒答）跑出来的，
这样每题都保留了"不拒答时的链与答案"以及 P(拒答)，扫参时不必重跑推理。
"""
import collections
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from schema import iter_pack_dirs, load_pack  # noqa: E402
from scorer import REFUSE_TEXT, Scorer  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "artifacts", "l0", "pred_dev_noabstain.json")
preds = {p["sample_id"]: p for p in json.load(open(path, encoding="utf-8"))["preds"]}

# 收集 (gold, tier, p_abstain, chain, answer, conf)
items = []
for d in iter_pack_dirs(os.path.join(ROOT, "赛题4", "数据集", "训练集")):
    pack = load_pack(d)
    for q in pack.questions:
        g = pack.gold.get(q.sample_id)
        p = preds.get(q.sample_id)
        if g is None or p is None:
            continue
        items.append((g, pack.tier, p.get("_p_abstain", 0.0), p["evidence_chain"],
                      p["answer"], p["confidence"], p.get("_intent", "")))
print("参与扫参的题目：%d" % len(items))


def run(threshold, conf_fn):
    scorer = Scorer()
    records = []
    for g, tier, p_ab, chain, answer, conf, _intent in items:
        if p_ab >= threshold:
            pred = {"answer": REFUSE_TEXT, "evidence_chain": [], "confidence": None}
        else:
            pred = {"answer": answer, "evidence_chain": chain,
                    "confidence": conf_fn(conf)}
        records.append(dict({"tier": tier}, **scorer.score_one(pred, g)))
    return scorer.aggregate(records)


conf_ops = {
    "原始": lambda c: c,
    "吸附到0.9/0.7/0.5": lambda c: min([0.9, 0.7, 0.5], key=lambda v: abs(v - (c or 0.7))),
    "上移一档": lambda c: min(1.0, (c or 0.7) + 0.2),
    "向0.7收缩75%": lambda c: 0.7 + 0.25 * ((c or 0.7) - 0.7),
    "向0.7收缩50%": lambda c: 0.7 + 0.50 * ((c or 0.7) - 0.7),
    "向0.7收缩25%": lambda c: 0.7 + 0.75 * ((c or 0.7) - 0.7),
    "恒0.9": lambda c: 0.9,
    "恒0.7": lambda c: 0.7,
}

print("\n%-18s %-8s %8s %8s %8s %8s" % ("置信度映射", "拒答阈值", "答案", "证据链", "置信度", "总分"))
best = None
for name, fn in conf_ops.items():
    for t in [0.5, 0.6, 0.7, 0.8]:
        agg = run(t, fn)
        row = (agg["weighted"], name, t, agg)
        if best is None or row[0] > best[0]:
            best = row
        print("%-18s %-8.2f %8.2f %8.2f %8.2f %8.2f"
              % (name, t, agg["ALL_answer"], agg["ALL_chain"], agg["ALL_confidence"],
                 agg["weighted"]))

print("\n最优：%s，阈值 %.2f，总分 %.2f" % (best[1], best[2], best[0]))
