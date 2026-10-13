# -*- coding: utf-8 -*-
"""检验假设：作答风格应该在 **pack 级** 决策，而不是逐题决策。

同一 pack 内的题目风格是同质的（要么全是模板生成的问答，要么全是真新闻问答），
所以用包内所有题的 p_extractive 均值来做整包决策，信噪比远高于逐题决策。
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from schema import iter_pack_dirs, load_pack
from scorer import Scorer, REFUSE_TEXT

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "artifacts", "l0_full", "pred_dev_v2.json")
preds = {p["sample_id"]: p for p in json.load(open(path, encoding="utf-8"))["preds"]}
scorer = Scorer()


def conf_shrink(c):
    return 0.7 + 0.75 * ((c if c is not None else 0.7) - 0.7)


packs = []
for d in iter_pack_dirs(os.path.join(ROOT, "赛题4", "数据集", "训练集")):
    pack = load_pack(d)
    qs = [q for q in pack.questions if q.sample_id in preds and q.sample_id in pack.gold]
    if qs:
        packs.append((pack, qs))
print("dev pack 数：%d" % len(packs))

means = []
for pack, qs in packs:
    m = sum(preds[q.sample_id].get("_p_extractive", 0.0) for q in qs) / len(qs)
    means.append((m, pack.name, len(qs)))
means.sort(reverse=True)
print("包级 p_extractive 均值最高的 5 个：", [(n, round(m, 3)) for m, n, _ in means[:5]])
print("最低的 3 个：", [(n, round(m, 3)) for m, n, _ in means[-3:]])


def evaluate(pack_thr, q_thr, mode="pk"):
    records = []
    for pack, qs in packs:
        pmean = sum(preds[q.sample_id].get("_p_extractive", 0.0) for q in qs) / len(qs)
        for q in qs:
            pr = preds[q.sample_id]
            g = pack.gold[q.sample_id]
            if pr.get("_p_abstain", 0.0) >= 0.7:
                pred = {"answer": REFUSE_TEXT, "evidence_chain": [], "confidence": None}
            else:
                if mode == "q":
                    use_ext = pr.get("_p_extractive", 0.0) >= q_thr
                elif mode == "pk":
                    use_ext = pmean >= pack_thr
                else:
                    use_ext = (pmean >= pack_thr
                               and pr.get("_p_extractive", 0.0) >= q_thr)
                pred = {"answer": (pr.get("_answer_extractive") if use_ext
                                   else pr.get("_answer_templated")) or pr["answer"],
                        "evidence_chain": pr["evidence_chain"],
                        "confidence": conf_shrink(pr["confidence"])}
            records.append(dict({"tier": pack.tier}, **scorer.score_one(pred, g)))
    return scorer.aggregate(records)


base = evaluate(2.0, 0.5, mode="pk")["weighted"]
print("\n=== pack 级决策（阈值扫描）===")
print("%-10s %8s %8s %8s" % ("包级阈值", "答案", "总分", "对比关闭"))
best = None
for t in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7]:
    agg = evaluate(t, 0.5, mode="pk")
    print("%-10.2f %8.2f %8.2f %+8.2f"
          % (t, agg["ALL_answer"], agg["weighted"], agg["weighted"] - base))
    if best is None or agg["weighted"] > best[0]:
        best = (agg["weighted"], t)
print("关闭抽取式基线：%.2f" % base)
print("最优包级阈值 %.2f -> %.2f" % (best[1], best[0]))
