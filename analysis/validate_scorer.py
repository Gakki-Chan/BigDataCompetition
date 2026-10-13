# -*- coding: utf-8 -*-
"""校准评分器：oracle 上界 + 几个退化基线，确认分数量纲合理。"""
import os, sys, collections

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from schema import load_pack, iter_pack_dirs  # noqa: E402
from scorer import Scorer, REFUSE_TEXT  # noqa: E402

TR = os.path.join(ROOT, "赛题4", "数据集", "训练集")
CONF_MAP = {"certain": 0.9, "probable": 0.7, "possible": 0.5}

dirs = iter_pack_dirs(TR)
step = max(1, len(dirs) // 150)
sample = dirs[::step]
print("参与校准的 pack 数:", len(sample))

scorer = Scorer()
rows = {"oracle": [], "all_events": [], "empty_answer": [], "always_refuse": []}

for d in sample:
    pack = load_pack(d)
    if not pack.questions or not pack.gold:
        continue
    all_ids = pack.event_ids
    for q in pack.questions:
        g = pack.gold.get(q.sample_id)
        if not g:
            continue
        base = {"tier": pack.tier, "sample_id": q.sample_id}

        if g.is_unanswerable:
            oracle = {"answer": REFUSE_TEXT, "evidence_chain": [], "confidence": None}
        else:
            oracle = {"answer": g.answers, "evidence_chain": g.evidence_chains[0],
                      "confidence": CONF_MAP.get(g.confidence_level or "", 0.7)}
        rows["oracle"].append(dict(base, **scorer.score_one(oracle, g)))

        rows["all_events"].append(dict(base, **scorer.score_one(
            {"answer": g.answers, "evidence_chain": all_ids, "confidence": 0.7}, g)))

        rows["empty_answer"].append(dict(base, **scorer.score_one(
            {"answer": "", "evidence_chain": [], "confidence": 0.7}, g)))

        rows["always_refuse"].append(dict(base, **scorer.score_one(
            {"answer": REFUSE_TEXT, "evidence_chain": [], "confidence": None}, g)))

for name, recs in rows.items():
    agg = scorer.aggregate(recs)
    print("\n=== %s (n=%d) ===" % (name, len(recs)))
    print("  weighted=%.2f  A=%.2f(n=%d) B=%.2f(n=%d) C=%.2f(n=%d)"
          % (agg["weighted"], agg.get("A_total", -1), agg.get("A_n", 0),
             agg.get("B_total", -1), agg.get("B_n", 0),
             agg.get("C_total", -1), agg.get("C_n", 0)))
    print("  答案=%.2f 证据链=%.2f 置信度=%.2f"
          % (agg.get("ALL_answer", 0), agg.get("ALL_chain", 0), agg.get("ALL_confidence", 0)))

# 细看 oracle 没拿满的样本
bad = sorted(rows["oracle"], key=lambda r: r["total"])[:8]
print("\noracle 上最低的 8 题：")
for r in bad:
    print("   %.1f  答案%.1f 链%.1f 置信%.1f  %s  %s"
          % (r["total"], r["answer"], r["chain"], r["confidence"], r["tier"], r["sample_id"]))
