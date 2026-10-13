# -*- coding: utf-8 -*-
"""看看"无答案/干扰/冲突"类样本的金标长什么样，用来定评分口径。"""
import os, sys, io, json, collections

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from schema import load_pack, iter_pack_dirs  # noqa: E402

TR = os.path.join(ROOT, "赛题4", "数据集", "训练集")
dirs = iter_pack_dirs(TR)
shown = collections.defaultdict(int)
stat = collections.Counter()
for d in dirs:
    pack = load_pack(d)
    for q in pack.questions:
        g = pack.gold.get(q.sample_id)
        if not g:
            continue
        rt = q.reasoning_type
        if rt not in ("unanswerable", "distractor_robustness", "conflicting_sources",
                      "missing_information"):
            continue
        stat[(rt, bool(g.evidence_chains), g.confidence_level is None)] += 1
        if shown[rt] < 3:
            shown[rt] += 1
            print("=" * 70)
            print("[%s] %s" % (rt, q.text))
            print("  answers      :", g.answers[:160])
            print("  required     :", g.required_facts[:3])
            print("  chains       :", g.evidence_chains[:2])
            print("  confidence   :", g.confidence_level)
            print("  annotations  :", json.dumps(g.annotations, ensure_ascii=False)[:200])

print("\n统计 (reasoning_type, 有链?, confidence为null?):")
for k, v in sorted(stat.items()):
    print("   ", k, v)
