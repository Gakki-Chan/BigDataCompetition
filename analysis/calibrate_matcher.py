# -*- coding: utf-8 -*-
"""校准 fact 匹配阈值：正例=该题金标答案，负例=同 pack 其它题的金标答案。"""
import os, sys, random

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from schema import load_pack, iter_pack_dirs  # noqa: E402
import scorer as S  # noqa: E402

TR = os.path.join(ROOT, "赛题4", "数据集", "训练集")
random.seed(0)
dirs = iter_pack_dirs(TR)
random.shuffle(dirs)

pos, neg = [], []
for d in dirs[:250]:
    pack = load_pack(d)
    qs = [q for q in pack.questions if q.sample_id in pack.gold]
    for q in qs:
        g = pack.gold[q.sample_id]
        facts = g.required_facts or ([g.answers] if g.answers else [])
        if not facts:
            continue
        for f in facts:
            surface = max(S.containment(f, g.answers, 2), S.containment(f, g.answers, 3))
            fe = S.content_elements(f)
            elem = len(fe & S.content_elements(g.answers)) / len(fe) if fe else 0.0
            pos.append((surface, elem, max(surface, elem)))
        for q2 in qs:
            if q2.sample_id == q.sample_id:
                continue
            other = pack.gold[q2.sample_id].answers
            for f in facts:
                surface = max(S.containment(f, other, 2), S.containment(f, other, 3))
                fe = S.content_elements(f)
                elem = len(fe & S.content_elements(other)) / len(fe) if fe else 0.0
                neg.append((surface, elem, max(surface, elem)))

print("正例 %d 条，负例 %d 条" % (len(pos), len(neg)))
import statistics as st
for name, i in (("表层覆盖", 0), ("要素召回", 1), ("取大", 2)):
    p = [x[i] for x in pos]
    n = [x[i] for x in neg]
    print("  %-8s 正例中位=%.3f 负例中位=%.3f" % (name, st.median(p), st.median(n)))

print("\n阈值扫描（正例=命中）:")
print("  tau   POS命中率  NEG命中率  分离度")
for tau in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
    pr = sum(1 for x in pos if x[2] >= tau) / len(pos)
    nr = sum(1 for x in neg if x[2] >= tau) / len(neg)
    print("  %.2f   %7.3f   %7.3f   %+.3f" % (tau, pr, nr, pr - nr))
