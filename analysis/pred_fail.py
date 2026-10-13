# -*- coding: utf-8 -*-
"""专门看 grounded_pred 的候选召回失败样本，判断金标链到底是什么形状。"""
import collections
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from chain import PackContext, anchors_of, build_graph, generate_candidates  # noqa: E402
from schema import iter_pack_dirs, load_pack  # noqa: E402

TR = os.path.join(ROOT, "赛题4", "数据集", "训练集")
dirs = iter_pack_dirs(TR)
random.Random(0).shuffle(dirs)

stat = collections.Counter()
fails = []
for d in dirs[:300]:
    pack = load_pack(d)
    graph = build_graph(pack)
    ctx = PackContext(pack)
    for q in pack.questions:
        if q.reasoning_type != "grounded_pred":
            continue
        g = pack.gold.get(q.sample_id)
        if g is None or not g.evidence_chains:
            continue
        gold = tuple(g.evidence_chains[0])
        stat["n"] += 1
        is_path = all(b in graph.successors(a) for a, b in zip(gold, gold[1:]))
        idx_sorted = all(ctx.idx.get(a, 0) < ctx.idx.get(b, 0) for a, b in zip(gold, gold[1:]))
        stat["path"] += int(is_path)
        stat["idx_sorted"] += int(idx_sorted)
        stat["len_%d" % min(len(gold), 9)] += 1
        anchors = anchors_of(pack, q)
        anchor_set = set(anchors)
        stat["gold_inside_anchors"] += int(set(gold) <= anchor_set)
        stat["gold_eq_anchors"] += int(list(gold) == list(anchors))
        stat["gold_ge_anchor_span"] += int(set(gold) >= anchor_set)
        cands = generate_candidates(pack, q, graph, ctx=ctx)
        if tuple(gold) not in [tuple(c) for c in cands]:
            if len(fails) < 8:
                fails.append((q.text[:100], anchors, list(gold),
                              sorted(ctx.idx.get(x, -1) for x in gold),
                              is_path, idx_sorted))

print("grounded_pred 样本统计：", dict(stat))
print("\n期望形状：")
print("  金标链是图上的路径            : %.3f" % (stat["path"] / max(1, stat["n"])))
print("  金标链按文档编号单调          : %.3f" % (stat["idx_sorted"] / max(1, stat["n"])))
print("  金标链全部落在问题提到的锚点内: %.3f" % (stat["gold_inside_anchors"] / max(1, stat["n"])))
print("  金标链恰好等于锚点序列        : %.3f" % (stat["gold_eq_anchors"] / max(1, stat["n"])))
print("\n召回失败样例：")
for t, a, g, pos, p, s in fails:
    print("  Q: %s" % t)
    print("     锚点=%s" % a)
    print("     金标=%s  编号=%s  路径=%s 单调=%s" % (g, pos, p, s))
