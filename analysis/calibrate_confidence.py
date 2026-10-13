# -*- coding: utf-8 -*-
"""置信度校准：把"预测金标 confidence_level"换成"预测期望答案正确率"。

评测里置信度那一项的式子是 20·(1-(conf-c)^2)，其中 c 是答案正确性。
这是平方损失（Brier 分数，proper scoring rule），其条件最优解是
    conf* = E[c | 特征]
所以正确答案不是去分类数据集给的 certain/probable/possible，而是**直接回归
答案正确率**。这里用交叉验证验证这一点，并给出最终要写进 bundle 的映射。
"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from schema import iter_pack_dirs, load_pack  # noqa: E402
from scorer import Scorer  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "artifacts", "l0_soft", "pred_dev_assign.json")
preds = {p["sample_id"]: p for p in json.load(open(path, encoding="utf-8"))["preds"]}
scorer = Scorer()

rows = []
for d in iter_pack_dirs(os.path.join(ROOT, "赛题4", "数据集", "训练集")):
    pack = load_pack(d)
    for q in pack.questions:
        g = pack.gold.get(q.sample_id)
        p = preds.get(q.sample_id)
        if g is None or p is None or not p.get("_meta_features"):
            continue
        if p.get("_p_abstain", 0.0) >= 0.7:      # 拒答的题不参与校准
            continue
        a = scorer.score_answer(p["answer"], g)["score"] / 40.0
        rows.append((np.asarray(p["_meta_features"], dtype=float), a,
                     float(p["confidence"] or 0.7), pack.name))

X = np.vstack([r[0] for r in rows])
y = np.asarray([r[1] for r in rows])
cur = np.asarray([r[2] for r in rows])
pack_names = [r[3] for r in rows]
print("参与校准的题数：%d，平均答案正确率 %.3f" % (len(y), y.mean()))


def conf_score(conf, c):
    return 20.0 * (1.0 - (conf - c) ** 2)


print("\n基线（同一批题）：")
print("  当前模型输出           %.3f" % conf_score(cur, y).mean())
print("  恒定 0.7              %.3f" % conf_score(np.full_like(y, 0.7), y).mean())
print("  恒定 0.9              %.3f" % conf_score(np.full_like(y, 0.9), y).mean())
print("  预言机（直接给答案分）  %.3f" % conf_score(y, y).mean())

# 按 pack 分半交叉验证，避免同 pack 泄漏
packs = sorted(set(pack_names))
half = set(packs[::2])
masks = [np.array([pn in half for pn in pack_names]),
         np.array([pn not in half for pn in pack_names])]

for shrink in (0.0, 0.25, 0.5, 0.75, 1.0):
    oof = np.zeros_like(y)
    for m in masks:
        reg = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                        subsample=0.8, random_state=0)
        reg.fit(X[m], y[m])
        pred = reg.predict(X[~m])
        oof[~m] = 0.7 + shrink * (pred - 0.7)
    oof = np.clip(oof, 0.05, 0.98)
    print("  校准回归(收缩 %.2f)     %.3f" % (shrink, conf_score(oof, y).mean()))

clip_lo, clip_hi = 0.35, 0.95
oof = np.zeros_like(y)
for m in masks:
    reg = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                    subsample=0.8, random_state=0)
    reg.fit(X[m], np.clip(y[m], clip_lo, clip_hi))
    oof[~m] = ret = reg.predict(X[~m])
oof = np.clip(oof, clip_lo, clip_hi)
print("  校准回归(目标截断)      %.3f" % conf_score(oof, y).mean())
