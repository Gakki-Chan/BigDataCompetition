# -*- coding: utf-8 -*-
"""只重拟合置信度校准器（答案层修好后，校准目标需要重算）。

避免整轮重训：其余组件与答案层修复无关，只有"期望正确率"这个回归目标变了。
"""
import json
import os
import sys
import time

import joblib

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import chain as chain_mod  # noqa: E402
from models import CorrectnessRegressor  # noqa: E402
from pipeline import Pipeline  # noqa: E402
from schema import load_pack  # noqa: E402
from scorer import Scorer  # noqa: E402
from train import TRAIN_DIR  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "artifacts", "l0_v2", "bundle.joblib")
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    ROOT, "artifacts", "l0_v3", "bundle.joblib")

b = joblib.load(src)
if b.get("linker") is not None:
    chain_mod.LINKER = b["linker"]
cfg = (b.get("meta") or {}).get("config") or {}
pipe = Pipeline(b["intent"], b["abstain"], b["ranker"], b["edge"], b["confidence"],
                b["answers"], style_router=b.get("style_router"),
                extractive=b.get("extractive"))
pipe.abstain_threshold = cfg.get("abstain_threshold", 0.7)
pipe.extractive_threshold = cfg.get("extractive_threshold", 1.01)

scorer = Scorer()
dev = [load_pack(os.path.join(TRAIN_DIR, n)) for n in sorted(set(b["meta"]["dev_packs"]))
       if os.path.isdir(os.path.join(TRAIN_DIR, n))]
print("用 %d 个 dev pack 重算校准目标" % len(dev))

t0 = time.time()
X, y = [], []
for pack in dev:
    for pred, q in zip(pipe.predict_pack(pack), pack.questions):
        g = pack.gold.get(q.sample_id)
        mf = pred.get("_meta_features")
        if g is None or not mf or pred["confidence"] is None:
            continue
        X.append(mf)
        y.append(scorer.score_answer(pred["answer"], g)["score"] / 40.0)
corr = CorrectnessRegressor().fit(X, y)
print("新校准集：%d 条，平均答案正确率 %.3f（%.1fs）"
      % (len(y), sum(y) / len(y), time.time() - t0))
vals = corr.predict_value(X[:200])
print("置信度输出示例：min %.3f 均值 %.3f max %.3f"
      % (min(vals), sum(vals) / len(vals), max(vals)))

b["correctness"] = corr
os.makedirs(os.path.dirname(dst), exist_ok=True)
joblib.dump(b, dst, compress=3)
print("已保存：%s" % dst)
