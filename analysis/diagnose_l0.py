# -*- coding: utf-8 -*-
"""按意图拆解 L0 的失分，定位下一步该改哪里。"""
import collections
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from schema import load_pack, iter_pack_dirs  # noqa: E402
from scorer import Scorer, lcs_len  # noqa: E402

PRED_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "artifacts", "l0", "pred_dev_v2.json")
preds = {p["sample_id"]: p for p in json.load(
    open(PRED_PATH, encoding="utf-8"))["preds"]}
TR = os.path.join(ROOT, "赛题4", "数据集", "训练集")

scorer = Scorer()
agg = collections.defaultdict(lambda: collections.Counter())
examples = collections.defaultdict(list)

for d in iter_pack_dirs(TR):
    pack = load_pack(d)
    for q in pack.questions:
        g = pack.gold.get(q.sample_id)
        pr = preds.get(q.sample_id)
        if g is None or pr is None:
            continue
        intent = q.reasoning_type
        a = agg[intent]
        a["n"] += 1
        gold_chains = [tuple(c) for c in g.evidence_chains]
        pred_chain = tuple(pr["evidence_chain"])
        exact = int(pred_chain in gold_chains) if gold_chains else int(not pred_chain)
        a["exact"] += exact
        if g.is_unanswerable:
            a["unanswerable"] += 1
            a["refused"] += int(pr["confidence"] is None and not pred_chain)
        if exact == 0 and len(examples[intent]) < 3:
            examples[intent].append((q.text[:90], list(pred_chain),
                                     [list(c) for c in gold_chains][:2]))

print("%-22s %5s %8s %8s %10s" % ("intent", "n", "链准确", "拒答正确", "空链占比"))
for k, v in sorted(agg.items(), key=lambda kv: -kv[1]["n"]):
    print("%-22s %5d %8.3f %8s %10.3f"
          % (k, v["n"], v["exact"] / max(1, v["n"]),
             ("%.3f" % (v["refused"] / v["unanswerable"])) if v["unanswerable"] else "-",
             v["unanswerable"] / max(1, v["n"])))

print("\n各意图最差案例（预测链 vs 金标链）：")
for k in ("multi_hop_causal", "temporal_causal", "counterfactual", "single_hop",
          "grounded_pred"):
    for qtext, pc, gc in examples[k][:2]:
        print("  [%s] %s" % (k, qtext))
        print("      预测: %s" % pc)
        print("      金标: %s" % gc)
