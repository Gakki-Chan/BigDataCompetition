# -*- coding: utf-8 -*-
"""定点验证：只检查"含外来事件 ID"的那些题，确认修复有效且不伤其它指标。"""
import json
import os
import sys

import joblib

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from schema import EVENT_ID_RE, load_pack  # noqa: E402
from scorer import Scorer  # noqa: E402
from train import TRAIN_DIR  # noqa: E402

bundle = joblib.load(os.path.join(ROOT, "artifacts", "l0_v2", "bundle.joblib"))
bank = bundle["answers"]
preds = {p["sample_id"]: p for p in json.load(
    open(os.path.join(ROOT, "artifacts", "l0_v2", "pred_dev_A.json"),
         encoding="utf-8"))["preds"]}
scorer = Scorer()

cases = []
for name in sorted(set(bundle["meta"]["dev_packs"])):
    d = os.path.join(TRAIN_DIR, name)
    if not os.path.isdir(d):
        continue
    pack = load_pack(d)
    ids = {e.event_id for e in pack.events}
    for q in pack.questions:
        g = pack.gold.get(q.sample_id)
        p = preds.get(q.sample_id)
        if g is None or p is None or p.get("confidence") is None:
            continue
        ans_old = p.get("_answer_templated") or ""
        if not [x for x in EVENT_ID_RE.findall(ans_old) if x not in ids]:
            continue
        cases.append((pack, q, g, p, ids, ans_old))

print("含外来 ID 的题数：%d" % len(cases))
bank.align_mode = "positional"
bank.align_lambda = 0.0
s_old = s_new = 0.0
still_bad = 0
for pack, q, g, p, ids, old in cases:
    chain = p["evidence_chain"]
    type_of = {e.event_id: e.event_type for e in pack.events}
    node_text = {e.event_id: (e.event_type + e.trigger + (e.args.get("影响") or ""))
                 for e in pack.events}
    new = bank.compose(q.text, p.get("_intent") or q.reasoning_type, chain, type_of,
                       abstain=False, node_text=node_text)
    s_old += scorer.score_answer(old, g)["score"]
    s_new += scorer.score_answer(new, g)["score"]
    if [x for x in EVENT_ID_RE.findall(new) if x not in ids]:
        still_bad += 1

n = len(cases)
print("修复前答案分均值：%.2f" % (s_old / n))
print("修复后答案分均值：%.2f" % (s_new / n))
print("修复后仍含外来 ID：%d" % still_bad)
if cases:
    pack, q, g, p, ids, old = cases[0]
    type_of = {e.event_id: e.event_type for e in pack.events}
    node_text = {e.event_id: (e.event_type + e.trigger + (e.args.get("影响") or ""))
                 for e in pack.events}
    new = bank.compose(q.text, p.get("_intent") or q.reasoning_type, p["evidence_chain"],
                       type_of, abstain=False, node_text=node_text)
    print("\n样例 链=%s" % p["evidence_chain"])
    print("  修复前：%s" % old[:130])
    print("  修复后：%s" % new[:130])
