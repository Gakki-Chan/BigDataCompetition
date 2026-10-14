# -*- coding: utf-8 -*-
"""在同一次模板检索结果上比较三种槽位对齐方式（位置 / 单调DP / 指派），
同时报告与评分器无关的客观缺陷指标。"""
import collections
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

items = []
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
        items.append((pack, q, g, p, ids))
print("参与对比的题数：%d" % len(items))

scorer = Scorer()


def run(mode, lam):
    stat = collections.Counter()
    scores = []
    for pack, q, g, p, ids in items:
        chain = p["evidence_chain"]
        type_of = {e.event_id: e.event_type for e in pack.events}
        node_text = {e.event_id: (e.event_type + e.trigger + (e.args.get("影响") or ""))
                     for e in pack.events}
        bank.align_mode = mode
        bank.align_lambda = lam
        ans = bank.compose(q.text, p.get("_intent") or q.reasoning_type, chain, type_of,
                           abstain=False, node_text=node_text)
        scores.append(scorer.score_answer(ans, g)["score"])
        found = list(dict.fromkeys(EVENT_ID_RE.findall(ans)))
        if [x for x in found if x not in ids]:
            stat["含外来ID"] += 1
        # 主指标：答案里出现的"链内事件"顺序，是否与提交的证据链一致
        inchain = [x for x in found if x in chain]
        if len(chain) >= 2 and len(inchain) >= 2:
            stat["链可判"] += 1
            if inchain == list(chain)[:len(inchain)]:
                stat["与提交链一致"] += 1
        gold = g.evidence_chains[0] if g.evidence_chains else []
        if len(gold) >= 2 and len(found) >= 2:
            stat["可判"] += 1
            if found[0] == gold[0] and found[1] == gold[1]:
                stat["与金标同向"] += 1
            elif found[0] == gold[1] and found[1] == gold[0]:
                stat["与金标反向"] += 1
    n = len(items)
    return (sum(scores) / n, stat["与提交链一致"] / max(1, stat["链可判"]),
            stat["与金标同向"] / max(1, stat["可判"]), stat["含外来ID"] / n)


GRID = [("positional", 0.0)]
print("%-22s %8s %12s %10s %10s"
      % ("对齐方式", "答案分", "与提交链一致", "与金标同向", "含外来ID"))
for mode, lam in GRID:
    a, same, rev, foreign = run(mode, lam)
    print("%-22s %8.2f %10.3f %10.3f %10.3f"
          % ("%s(lam=%.2f)" % (mode, lam), a, same, rev, foreign))
