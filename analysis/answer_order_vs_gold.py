# -*- coding: utf-8 -*-
"""用 dev 金标检验：答案文本里的事件 ID 顺序，是否与金标证据链一致。"""
import collections
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from schema import EVENT_ID_RE, iter_pack_dirs, load_pack  # noqa: E402

caches = {
    "v1(按位置替换)": os.path.join(ROOT, "artifacts", "l0_soft", "pred_dev_A.json"),
    "v2(指派对齐)": os.path.join(ROOT, "artifacts", "l0_v2", "pred_dev_A.json"),
}
preds = {}
for nm, pth in caches.items():
    if os.path.exists(pth):
        preds[nm] = {r["sample_id"]: r
                     for r in json.load(open(pth, encoding="utf-8"))["preds"]}

packs = {}
for d in iter_pack_dirs(os.path.join(ROOT, "赛题4", "数据集", "训练集")):
    pack = load_pack(d)
    for q in pack.questions:
        packs[q.sample_id] = (pack, {e.event_id for e in pack.events})

for nm, pv in preds.items():
    stat = collections.Counter()
    for sid, r in pv.items():
        if sid not in packs or r.get("confidence") is None:
            continue
        pack, ids = packs[sid]
        g = pack.gold.get(sid)
        if g is None:
            continue
        gold = g.evidence_chains[0] if g.evidence_chains else []
        ans = r.get("_answer_templated") or r["answer"]
        found = list(dict.fromkeys(EVENT_ID_RE.findall(ans)))
        if not gold or len(gold) < 2 or len(found) < 2:
            stat["不可判"] += 1
            continue
        stat["可判"] += 1
        if found[0] == gold[0] and found[1] == gold[1]:
            stat["与金标同向"] += 1
        elif len(gold) >= 2 and found[0] == gold[1] and found[1] == gold[0]:
            stat["与金标反向"] += 1
        else:
            stat["其他"] += 1
        if [x for x in found if x not in ids]:
            stat["含外来ID"] += 1
    n = stat["可判"]
    print("%-16s 可判 %d" % (nm, n))
    for k in ("与金标同向", "与金标反向", "其他", "含外来ID"):
        print("     %-10s %4d (%.3f)" % (k, stat[k], stat[k] / max(1, n)))
