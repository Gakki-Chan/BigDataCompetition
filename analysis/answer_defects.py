# -*- coding: utf-8 -*-
"""答案层的客观缺陷检测（与评分器无关，只看构造上对不对）：

  D1 外来事件 ID：答案里出现的事件 ID 不属于本 pack（模板残留）
  D2 方向颠倒：答案里前两个 ID 的顺序与预测链相反
  D3 外来内容：答案里的中文实体词来自另一个话题（用"模板自带内容是否被替换"近似）
"""
import collections
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from schema import (EVENT_ID_RE, iter_pack_dirs, load_pack,  # noqa: E402
                    materialize_events_from_docs)

TEST = os.path.join(ROOT, "赛题4", "数据集", "抽样测试集_100")
SUP = os.path.join(ROOT, "赛题4补充")

packs = {}
for d in iter_pack_dirs(TEST):
    p = load_pack(d, with_gold=False)
    if not p.events:
        cand = os.path.join(SUP, os.path.relpath(d, TEST))
        if os.path.exists(cand):
            s = load_pack(cand, with_gold=False)
            p.events = s.events
    materialize_events_from_docs(p)      # C 档：与流水线一致，一文档一事件
    for q in p.questions:
        packs[q.sample_id] = (p, {e.event_id for e in p.events})


def analyse(name):
    path = os.path.join(ROOT, "submissions", name)
    data = json.load(open(path, encoding="utf-8"))
    stat = collections.Counter()
    for r in data:
        pack, ids = packs.get(r["sample_id"], (None, set()))
        ans = r["answer"]
        found = list(dict.fromkeys(EVENT_ID_RE.findall(ans)))
        if not found:
            stat["无ID"] += 1
            continue
        stat["有ID"] += 1
        foreign = [x for x in found if x not in ids]
        if foreign:
            stat["D1外来ID"] += 1
        chain = r["evidence_chain"]
        if len(found) >= 2 and len(chain) >= 2:
            if found[0] == chain[1] and found[1] == chain[0]:
                stat["D2方向颠倒"] += 1
            elif found[0] == chain[0] and found[1] == chain[1]:
                stat["顺序一致"] += 1
    return stat, len(data)


for name in (sys.argv[1:] or ("submission_test_l0.json", "submission_test_v2.json",
                              "submission_test_v3.json")):
    st, n = analyse(name)
    print("%-28s n=%d" % (name, n))
    for k in ("有ID", "无ID", "D1外来ID", "D2方向颠倒", "顺序一致"):
        if st[k]:
            print("    %-10s %4d (%.3f)" % (k, st[k], st[k] / n))
