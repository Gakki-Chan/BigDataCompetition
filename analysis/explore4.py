# -*- coding: utf-8 -*-
"""Test-set inventory + question template families."""
import json, os, re, sys, collections, io

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TEST = os.path.join(ROOT, "赛题4", "数据集", "抽样测试集_100")
SUP = os.path.join(ROOT, "赛题4补充")


def load(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


tot = collections.Counter()
qtype = collections.Counter()
rt = collections.Counter()
templ = collections.Counter()
prefix = collections.Counter()
for tier in sorted(os.listdir(TEST)):
    tp = os.path.join(TEST, tier)
    for pk in sorted(os.listdir(tp)):
        qs = load(os.path.join(tp, pk, "问题.json"))
        tot[tier] += len(qs)
        for q in qs:
            qtype[(tier, q["question_type"])] += 1
            rt[(tier, q.get("reasoning_type"))] += 1
            t = re.sub(r"D\d{3}", "<E>", q["question"])
            t = re.sub(r"[（(][^）)]*[）)]", "(E)", t)
            t = re.sub(r"\d{4}-\d{2}-\d{2}", "<D>", t)
            templ[t[:34]] += 1

print("每档问题数:", dict(tot), "总计", sum(tot.values()))
print("\nquestion_type by tier:")
for k in sorted(qtype):
    print("   ", k, qtype[k])
print("\nreasoning_type by tier:")
for k in sorted(rt):
    print("   ", k, rt[k])
print("\n问题模板 top20:")
for k, v in templ.most_common(20):
    print("   %5d  %s" % (v, k))

# B 补充事件列表 vs 隐藏? 统计 B 事件列表事件数
n = 0
nb = 0
for pk in sorted(os.listdir(os.path.join(SUP, "测试集B_推理"))):
    d = os.path.join(SUP, "测试集B_推理", pk)
    evs = load(os.path.join(d, "事件列表.json"))
    docs = [f for f in os.listdir(d) if f.endswith(".txt")]
    nb += 1
    if len(evs) != len(docs):
        print("!! B事件数!=文档数", pk, len(evs), len(docs))
    n += len(evs)
print("\n补充B: pack 数 %d, 事件数 %d" % (nb, n))
