# -*- coding: utf-8 -*-
"""对比两份提交文件，定位线上掉分的来源。"""
import collections
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def load(name):
    p = os.path.join(ROOT, "submissions", name)
    return {r["sample_id"]: r for r in json.load(open(p, encoding="utf-8"))}


v1 = load(sys.argv[1] if len(sys.argv) > 1 else "submission_test_l0.json")
v2 = load(sys.argv[2] if len(sys.argv) > 2 else "submission_test_v2.json")
ids = sorted(set(v1) & set(v2))
print("对比题数：%d" % len(ids))

same_chain = same_ans = same_conf = 0
conf1, conf2 = [], []
refuse1 = refuse2 = 0
for i in ids:
    a, b = v1[i], v2[i]
    same_chain += int(a["evidence_chain"] == b["evidence_chain"])
    same_ans += int(a["answer"] == b["answer"])
    same_conf += int(a["confidence"] == b["confidence"])
    refuse1 += int(a["confidence"] is None)
    refuse2 += int(b["confidence"] is None)
    if a["confidence"] is not None:
        conf1.append(a["confidence"])
    if b["confidence"] is not None:
        conf2.append(b["confidence"])

print("\n证据链完全相同：%d / %d = %.3f" % (same_chain, len(ids), same_chain / len(ids)))
print("答案完全相同  ：%d / %d = %.3f" % (same_ans, len(ids), same_ans / len(ids)))
print("置信度完全相同：%d / %d = %.3f" % (same_conf, len(ids), same_conf / len(ids)))
print("拒答数：v1=%d  v2=%d" % (refuse1, refuse2))
print("置信度均值：v1=%.3f  v2=%.3f" % (sum(conf1) / len(conf1), sum(conf2) / len(conf2)))

def levels(vals):
    c = collections.Counter()
    for v in vals:
        for lv in (0.9, 0.7, 0.5):
            if abs(v - lv) <= 0.075:
                c[lv] += 1
                break
        else:
            c["其他"] += 1
    return dict(sorted(c.items(), key=lambda kv: str(kv[0])))


print("置信度落在档位附近：v1=%s" % levels(conf1))
print("                    v2=%s" % levels(conf2))

buckets = collections.Counter()
for v in conf2:
    buckets[round(v, 1)] += 1
print("v2 置信度分布：", dict(sorted(buckets.items())))

print("\n答案被改动的样例（v1 -> v2）：")
shown = 0
for i in ids:
    if v1[i]["answer"] != v2[i]["answer"] and shown < 3:
        shown += 1
        print("  [%s]" % i)
        print("    v1: %s" % v1[i]["answer"][:150])
        print("    v2: %s" % v2[i]["answer"][:150])
