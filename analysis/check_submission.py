# -*- coding: utf-8 -*-
"""按 提交要求.md 校验提交文件的格式与完整性。"""
import collections
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from schema import iter_pack_dirs, load_pack  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "submissions", "submission_test_l0.json")
data = json.load(open(path, encoding="utf-8"))
print("文件：%s" % path)
print("条数：%d" % len(data))

# 期望的 sample_id 全集
expected = []
per_tier = collections.Counter()
for d in iter_pack_dirs(os.path.join(ROOT, "赛题4", "数据集", "抽样测试集_100")):
    pack = load_pack(d, with_gold=False)
    if not pack.events:
        cand = os.path.join(ROOT, "赛题4补充", os.path.relpath(
            d, os.path.join(ROOT, "赛题4", "数据集", "抽样测试集_100")))
        if os.path.exists(cand):
            sup = load_pack(cand, with_gold=False)
            pack.events, pack.edges = sup.events, sup.edges
    for q in pack.questions:
        expected.append(q.sample_id)
        per_tier[q.sample_id] = pack.tier
print("测试集应有题数：%d" % len(expected))

ids = [r.get("sample_id") for r in data]
missing = set(expected) - set(ids)
extra = set(ids) - set(expected)
dup = [k for k, v in collections.Counter(ids).items() if v > 1]
print("缺失：%d  多余：%d  重复：%d" % (len(missing), len(extra), len(dup)))
if missing:
    print("  缺失示例：", list(missing)[:5])

REQUIRED = ["sample_id", "answer", "evidence_chain", "confidence", "question_type"]
bad = []
for r in data:
    for k in REQUIRED:
        if k not in r:
            bad.append((r.get("sample_id"), "缺字段 " + k))
            break
    else:
        if not isinstance(r["answer"], str) or not r["answer"].strip():
            bad.append((r["sample_id"], "answer 为空或非字符串"))
        if not isinstance(r["evidence_chain"], list) or any(
                not isinstance(x, str) for x in r["evidence_chain"]):
            bad.append((r["sample_id"], "evidence_chain 非字符串列表"))
        if r["confidence"] is not None and not (
                isinstance(r["confidence"], (int, float)) and 0 <= r["confidence"] <= 1):
            bad.append((r["sample_id"], "confidence 越界或类型错"))
        if r["confidence"] is None and r["evidence_chain"]:
            bad.append((r["sample_id"], "拒答却给了证据链"))
        if r["question_type"] not in ("retrospective", "prospective", "counterfactual"):
            bad.append((r["sample_id"], "question_type 非法：%s" % r["question_type"]))
print("\n格式问题：%d" % len(bad))
for sid, why in bad[:10]:
    print("  %s: %s" % (sid, why))

print("\n统计：")
refuse = sum(1 for r in data if r["confidence"] is None)
print("  拒答（confidence=null，链为空）：%d / %d = %.3f" % (refuse, len(data), refuse / len(data)))
print("  question_type 分布：", collections.Counter(r["question_type"] for r in data))
lens = collections.Counter(len(r["evidence_chain"]) for r in data)
print("  证据链长度分布：", dict(sorted(lens.items())))
by_tier = collections.Counter()
for r in data:
    by_tier[per_tier.get(r["sample_id"], "?")] += 1
print("  按档位：", dict(by_tier))
empty_answer = sum(1 for r in data if r["answer"] == "无法确定")
print("  答案=\"无法确定\" 的题数：%d" % empty_answer)
