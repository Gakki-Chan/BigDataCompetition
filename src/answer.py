# -*- coding: utf-8 -*-
"""答案层：检索式（非参数）答案组装 + 槽位对齐。

不做自由生成，也不写死任何句式：
  1. 检索 K 条最相似的历史问题（同意图、同链形状优先，再按文本相似度）；
  2. 每条历史答案看作一个"表面模板"，模板里的事件 ID 是**槽位**；
     把"模板槽位 → 预测链节点"建成**线性指派问题**（Kuhn-Munkres / Hungarian，
     代价矩阵由槽位的类型提示与候选节点的文本相似度给出），求最小代价的一一对应；
  3. 指派代价同时用于**模板重排**：对齐得越好的模板越可能给出正确的表述；
  4. 括号里的类型名一并换成目标 pack 中该事件的类型。

指派问题是全幺模的，松弛后最优解仍是整数解，所以贪心/启发式对齐不是必须的——
这点让"槽位替换"从工程 trick 变成一个有最优解的建模问题。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scorer import REFUSE_TEXT, cosine, containment
from schema import EVENT_ID_RE, PAREN_RE


def extract_slots(text, id_pattern=EVENT_ID_RE):
    """把一段模板答案里的"事件 ID（类型提示）"抽成有序槽位。"""
    slots, seen = [], set()
    for m in id_pattern.finditer(text or ""):
        eid = m.group(0)
        if eid in seen:
            continue
        seen.add(eid)
        tail = (text or "")[m.end():m.end() + 40]
        pm = PAREN_RE.match(tail.lstrip())
        slots.append((eid, pm.group(1).strip() if pm else ""))
    return slots


class AnswerBank:
    # 类级默认值：兼容旧版本 pickle 出来的实例（没有这些实例属性的情况）
    topk = 8
    align_lambda = 0.6
    use_assignment = True
    pos_weight = 0.25

    def __init__(self, id_pattern=EVENT_ID_RE, topk=8, align_lambda=0.6,
                 use_assignment=True, pos_weight=0.25):
        self.entries: List[Dict] = []
        self.id_pattern = id_pattern
        self.topk = topk
        self.align_lambda = align_lambda
        self.use_assignment = use_assignment
        self.pos_weight = pos_weight

    # ------------------------------------------------------------------ #
    def add_pack(self, pack):
        for q in pack.questions:
            g = pack.gold.get(q.sample_id)
            if g is None or g.is_unanswerable:
                continue
            chain = g.evidence_chains[0] if g.evidence_chains else []
            self.entries.append({
                "q": q.text,
                "intent": q.reasoning_type,
                "chain": list(chain),
                "answer": g.answers,
                "n_anchor": len(list(dict.fromkeys(self.id_pattern.findall(q.text)))),
                "slots": extract_slots(g.answers, self.id_pattern),
            })
        return self

    def fit(self, packs):
        for p in packs:
            self.add_pack(p)
        return self

    # ------------------------------------------------------------------ #
    def _retrieve_topk(self, qtext, intent, chain, k=8):
        pool = [e for e in self.entries if e["intent"] == intent] or self.entries
        n_anchor = len(list(dict.fromkeys(self.id_pattern.findall(qtext))))
        scored = []
        for e in pool:
            shape = 1.0 if len(e["chain"]) == len(chain) else -abs(len(e["chain"]) - len(chain)) * 0.1
            anchor = 1.0 if e["n_anchor"] == n_anchor else 0.0
            sim = max(cosine(qtext, e["q"], 2), cosine(qtext, e["q"], 3) * 0.95)
            # 问题文本里除了 ID 之外的措辞相似度（把 ID 抹掉再比）
            q_a = self.id_pattern.sub("", qtext)
            q_b = self.id_pattern.sub("", e["q"])
            sim2 = max(cosine(q_a, q_b, 3), containment(q_a, q_b, 3))
            scored.append((sim * 0.5 + sim2 * 1.0 + shape * 0.6 + anchor * 0.3, e))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]

    # ------------------------------------------------------------------ #
    def _align(self, entry, chain, node_text):
        """模板槽位 -> 预测链节点 的最小代价一一指派（线性指派问题）。"""
        slots = entry.get("slots") or extract_slots(entry.get("answer", ""), self.id_pattern)
        if not slots or not chain:
            return {}, 0.0
        if not self.use_assignment:
            mapping = {slots[i][0]: chain[i] for i in range(min(len(slots), len(chain)))}
            return mapping, 0.0

        from scipy.optimize import linear_sum_assignment

        n_s, n_c = len(slots), len(chain)
        cost = np.zeros((n_s, n_c), dtype=float)
        for i, (_eid, hint) in enumerate(slots):
            for j, cid in enumerate(chain):
                text = node_text.get(cid, "")
                sim = 0.0
                if hint and text:
                    sim = max(containment(hint, text, 2), cosine(hint, text, 2),
                              containment(text, hint, 2))
                pos_s = i / max(1, n_s - 1)
                pos_c = j / max(1, n_c - 1)
                cost[i, j] = (1.0 - sim) + self.pos_weight * abs(pos_s - pos_c)
        rows, cols = linear_sum_assignment(cost)
        mapping = {slots[i][0]: chain[j] for i, j in zip(rows, cols)}
        return mapping, float(cost[rows, cols].mean())

    # ------------------------------------------------------------------ #
    def _substitute(self, text, mapping, type_of):
        if mapping:
            text = self.id_pattern.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)

        def _fix_paren(m):
            inner = m.group(1)
            ids = self.id_pattern.findall(inner)
            if not ids:
                return m.group(0)
            target = type_of.get(ids[0])
            return "（%s）" % target if target else m.group(0)

        return PAREN_RE.sub(_fix_paren, text)

    # ------------------------------------------------------------------ #
    def compose(self, qtext, intent, chain, type_of, abstain=False, node_text=None):
        """返回答案文本。type_of: {event_id: event_type}；node_text: {event_id: 节点文本}。"""
        if abstain:
            return REFUSE_TEXT
        node_text = node_text or {k: v for k, v in type_of.items()}
        cands = self._retrieve_topk(qtext, intent, chain, self.topk)
        if not cands:
            return REFUSE_TEXT
        best_text, best_score = None, None
        for base, entry in cands:
            mapping, cost = self._align(entry, chain, node_text)
            total = base - self.align_lambda * cost
            if best_score is None or total > best_score:
                best_score, best_text = total, self._substitute(
                    entry["answer"], mapping, type_of)
        return best_text or REFUSE_TEXT
