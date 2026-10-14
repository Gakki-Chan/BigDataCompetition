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

    # 类级默认值：兼容旧 pickle
    align_mode = "positional"
    gap_cost = 0.5
    align_lambda = 0.0

    def __init__(self, id_pattern=EVENT_ID_RE, topk=8, align_lambda=0.0,
                 align_mode="positional", pos_weight=0.25, gap_cost=0.5):
        self.entries: List[Dict] = []
        self.id_pattern = id_pattern
        self.topk = topk
        self.align_lambda = align_lambda
        self.align_mode = align_mode
        self.pos_weight = pos_weight
        self.gap_cost = gap_cost

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
        """模板槽位 -> 预测链节点 的对齐。

        关键：证据链是**有序序列**，所以对齐必须是**单调**的（保序）。
        用最小代价一一指派（Hungarian）会允许任意置换，实测会把因果方向调转、
        让答案与证据链自相矛盾（dev 上"与金标同向"从 0.565 掉到 0.526）。
        有序结构对应的是**单调序列对齐**，用动态规划求解（Needleman-Wunsch 家族）。
        """
        slots = entry.get("slots") or extract_slots(entry.get("answer", ""), self.id_pattern)
        if not slots or not chain:
            return {}, 0.0, set()
        mode = getattr(self, "align_mode", "monotone")
        if mode == "positional":
            mapping = {slots[i][0]: chain[i] for i in range(min(len(slots), len(chain)))}
            excess = {slots[i][0] for i in range(len(chain), len(slots))}
            # 位置映射天然保序（与提交的证据链 100% 一致），唯一缺陷是模板槽位
            # 多于链节点时会残留外来 ID，这里补齐映射把那个缺陷去掉。
            mapping = self._fill_unmatched(slots, chain, mapping)
            return mapping, 0.0, excess

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

        if mode == "assignment":
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(cost)
            mapping = {slots[i][0]: chain[j] for i, j in zip(rows, cols)}
            excess = {slots[i][0] for i in range(n_s) if i not in set(rows)}
            mapping = self._fill_unmatched(slots, chain, mapping)
            return mapping, float(cost[rows, cols].mean()), excess

        # 单调对齐（保序）：DP，允许跳过槽位或跳过链节点，代价 = 匹配代价 + 间隔罚
        gap = getattr(self, "gap_cost", 0.5)
        INF = float("inf")
        dp = [[INF] * (n_c + 1) for _ in range(n_s + 1)]
        back = [[None] * (n_c + 1) for _ in range(n_s + 1)]
        dp[0][0] = 0.0
        for i in range(n_s + 1):
            for j in range(n_c + 1):
                if dp[i][j] == INF:
                    continue
                if i < n_s and j < n_c:
                    v = dp[i][j] + cost[i, j]
                    if v < dp[i + 1][j + 1]:
                        dp[i + 1][j + 1] = v
                        back[i + 1][j + 1] = ("m", i, j)
                if i < n_s:
                    v = dp[i][j] + gap
                    if v < dp[i + 1][j]:
                        dp[i + 1][j] = v
                        back[i + 1][j] = ("s", i, j)
                if j < n_c:
                    v = dp[i][j] + gap
                    if v < dp[i][j + 1]:
                        dp[i][j + 1] = v
                        back[i][j + 1] = ("c", i, j)

        matched = {}          # slot index -> chain index
        i, j = n_s, n_c
        while back[i][j] is not None:
            kind, pi, pj = back[i][j]
            if kind == "m":
                matched[pi] = pj
            i, j = pi, pj

        # 未匹配的槽位：绝不留下外来 ID，一律跟着最近的已匹配邻居
        mapping = {slots[i][0]: chain[j] for i, j in matched.items()}
        excess = {slots[i][0] for i in range(n_s) if i not in matched}
        mapping = self._fill_unmatched(slots, chain, mapping, matched_indices=matched)
        total = sum(cost[i, matched[i]] for i in matched)
        total += gap * (n_s - len(matched))
        return mapping, total / max(1, n_s), excess

    @staticmethod
    def _fill_unmatched(slots, chain, mapping, matched_indices=None):
        """保证每个槽位都有映射目标——否则模板里的外来事件 ID 会残留在答案里。"""
        if not chain:
            return mapping
        # 已映射到的链下标（用于就近回退）
        pos_of = {cid: k for k, cid in enumerate(chain)}
        known = {}
        for i, (eid, _h) in enumerate(slots):
            if eid in mapping:
                known[i] = pos_of.get(mapping[eid])
        for i, (eid, _h) in enumerate(slots):
            if eid in mapping:
                continue
            prev_j = max((k for k in known if k < i), default=None)
            next_j = min((k for k in known if k > i), default=None)
            if prev_j is not None:
                j = known[prev_j]
            elif next_j is not None:
                j = known[next_j]
            else:
                j = min(i, len(chain) - 1)
            mapping[eid] = chain[j]
        return mapping

    # ------------------------------------------------------------------ #
    def _substitute(self, text, mapping, type_of, excess_ids=None):
        """ID 替换 + 类型名改写 + 多余槽位删除。

        三件事缺一件，答案里就会留下"外来事实"：
          1. 事件 ID 换成预测链的节点；
          2. ID 后面括号里的类型名必须换成**本 pack 中该事件的真实类型**——
             模板的类型名属于另一个 pack，不改就会出现 "D002（司法追责）" 而本 pack
             的 D002 是"深度报道"这种自相矛盾的表述；
          3. 模板槽位多于链节点时，多出来的槽位整段删掉，而不是重复指向同一节点。
        """
        excess_ids = excess_ids or set()
        # 先删多余槽位（必须用**替换前**的原始 ID 去找，否则已被映射成别的 ID 就找不到了）
        for eid in sorted(excess_ids, key=len, reverse=True):
            text = re.sub(re.escape(eid) + r"\s*[（(][^）)]*[）)]\s*[、,，]?", "", text)
            text = re.sub(re.escape(eid) + r"\s*[、,，]?", "", text)
        if mapping:
            text = self.id_pattern.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)

        id_then_paren = re.compile(r"(%s)\s*[（(]([^）)]*)[）)]" % EVENT_ID_RE.pattern)

        def _fix(m):
            eid = m.group(1)
            target = type_of.get(eid)
            return "%s（%s）" % (eid, target) if target else m.group(0)

        text = id_then_paren.sub(_fix, text)

        # 安全网：任何不属于本 pack 的事件 ID 都不允许留在答案里（模板残留）
        valid = set(type_of)
        stray = [x for x in dict.fromkeys(self.id_pattern.findall(text)) if x not in valid]
        for eid in sorted(stray, key=len, reverse=True):
            text = re.sub(re.escape(eid) + r"\s*[（(][^）)]*[）)]\s*[、,，]?", "", text)
            text = re.sub(re.escape(eid) + r"\s*[、,，]?", "", text)
        text = re.sub(r"[、,，]{2,}", "、", text)
        text = re.sub(r"[、,，]\s*([；;。])", r"\1", text)
        return text

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
            mapping, cost, excess = self._align(entry, chain, node_text)
            total = base - self.align_lambda * cost
            if best_score is None or total > best_score:
                best_score, best_text = total, self._substitute(
                    entry["answer"], mapping, type_of, excess)
        return best_text or REFUSE_TEXT
