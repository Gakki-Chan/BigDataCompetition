# -*- coding: utf-8 -*-
"""答案层：检索式（非参数）答案组装。

不做自由生成，也不写死任何句式：
  1. 在训练集里检索"最相似的问题"（同意图、同链形状优先，再按问题文本相似度）；
  2. 取其金标答案作为"表面模板"，把里面的事件 ID 按**预测链的位置**替换过去；
  3. 括号里的类型名一并换成目标 pack 中该事件的类型。

换成别的赛题时，只需要换 id 的正则与"槽位"的定义，框架不变。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from scorer import REFUSE_TEXT, cosine, containment
from schema import EVENT_ID_RE, PAREN_RE


class AnswerBank:
    def __init__(self, id_pattern=EVENT_ID_RE):
        self.entries: List[Dict] = []
        self.id_pattern = id_pattern

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
            })
        return self

    def fit(self, packs):
        for p in packs:
            self.add_pack(p)
        return self

    # ------------------------------------------------------------------ #
    def _retrieve(self, qtext, intent, chain):
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
        return scored[0][1] if scored else None

    # ------------------------------------------------------------------ #
    def compose(self, qtext, intent, chain, type_of, abstain=False):
        """返回答案文本。type_of: {event_id: event_type}。"""
        if abstain:
            return REFUSE_TEXT
        entry = self._retrieve(qtext, intent, chain)
        if entry is None:
            return REFUSE_TEXT
        text = entry["answer"]

        # 1) ID 位置映射：用检索到的问题的链做参照，把答案里的 ID 换成预测链的 ID
        ref_chain = entry["chain"]
        mapping = {}
        for i, rid in enumerate(ref_chain):
            if i < len(chain):
                mapping[rid] = chain[i]
        # 预测链里没覆盖到的 ID（如答案提到链外事件）→ 退化为"链内最近位置"
        if chain and ref_chain:
            for rid in self.id_pattern.findall(text):
                if rid not in mapping:
                    try:
                        j = ref_chain.index(rid)
                    except ValueError:
                        continue
                    mapping[rid] = chain[min(j, len(chain) - 1)]

        def _sub(m):
            return mapping.get(m.group(0), m.group(0))

        if mapping:
            text = self.id_pattern.sub(_sub, text)

        # 2) 括号里的类型名替换成目标 pack 的真实类型
        def _fix_paren(m):
            inner = m.group(1)
            ids = self.id_pattern.findall(inner)
            if not ids:
                return m.group(0)
            target = type_of.get(ids[0])
            if not target:
                return m.group(0)
            return "（%s）" % target

        text = PAREN_RE.sub(_fix_paren, text)
        return text
