# -*- coding: utf-8 -*-
"""事件提及链接：把问题里的**自由文本指代**对齐到事件图节点。

为什么需要它
------------
模板化的问题会直接写 `D005（调查报告）`，正则就能抓到锚点；但真实语料里问题只会说
"那份调查报告""四部门首次联合约谈"。实测中崩掉到 41 分的两个 pack 正是这种情况。

做法（借鉴实体链接的两段式 + 多示例弱监督）
------------------------------------------
  * 候选空间被图结构天然限制：同 pack 内 8~26 个事件节点；
  * 监督信号来自金标证据链的**结构**（远程/弱监督，不需要额外标注）：
      - single_hop          -> 链首 = 问题所指事件
      - multi_hop_causal    -> 链尾 = 问题所指事件（"什么导致了 X"）
      - temporal_causal /
        counterfactual      -> 链首与链尾都是问题点名的事件
    这就是多示例学习（Dietterich et al., 1997）的设定：只有"包"级信息，包内哪个
    实例为正需要模型自己推。
  * 打分器用通用特征（文本重叠 + 位置先验 + 图统计），不写领域词典。
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from scorer import containment, cosine, jaccard_tokens
from schema import EVENT_ID_RE, mentions, type_hint


def weak_anchor_labels(question, gold) -> List[str]:
    """从金标链的结构反推"问题指的是哪个事件"（弱监督标签）。"""
    if not gold.evidence_chains:
        return []
    chain = gold.evidence_chains[0]
    rt = question.reasoning_type
    if rt == "single_hop":
        return chain[:1]
    if rt == "multi_hop_causal":
        return chain[-1:]
    if rt in ("temporal_causal", "counterfactual"):
        return [chain[0], chain[-1]] if len(chain) > 1 else list(chain)
    return []


def pair_features(pack, question, event, n_events, ctx=None) -> List[float]:
    """(问题, 事件) 对的通用特征。"""
    q = question.text
    idx = pack.event_index
    pos = idx.get(event.event_id, 0)
    args_text = " ".join(str(v) for v in (event.args or {}).values())
    hint = type_hint(q, event.event_id)
    return [
        1.0 if event.event_id in q else 0.0,
        containment(event.event_type, q, 2) if event.event_type else 0.0,
        containment(event.trigger, q, 2) if event.trigger else 0.0,
        cosine(q, event.text, 2),
        max(cosine(q, event.event_type or "", 3), containment(event.event_type or "", q, 3)),
        containment(args_text[:80], q, 2) if args_text else 0.0,
        jaccard_tokens(q, event.text),
        pos / max(1, n_events - 1),
        1.0 if pos == 0 else 0.0,
        1.0 if pos == n_events - 1 else 0.0,
        float(len(event.event_type or "")),
        float(n_events),
        1.0 if (hint and event.event_type and (
            containment(hint, event.event_type, 2) >= 0.5
            or containment(event.event_type, hint, 2) >= 0.5)) else 0.0,
        1.0 if hint else 0.0,
    ]


PAIR_FEATURE_NAMES = [
    "eid_in_q", "type_in_q", "trigger_in_q", "q_ev_cos", "type_cos",
    "args_in_q", "q_ev_jaccard", "pos_ratio", "is_first", "is_last",
    "type_len", "n_events", "hint_match", "has_hint",
]


class MentionLinker:
    """问题 -> 事件节点。输出分数，由调用方决定取哪些。"""

    def __init__(self):
        self.clf = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                              learning_rate=0.12, subsample=0.9,
                                              random_state=0)

    # ------------------------------------------------------------------ #
    def build_dataset(self, packs):
        X, y = [], []
        for pack in packs:
            n = len(pack.events)
            for q in pack.questions:
                g = pack.gold.get(q.sample_id)
                if g is None:
                    continue
                pos = set(weak_anchor_labels(q, g))
                if not pos:
                    continue
                for ev in pack.events:
                    X.append(pair_features(pack, q, ev, n))
                    y.append(1 if ev.event_id in pos else 0)
        return X, y

    def fit(self, packs):
        X, y = self.build_dataset(packs)
        if X:
            self.clf.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
        return self

    # ------------------------------------------------------------------ #
    def score_events(self, pack, question):
        n = len(pack.events)
        if n == 0:
            return []
        X = np.asarray([pair_features(pack, question, ev, n) for ev in pack.events],
                       dtype=float)
        p = self.clf.predict_proba(X)[:, 1]
        return sorted(zip(pack.event_ids, p.tolist()), key=lambda kv: -kv[1])

    def link(self, pack, question, threshold=0.5, topk=3):
        scored = self.score_events(pack, question)
        picked = [eid for eid, s in scored[:topk] if s >= threshold]
        if not picked and scored and scored[0][1] >= threshold * 0.8:
            picked = [scored[0][0]]
        # 保持事件在文档中的顺序，便于下游按出现顺序使用
        order = pack.event_index
        return sorted(picked, key=lambda e: order.get(e, 0))


def combined_anchors(pack, question, linker=None, threshold=0.5, topk=3):
    """正则锚点 ∪ 链接器锚点（正则命中的优先级更高，排前面）。"""
    node_set = set(pack.event_ids)
    regex_anchors = [m for m in mentions(question.text) if m in node_set]
    if linker is None:
        return regex_anchors
    linked = linker.link(pack, question, threshold=threshold, topk=topk)
    out = list(regex_anchors)
    for eid in linked:
        if eid not in out:
            out.append(eid)
    return out
