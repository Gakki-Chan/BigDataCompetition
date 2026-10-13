# -*- coding: utf-8 -*-
"""端到端推理流水线（L0：纯 CPU）。

    pack -> [意图] -> [图（给定 or 预测）] -> [候选枚举] -> [排序选链]
         -> [拒答判定] -> [检索式答案组装] -> [置信度] -> 提交记录
"""
from __future__ import annotations

from typing import Dict, List, Optional

from answer import AnswerBank
from chain import PackContext, anchors_of, build_graph, candidate_features, generate_candidates
from models import AbstainModel, ChainRanker, ConfidenceModel, EdgeModel, IntentModel
from schema import Pack
from scorer import REFUSE_TEXT

INTENT_TO_QTYPE = {
    "counterfactual": "counterfactual",
    "grounded_pred": "prospective",
}
DEFAULT_QTYPE = "retrospective"


def meta_features(pack, question, chain, graph, intent, anchor_ids, prob_abstain=0.0,
                  ctx=None):
    """置信度模型的特征：候选特征 + 意图/拒答信号。"""
    f = candidate_features(pack, question, chain, graph, anchor_ids=anchor_ids, ctx=ctx)
    f.append(float(prob_abstain))
    f.append(1.0 if intent == "counterfactual" else 0.0)
    f.append(1.0 if intent == "grounded_pred" else 0.0)
    f.append(1.0 if intent == "temporal_causal" else 0.0)
    return f


class Pipeline:
    def __init__(self, intent, abstain, ranker, edge, confidence, answers,
                 abstain_threshold=0.7, confidence_prior=0.7, confidence_shrink=0.75):
        self.intent = intent
        self.abstain = abstain
        self.ranker = ranker
        self.edge = edge
        self.confidence = confidence
        self.answers = answers
        self.abstain_threshold = abstain_threshold
        self.confidence_prior = confidence_prior
        self.confidence_shrink = confidence_shrink

    def build_edges(self, pack):
        """A 档直接用给定边；B/C 档用学习到的边分类器重建图。"""
        if pack.edges:
            return {(e.cause, e.result): 1.0 for e in pack.edges}
        if self.edge is None:
            return {}
        return self.edge.predict_scores(pack)

    def predict_one(self, pack, question, edge_scores=None):
        graph = build_graph(
            pack, edge_scores if edge_scores is not None else self.build_edges(pack))
        ctx = PackContext(pack)
        intent = self.intent.predict([question.text])[0]
        p_abstain = self.abstain.prob_positive([question.text])[0]
        anchor_ids = anchors_of(pack, question)

        cands = generate_candidates(pack, question, graph, intent, ctx=ctx)
        feats = [candidate_features(pack, question, c, graph, anchor_ids=anchor_ids, ctx=ctx)
                 for c in cands]
        try:
            scores = self.ranker.score(feats, intent=intent)
        except TypeError:
            scores = self.ranker.score(feats)
        best = int(scores.argmax()) if len(scores) else 0
        chain = list(cands[best]) if len(cands) else []

        abstain = p_abstain >= self.abstain_threshold
        if abstain:
            chain, confidence, answer = [], None, REFUSE_TEXT
        else:
            mf = meta_features(pack, question, chain, graph, intent, anchor_ids, p_abstain,
                               ctx=ctx)
            raw = float(self.confidence.predict_value([mf])[0])
            # 向先验收缩：dev 实测置信度模型的排序能力弱于"恒输出 probable"，
            # 收缩系数在 dev 上扫过（0.75 与 1.0 的差距在噪声内），保留部分模型信号。
            confidence = self.confidence_prior + self.confidence_shrink * (
                raw - self.confidence_prior)
            type_of = {e.event_id: e.event_type for e in pack.events}
            answer = self.answers.compose(question.text, intent, chain, type_of, abstain=False)

        return {
            "sample_id": question.sample_id,
            "answer": answer,
            "evidence_chain": chain,
            "confidence": confidence,
            "question_type": question.question_type or INTENT_TO_QTYPE.get(intent, DEFAULT_QTYPE),
            "_intent": intent,
            "_p_abstain": p_abstain,
        }

    def predict_pack(self, pack):
        edge_scores = self.build_edges(pack)
        return [self.predict_one(pack, q, edge_scores) for q in pack.questions]
