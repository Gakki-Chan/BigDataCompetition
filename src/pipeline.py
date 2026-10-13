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
from schema import Pack, materialize_events_from_docs
from scorer import REFUSE_TEXT

INTENT_TO_QTYPE = {
    "counterfactual": "counterfactual",
    "grounded_pred": "prospective",
}
DEFAULT_QTYPE = "retrospective"
ALLOWED_QTYPES = ("retrospective", "prospective", "counterfactual")


def normalize_qtype(raw, intent):
    """提交格式只接受 retrospective / prospective / counterfactual 三值。

    数据集里还有 unanswerable / fact_query 等取值，统一归到 retrospective
    （拒答由 answer 与 confidence 判定，与该字段无关）。
    """
    if raw in ALLOWED_QTYPES:
        return raw
    mapped = INTENT_TO_QTYPE.get(intent)
    return mapped if mapped in ALLOWED_QTYPES else DEFAULT_QTYPE


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
                 abstain_threshold=0.7, confidence_prior=0.7, confidence_shrink=0.75,
                 style_router=None, extractive=None, extractive_threshold=0.5,
                 correctness=None):
        self.intent = intent
        self.abstain = abstain
        self.ranker = ranker
        self.edge = edge
        self.confidence = confidence
        self.answers = answers
        self.abstain_threshold = abstain_threshold
        self.confidence_prior = confidence_prior
        self.confidence_shrink = confidence_shrink
        self.style_router = style_router
        self.extractive = extractive
        self.extractive_threshold = extractive_threshold
        self.correctness = correctness

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
        answer_tpl, answer_ext, p_ext, mf = "", "", 0.0, None
        if abstain:
            chain, confidence, answer = [], None, REFUSE_TEXT
        else:
            mf = meta_features(pack, question, chain, graph, intent, anchor_ids, p_abstain,
                               ctx=ctx)
            if self.correctness is not None:
                # 置信度 = 预测的期望答案正确率（平方损失下的条件最优解）
                confidence = float(self.correctness.predict_value([mf])[0])
            else:
                raw = float(self.confidence.predict_value([mf])[0])
                confidence = self.confidence_prior + self.confidence_shrink * (
                    raw - self.confidence_prior)
            type_of = {e.event_id: e.event_type for e in pack.events}
            node_text = {e.event_id: (e.event_type + e.trigger + (e.args.get("影响") or ""))
                         for e in pack.events}
            answer_tpl = self.answers.compose(question.text, intent, chain, type_of,
                                              abstain=False, node_text=node_text)
            p_ext = 0.0
            answer_ext = ""
            if self.extractive is not None:
                answer_ext = self.extractive.answer(pack, question, focus_events=set(chain))
            if self.style_router is not None and self.extractive is not None:
                p_ext = self.style_router.prob_extractive(question.text)
            answer = answer_ext if p_ext >= self.extractive_threshold else answer_tpl

        return {
            "sample_id": question.sample_id,
            "answer": answer,
            "evidence_chain": chain,
            "confidence": confidence,
            "question_type": normalize_qtype(question.question_type, intent),
            "_intent": intent,
            "_p_abstain": p_abstain,
            "_p_extractive": p_ext,
            "_answer_templated": answer_tpl,
            "_answer_extractive": answer_ext,
            "_meta_features": (list(mf) if not abstain else None),
        }

    def predict_pack(self, pack):
        materialize_events_from_docs(pack)
        edge_scores = self.build_edges(pack)
        return [self.predict_one(pack, q, edge_scores) for q in pack.questions]
