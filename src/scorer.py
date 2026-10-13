# -*- coding: utf-8 -*-
"""本地评分器：按赛题公布的评分口径复刻一份可比对的分值。

评测口径（见 评测.md）：
  答案正确性 40 = required_facts 的 P/R/F1 × 25 + 字符级 F1 × 10 + 语义相似度 × 5
                  （命中 forbidden_facts 额外扣分）
  证据链准确 40 = 证据节点 P/R/F1 × 25 + 链顺序 / Exact Match × 15
  置信度/拒答 20 = 普通题按 confidence 与答案正确性的匹配度；无答案题只判是否输出"无法确定"
  三档加权    = 0.6·A + 0.3·B + 0.1·C

注意：官方脚本未公开，这里是**代理实现**（语义相似度用字符 n-gram 余弦近似，
不需要向量模型即可在 CPU 上跑）。它用于横向比较模型版本，不能当成官方分数。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

REFUSE_TEXT = "无法确定"

_PUNCT = re.compile(r"[\s，。、；：？！""''（）()\[\]【】《》,.;:?!\"'\-—_/\\|~`]+")
_STOP = set("的了是在有与和及或也都很就都而而且但并且因为所以导致引发造成"
            "会对从被把为由以及之其这那些个种项条尤其更最非常可能应该需要要求"
            "给出说明判断依据依据存在可否能否是否与否依次顺序先后步骤环节")
_ASCII = re.compile(r"[a-z0-9]+")
_EID = re.compile(r"[A-Z]{1,3}\d{3,}")


def content_elements(text: str):
    """把一段文本拆成"要素"集合：事件 ID + ASCII 词 + 非虚词的中文二元组。

    用途：required_facts 是判分要点、措辞与答案正文并不一致，
    直接做表层相似度会系统性低估；只比较要素（实体、数值、内容二元组）更接近人判。
    """
    t = normalize(text)
    elems = set(_EID.findall((text or "").upper()))
    elems.update(_ASCII.findall(t))
    chars = [c for c in t if "\u4e00" <= c <= "\u9fff"]
    for i in range(len(chars) - 1):
        bg = chars[i] + chars[i + 1]
        if bg[0] in _STOP and bg[1] in _STOP:
            continue
        elems.add(bg)
    return elems


def normalize(text: str) -> str:
    return _PUNCT.sub("", (text or "").strip().lower())


def ngrams(text: str, n: int = 2):
    t = normalize(text)
    if len(t) < n:
        return Counter({t: 1}) if t else Counter()
    return Counter(t[i:i + n] for i in range(len(t) - n + 1))


def cosine(a: str, b: str, n: int = 2) -> float:
    ca, cb = ngrams(a, n), ngrams(b, n)
    if not ca or not cb:
        return 0.0
    inter = sum((ca & cb).values())
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return inter / (na * nb) if na and nb else 0.0


def jaccard_tokens(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]{1,2}", normalize(a)))
    tb = set(re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]{1,2}", normalize(b)))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def similarity(a: str, b: str) -> float:
    """字符 n-gram 余弦 与 片段 Jaccard 取大，作为"语义相似"的免模型近似。"""
    return max(cosine(a, b, 2), cosine(a, b, 3) * 0.95, jaccard_tokens(a, b))


def containment(needle: str, haystack: str, n: int = 2) -> float:
    """needle 的字符 n-gram 有多少比例出现在 haystack 里。

    required_facts 是"判分要点"而非答案原句（例如"起点为 D001（事故）"），
    所以用覆盖度而不是句式相似度来衡量"这条要点是否被答到了"。
    """
    cn = ngrams(needle, n)
    if not cn:
        return 0.0
    ch = ngrams(haystack, n)
    return sum((cn & ch).values()) / sum(cn.values())


def char_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred), normalize(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    cp, cg = Counter(p), Counter(g)
    inter = sum((cp & cg).values())
    if inter == 0:
        return 0.0
    precision = inter / len(p)
    recall = inter / len(g)
    return 2 * precision * recall / (precision + recall)


def _prf(hit: int, n_pred: int, n_gold: int):
    precision = hit / n_pred if n_pred else 0.0
    recall = hit / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


class Scorer:
    """fact 命中阈值等超参集中在这里，方便在 dev 上校准。"""

    def __init__(self, fact_threshold: float = 0.38, support_threshold: float = 0.30,
                 forbidden_threshold: float = 0.72,
                 forbidden_penalty: float = 3.0, forbidden_penalty_cap: float = 10.0):
        self.fact_threshold = fact_threshold
        self.support_threshold = support_threshold
        self.forbidden_threshold = forbidden_threshold
        self.forbidden_penalty = forbidden_penalty
        self.forbidden_penalty_cap = forbidden_penalty_cap

    # ---------------- 答案 40 分 ----------------
    def score_answer(self, pred: str, gold) -> Dict[str, float]:
        facts = gold.required_facts or ([gold.answers] if gold.answers else [])
        hit = sum(1 for f in facts if self._covers(f, pred))
        # precision 的分母：答案切句后，能被"required/optional 要点 + 标准答案"支撑的句子数；
        # 未被支撑的句子视为可能编造，拉低 precision（材料外事实另由 forbidden_facts 扣分）。
        pred_units = [s for s in re.split(r"[；;。\n]", pred) if normalize(s)]
        pool = list(facts) + list(gold.optional_facts) + [gold.answers]
        pool += [s for s in re.split(r"[；;。\n]", gold.answers) if normalize(s)]
        n_pred = max(1, sum(1 for u in pred_units
                            if any(self._supported(f, u) or self._supported(u, f) for f in pool)))
        p, r, f1 = _prf(hit, n_pred, len(facts))

        cf1 = char_f1(pred, gold.answers)
        sem = max(similarity(pred, gold.answers), containment(gold.answers, pred))
        penalty = 0.0
        for f in gold.forbidden_facts:
            if self._covers(f, pred) >= self.forbidden_threshold:
                penalty += self.forbidden_penalty
        penalty = min(penalty, self.forbidden_penalty_cap)

        score = 25 * f1 + 10 * cf1 + 5 * sem - penalty
        return {"score": max(0.0, min(40.0, score)), "fact_p": p, "fact_r": r,
                "fact_f1": f1, "char_f1": cf1, "sem": sem, "penalty": penalty}

    def _covers(self, fact: str, text: str) -> bool:
        """fact 的要点是否被 text 覆盖：表层覆盖度 与 要素召回率 取大。"""
        surface = max(containment(fact, text, 2), containment(fact, text, 3))
        fe = content_elements(fact)
        elem = len(fe & content_elements(text)) / len(fe) if fe else 0.0
        return max(surface, elem) >= self.fact_threshold

    def _supported(self, pool_text: str, unit: str) -> bool:
        """判 precision 用的宽松支撑判定（阈值低于 fact 命中）。"""
        surface = max(containment(pool_text, unit, 2), containment(pool_text, unit, 3))
        pe = content_elements(pool_text)
        elem = len(pe & content_elements(unit)) / len(pe) if pe else 0.0
        return max(surface, elem) >= self.support_threshold

    # ---------------- 证据链 40 分 ----------------
    def score_chain(self, pred: List[str], gold) -> Dict[str, float]:
        gold_chains = gold.evidence_chains or []
        if not gold_chains:
            # 无答案题：期望空链
            ok = 0.0 if pred else 1.0
            return {"score": 40.0 * ok, "node_f1": ok, "exact": ok, "matched": -1}

        best = None
        for i, chain in enumerate(gold_chains):
            p, r, f1 = _prf(len(set(pred) & set(chain)), len(set(pred)), len(set(chain)))
            if pred and list(pred) == list(chain):
                order = 1.0
            else:
                order = 0.5 * (lcs_len(list(pred), list(chain)) / max(1, len(chain)))
            total = 25 * f1 + 15 * order
            if best is None or total > best["total"]:
                best = {"total": total, "node_f1": f1, "order": order, "matched": i,
                        "p": p, "r": r}
        best["score"] = min(40.0, best["total"])
        return best

    # ---------------- 置信度 / 拒答 20 分 ----------------
    def score_confidence(self, pred_confidence: Optional[float], pred_answer: str,
                         pred_chain: List[str], gold, answer_score: float) -> Dict[str, float]:
        if gold.is_unanswerable:
            refused = (REFUSE_TEXT in (pred_answer or "")) and not pred_chain \
                      and pred_confidence is None
            return {"score": 20.0 if refused else 0.0, "refused": float(refused)}
        if pred_confidence is None:
            return {"score": 0.0, "refused": 1.0}
        c = min(1.0, max(0.0, answer_score / 40.0))
        conf = min(1.0, max(0.0, pred_confidence))
        return {"score": 20.0 * (1.0 - (conf - c) ** 2), "refused": 0.0}

    # ---------------- 单题 100 分 ----------------
    def score_one(self, pred: Dict, gold) -> Dict[str, float]:
        a = self.score_answer(pred.get("answer", ""), gold)
        ch = self.score_chain(pred.get("evidence_chain") or [], gold)
        cf = self.score_confidence(pred.get("confidence"), pred.get("answer", ""),
                                   pred.get("evidence_chain") or [], gold, a["score"])
        return {"answer": a["score"], "chain": ch["score"], "confidence": cf["score"],
                "total": a["score"] + ch["score"] + cf["score"],
                "fact_f1": a["fact_f1"], "char_f1": a["char_f1"], "sem": a["sem"],
                "node_f1": ch["node_f1"], "exact": ch.get("order", 0.0)}

    # ---------------- 汇总（按 A/B/C 分档加权） ----------------
    def aggregate(self, records: List[Dict]) -> Dict[str, float]:
        if not records:
            return {}
        out: Dict[str, float] = {}
        for tier in ("A", "B", "C", "ALL"):
            rows = [r for r in records if tier == "ALL" or r["tier"] == tier]
            if not rows:
                continue
            out["%s_n" % tier] = len(rows)
            for key in ("answer", "chain", "confidence", "total"):
                out["%s_%s" % (tier, key)] = sum(r[key] for r in rows) / len(rows)
        weights = {"A": 0.6, "B": 0.3, "C": 0.1}
        present = {t: out.get("%s_total" % t) for t in weights if out.get("%s_total" % t) is not None}
        wsum = sum(weights[t] for t in present) or 1.0
        out["weighted"] = sum(weights[t] * v for t, v in present.items()) / wsum
        return out
