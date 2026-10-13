# -*- coding: utf-8 -*-
"""可训练模块集合（全部基于 sklearn，CPU 可跑；换成 torch 只替换实现，接口不变）。

  IntentModel      : 问题 -> 意图（reasoning_type）
  AbstainModel     : 问题 -> 是否应输出"无法确定"
  ChainRanker      : (问题, 候选链) -> 相关性分数，用于在候选里选证据链
  EdgeModel        : (事件对) -> 因果边存在概率（B/C 档没有给边时用）
  ConfidenceModel  : (问题, 链) -> 置信度档位 certain/probable/possible
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


class IntentModel:
    """问题文本 -> 意图。字符 n-gram TF-IDF + 线性分类器（无任何关键词规则）。"""

    def __init__(self, **kw):
        self.pipe = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2,
                            sublinear_tf=True),
            LogisticRegression(max_iter=1200, C=8.0, class_weight="balanced"),
        )

    def fit(self, texts, labels):
        self.pipe.fit(texts, labels)
        return self

    def predict(self, texts):
        return list(self.pipe.predict(texts))

    def predict_proba(self, texts):
        return self.pipe.predict_proba(texts)

    @property
    def classes_(self):
        return list(self.pipe.classes_)


class AbstainModel:
    """问题 -> 是否应拒答（输出"无法确定"）。"""

    def __init__(self):
        self.pipe = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2,
                            sublinear_tf=True),
            LogisticRegression(max_iter=1200, C=6.0, class_weight="balanced"),
        )

    def fit(self, texts, labels):
        self.pipe.fit(texts, labels)
        return self

    def predict_proba(self, texts):
        return self.pipe.predict_proba(texts)

    def predict(self, texts, threshold=0.5):
        p = self.predict_proba(texts)
        idx = list(self.pipe.classes_).index(1) if 1 in self.pipe.classes_ else 0
        return [1 if x[idx] >= threshold else 0 for x in p]

    def prob_positive(self, texts):
        """返回 P(应拒答)。模型没见过正类时返回全 0。"""
        p = self.predict_proba(texts)
        classes = list(self.pipe.classes_)
        if 1 not in classes:
            return [0.0 for _ in range(len(texts))]
        idx = classes.index(1)
        return [float(row[idx]) for row in p]

    @property
    def classes_(self):
        return list(self.pipe.classes_)


class ChainRanker:
    """候选链打分排序。用 GBDT 做 pointwise 打分，按分数取 top-1 / top-k。"""

    def __init__(self):
        self.clf = GradientBoostingClassifier(n_estimators=260, max_depth=3,
                                              learning_rate=0.12, subsample=0.9,
                                              random_state=0)

    def fit(self, X, y):
        self.clf.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
        return self

    def score(self, X):
        X = np.asarray(X, dtype=float)
        if len(X) == 0:
            return np.zeros(0)
        return self.clf.predict_proba(X)[:, 1]

    def rank(self, X):
        s = self.score(X)
        return list(np.argsort(-s))


class MoEBase:
    MIN_SAMPLES = 300

    def __init__(self, base_factory):
        self.base_factory = base_factory
        self.rankers = {}
        self.global_ranker = base_factory()

    def fit(self, X, y, intents):
        self.global_ranker.fit(X, y)
        by_intent = {}
        for row, lab, it in zip(X, y, intents):
            by_intent.setdefault(it, ([], []))
            by_intent[it][0].append(row)
            by_intent[it][1].append(lab)
        for it, (xs, ys) in by_intent.items():
            if len(ys) >= self.MIN_SAMPLES and sum(1 for v in ys if v > 0) >= 10:
                self.rankers[it] = self.base_factory().fit(xs, ys)
        return self

    def score(self, X, intent=None):
        r = self.rankers.get(intent)
        if r is not None:
            return r.score(X)
        return self.global_ranker.score(X)


class MoEChainRanker(MoEBase):
    """按（预测的）意图分流到不同排序器——模型层面的混合专家，不是规则分支。

    候选特征里"形状"与"意图"的耦合很强（例如同一组特征在单跳题和多跳题里的最优
    选择不同），让每个意图拥有自己的排序器能显著提升 top-1 准确率。
    """

    def __init__(self, **kw):
        MoEBase.__init__(self, ChainRanker)
        self.kw = kw


class MoESoftChainRanker(MoEBase):
    """软标签版本：目标对齐评测口径（部分分），见 SoftLabelChainRanker。"""

    def __init__(self, **kw):
        MoEBase.__init__(self, SoftLabelChainRanker)
        self.kw = kw


class SoftLabelChainRanker:
    """多示例软标签排序器。

    标准答案给了多条**可接受**证据链，评测也是"取最高分那条标准链比较"，且按节点
    F1 给部分分。所以候选不应该只有 0/1 标签：用"与最好那条金标链的对齐程度"作为
    连续目标（0.625·节点F1 + 0.375·顺序分，与 25/15 的配分一致），直接对齐评测口径。
    这是多示例学习（Dietterich et al., 1997）在链级监督上的自然形式。
    """

    def __init__(self):
        from sklearn.ensemble import GradientBoostingRegressor
        self.clf = GradientBoostingRegressor(n_estimators=260, max_depth=3,
                                             learning_rate=0.12, subsample=0.9,
                                             random_state=0)

    def fit(self, X, y):
        self.clf.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def score(self, X):
        X = np.asarray(X, dtype=float)
        if len(X) == 0:
            return np.zeros(0)
        return self.clf.predict(X)


class EdgeModel:
    """事件对 -> 因果边概率。B/C 档没给边时用它重建图。

    特征：位置差 / 时间差 / 类型对先验 / 事件文本重叠等（全部通用）。
    """

    def __init__(self, n_estimators=140, max_depth=3, learning_rate=0.15):
        self.clf = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                              learning_rate=learning_rate, subsample=0.9,
                                              random_state=0)
        self.type_pair = {}
        self.cause_total = {}
        self.types = []

    # -- 先验统计只在训练折上估计 --
    def fit_priors(self, packs):
        from collections import Counter
        pair, cause = Counter(), Counter()
        for pack in packs:
            tof = {e.event_id: e.event_type for e in pack.events}
            for ed in pack.edges:
                a, b = tof.get(ed.cause, "UNK"), tof.get(ed.result, "UNK")
                pair[(a, b)] += 1
                cause[a] += 1
        self.type_pair = dict(pair)
        self.cause_total = dict(cause)
        self.types = sorted({t for p in pair for t in p} | {"UNK"})
        return self

    def prior(self, ta, tb):
        k = (ta, tb)
        return (self.type_pair.get(k, 0) + 1.0) / (self.cause_total.get(ta, 0) + len(self.types) + 1.0)

    def features(self, pack):
        """返回 (X, meta)；meta 为 (cause_id, result_id)。"""
        ids = pack.event_ids
        idx = pack.event_index
        ev_of = {e.event_id: e for e in pack.events}
        X, meta = [], []
        for a in ids:
            for b in ids:
                if a == b:
                    continue
                ea, eb = ev_of[a], ev_of[b]
                delta = idx[b] - idx[a]
                ta, tb = ea.event_type, eb.event_type
                dt = 0.0
                if ea.time and eb.time:
                    dt = float(_days(ea.time, eb.time))
                X.append([
                    float(delta), float(abs(delta)), 1.0 if delta == 1 else 0.0,
                    1.0 if delta > 0 else 0.0,
                    1.0 if idx[a] == 0 else 0.0,
                    1.0 if idx[b] == len(ids) - 1 else 0.0,
                    self.prior(ta, tb), math.log1p(self.type_pair.get((ta, tb), 0)),
                    1.0 if ta == tb else 0.0, dt,
                    1.0 if dt >= 0 else 0.0,
                    _safe_cos(ea.text, eb.text),
                ])
                meta.append((a, b))
        return np.asarray(X, dtype=float), meta

    def fit(self, packs):
        X, y = [], []
        for pack in packs:
            truth = {(e.cause, e.result) for e in pack.edges}
            fx, meta = self.features(pack)
            X.append(fx)
            y.extend([1 if m in truth else 0 for m in meta])
        self.clf.fit(np.vstack(X), np.asarray(y, dtype=int))
        return self

    def predict_scores(self, pack, threshold=0.5):
        fx, meta = self.features(pack)
        if len(meta) == 0:
            return {}
        p = self.clf.predict_proba(fx)[:, 1]
        return {(a, b): float(w) for (a, b), w in zip(meta, p) if w >= threshold}


EDGE_FEATURE_NAMES = [
    "idx_delta", "abs_idx_delta", "adjacent", "forward", "cause_is_first",
    "result_is_last", "type_prior", "log_type_pair_cnt", "same_type",
    "day_delta", "day_forward", "text_cos",
]


class ConfidenceModel:
    """(问题, 链) -> 置信度档位。档位到数值的映射沿用赛题规定：0.9 / 0.7 / 0.5。"""

    LABELS = ["certain", "probable", "possible"]
    VALUES = {"certain": 0.9, "probable": 0.7, "possible": 0.5}

    def __init__(self):
        self.clf = GradientBoostingClassifier(n_estimators=120, max_depth=3,
                                              learning_rate=0.15, random_state=0)

    def fit(self, X, labels):
        self.clf.fit(np.asarray(X, dtype=float), np.asarray(labels))
        return self

    def predict_value(self, X):
        X = np.asarray(X, dtype=float)
        if len(X) == 0:
            return []
        proba = self.clf.predict_proba(X)
        classes = list(self.clf.classes_)
        out = []
        for row in proba:
            out.append(sum(p * self.VALUES.get(c, 0.5) for c, p in zip(classes, row)))
        return out


class CorrectnessRegressor:
    """置信度 = 预测的期望答案正确率 E[c | 特征]。

    评测的置信度项是 20·(1-(conf-c)^2)，属于平方损失（Brier 分数，proper scoring
    rule），其条件最优解就是 conf* = E[c|特征]。所以不该去分类数据集标注的
    certain/probable/possible，而应**直接回归答案正确率**。

    训练数据必须是"诚实的"预测（在模型没见过的 pack 上跑出来的），否则模型会学到
    "我永远是对的"。所以这个回归器在训练流程的最后、用留出的 dev 折来拟合，
    相当于一个校准集（calibration set）。
    """

    LO, HI = 0.05, 0.98

    def __init__(self):
        from sklearn.ensemble import GradientBoostingRegressor
        self.reg = GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                             learning_rate=0.05, subsample=0.8,
                                             random_state=0)
        self.fitted = False
        self.base = 0.7

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if len(y) >= 100:
            self.reg.fit(X, np.asarray(y, dtype=float))
            self.fitted = True
            self.base = float(np.mean(y))
        return self

    def predict_value(self, X):
        X = np.asarray(X, dtype=float)
        if len(X) == 0:
            return []
        if not self.fitted:
            return [self.base] * len(X)
        return list(np.clip(self.reg.predict(X), self.LO, self.HI))


def _days(a, b):
    import datetime
    import re as _re

    def parse(s):
        s = str(s)
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y-%m"):
            try:
                return datetime.datetime.strptime(s[:len(fmt) + 2 if fmt == "%Y-%m" else 10], fmt)
            except Exception:
                continue
        m = _re.match(r"(\d{4})", s)
        return datetime.datetime(int(m.group(1)), 1, 1) if m else None

    da, db = parse(a), parse(b)
    if not da or not db:
        return 0
    return (db - da).days


def _safe_cos(a, b):
    from scorer import cosine
    return cosine(a, b, 2)
