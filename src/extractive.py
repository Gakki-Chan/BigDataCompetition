# -*- coding: utf-8 -*-
"""抽取式作答路径：当答案需要"读原文回答"而不是"套因果链模板"时启用。

动机（实测）
------------
dev 里有 2 个 pack 是纯真新闻问答，答案是长事实描述（含时间、金额、人名），
金标答案的绝大部分内容**原样出现在文档里**；这类题上"检索+槽位替换"的模板式作答
直接失效（那 2 个 pack 只有 41 分）。

做法（都可迁移，不含领域规则）
------------------------------
1. **自监督路由**：用"金标答案有多少比例的字面内容出现在本 pack 文档里"作为标签，
   自动判定"这题该走抽取式还是模板式"。这个标签完全从数据里算出来，不需要人工标注，
   换数据集时自动成立。
2. **抽取式作答**：句子级检索 + MMR 去冗余（Carbonell & Goldstein, 1998），
   从文档里挑与问题最相关、彼此不重复的若干句拼成答案。这是抽取式问答/答案句选择的
   经典形式（如 answer sentence selection 一路），不需要生成模型。
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from scorer import containment, cosine, ngrams

SENT_SPLIT = re.compile(r"(?<=[。！？；])")


def split_sentences(text, min_len=8):
    out = []
    for s in SENT_SPLIT.split(text or ""):
        s = s.strip()
        if len(s) >= min_len:
            out.append(s)
    return out


def document_containment(answer, docs_text, n=2):
    """金标答案有多少比例的字符 n-gram 出现在文档里（0~1）。"""
    an = ngrams(answer, n)
    if not an:
        return 0.0
    dn = ngrams(docs_text, n)
    return sum((an & dn).values()) / sum(an.values())


class AnswerStyleRouter:
    """问题 -> 该走抽取式还是模板式。二分类，标签自监督生成。"""

    EXTRACTIVE_THRESHOLD = 0.55

    def __init__(self):
        self.pipe = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2,
                            sublinear_tf=True),
            LogisticRegression(max_iter=1200, C=6.0, class_weight="balanced"),
        )
        self.n_extractive = 0
        self.n_total = 0

    def fit(self, packs):
        texts, labels = [], []
        for pack in packs:
            docs_text = " ".join(pack.docs.values())
            for q in pack.questions:
                g = pack.gold.get(q.sample_id)
                if g is None or not g.answers:
                    continue
                ratio = document_containment(g.answers, docs_text)
                texts.append(q.text)
                labels.append(1 if ratio >= self.EXTRACTIVE_THRESHOLD else 0)
        self.n_total = len(labels)
        self.n_extractive = int(sum(labels))
        if texts:
            self.pipe.fit(texts, labels)
        return self

    def prob_extractive(self, question_text):
        classes = list(self.pipe.classes_)
        if 1 not in classes:
            return 0.0
        p = self.pipe.predict_proba([question_text])[0]
        return float(p[classes.index(1)])


class ExtractiveAnswerer:
    """句子级检索 + MMR 去冗余。"""

    def __init__(self, lam=0.6, max_sentences=5):
        self.lam = lam
        self.max_sentences = max_sentences

    def answer(self, pack, question, focus_events=None):
        q = question.text
        # 句子池：优先取"被链到的事件"所在文档，其次全 pack（长文档场景下这是标准做法）
        docs = pack.docs
        focus_docs = {e.doc_id for e in pack.events
                      if focus_events and e.event_id in focus_events}
        pool = []
        for did, text in docs.items():
            weight = 1.15 if did in focus_docs else 1.0
            for sent in split_sentences(text):
                pool.append((sent, weight))
        if not pool:
            return ""

        # 与问题的相关性
        scored = [(cosine(q, s, 2) * 2.0 + containment(s, q, 2) * w, s)
                  for s, w in pool]
        scored.sort(key=lambda kv: -kv[0])
        chosen = []
        for _rel, sent in scored:
            if len(chosen) >= self.max_sentences:
                break
            redundancy = max((cosine(sent, c, 2) for c in chosen), default=0.0)
            if redundancy > self.lam:
                continue
            chosen.append(sent)
        order = {s: i for i, (s, _w) in enumerate(pool)}
        chosen.sort(key=lambda s: order.get(s, 0))
        return "".join(chosen)
