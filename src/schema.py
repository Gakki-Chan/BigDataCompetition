# -*- coding: utf-8 -*-
"""统一 Schema 适配层。

把任意一个"文档包"（docs + 可选事件列表 + 可选因果边 + 问题 + 可选金标）
归一化成下面这套与赛题无关的内部表示：

    Pack
      docs      : {doc_id: 正文}
      events    : [Event]          —— 图节点
      edges     : [Edge]           —— 图边（可能为空 = 任务B/C）
      questions : [Question]       —— 待回答
      gold      : {sample_id: Gold}

换赛题时只需要改本文件的字段映射，下游模块不动。
"""
from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

EVENT_ID_RE = re.compile(r"[A-Z]\d{3,}")
PAREN_RE = re.compile(r"[（(]([^）)]*)[）)]")


def read_json(path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def read_text(path):
    with io.open(path, encoding="utf-8") as f:
        return f.read()


@dataclass
class Event:
    event_id: str
    doc_id: str
    event_type: str
    trigger: str
    args: Dict[str, str] = field(default_factory=dict)
    time: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self):
        """给编码器/特征用的单事件文本。"""
        parts = [self.event_type, self.trigger]
        for k in ("主体", "地点", "内容", "影响"):
            v = self.args.get(k)
            if v:
                parts.append(v)
        return " ".join(p for p in parts if p)


@dataclass
class Edge:
    cause: str
    result: str
    causal_type: str = ""
    confidence: str = ""


@dataclass
class Question:
    sample_id: str
    pack_id: str
    text: str
    question_type: str = ""
    reasoning_type: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Gold:
    sample_id: str
    answers: str = ""
    required_facts: List[str] = field(default_factory=list)
    optional_facts: List[str] = field(default_factory=list)
    forbidden_facts: List[str] = field(default_factory=list)
    evidence_chains: List[List[str]] = field(default_factory=list)
    confidence_level: Optional[str] = None
    annotations: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_unanswerable(self):
        return (not self.evidence_chains) and (self.confidence_level is None)


@dataclass
class Pack:
    name: str
    domain: str
    docs: Dict[str, str] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    questions: List[Question] = field(default_factory=list)
    gold: Dict[str, Gold] = field(default_factory=dict)

    @property
    def event_index(self):
        return {e.event_id: i for i, e in enumerate(self.events)}

    @property
    def event_ids(self):
        return [e.event_id for e in self.events]

    @property
    def tier(self):
        """A: 给了事件+边；B: 给了事件没给边；C: 都没给。"""
        if self.events and self.edges:
            return "A"
        if self.events:
            return "B"
        return "C"


# --------------------------------------------------------------------------- #
# 加载
# --------------------------------------------------------------------------- #
def _edge_pair(raw):
    """兼容多种字段命名（赛题给出的因果边出现过 5 种键组合）。"""
    a = raw.get("cause_event_id") or raw.get("cause") or raw.get("cause_id")
    b = (raw.get("result_event_id") or raw.get("result")
         or raw.get("effect_event_id") or raw.get("result_id"))
    return a, b


def _parse_event(raw):
    args = raw.get("argument") or {}
    if not isinstance(args, dict):
        args = {"内容": str(args)}
    return Event(
        event_id=raw.get("event_id") or raw.get("doc_id"),
        doc_id=raw.get("doc_id") or raw.get("event_id"),
        event_type=raw.get("event_type", "") or raw.get("type", ""),
        trigger=raw.get("trigger_word", "") or raw.get("trigger", ""),
        args=args,
        time=str(raw.get("occur_time", "") or ""),
        raw=raw,
    )


def _parse_question(raw):
    return Question(
        sample_id=raw["sample_id"],
        pack_id=str(raw.get("pack_id", "")),
        text=raw.get("question", ""),
        question_type=raw.get("question_type", ""),
        reasoning_type=raw.get("reasoning_type", "") or raw.get("question_type", ""),
        raw=raw,
    )


def _parse_gold(raw):
    facts = raw.get("answer_facts") or {}
    if not isinstance(facts, dict):
        facts = {}
    return Gold(
        sample_id=raw["sample_id"],
        answers=raw.get("answers", "") or "",
        required_facts=list(facts.get("required_facts") or []),
        optional_facts=list(facts.get("optional_facts") or []),
        forbidden_facts=list(facts.get("forbidden_facts") or []),
        evidence_chains=[list(c) for c in (raw.get("evidence_chains") or [])],
        confidence_level=raw.get("confidence_level"),
        annotations=raw.get("annotations") or {},
    )


def load_pack(pack_dir, with_gold=True):
    """加载一个 pack 目录。缺哪个文件就跳过哪一块（A/B/C 三档通吃）。"""
    name = os.path.basename(os.path.normpath(pack_dir))
    domain = re.sub(r"_\d+$", "", name)
    pack = Pack(name=name, domain=domain)

    for fn in sorted(os.listdir(pack_dir)):
        if fn.lower().endswith(".txt"):
            pack.docs[os.path.splitext(fn)[0]] = read_text(os.path.join(pack_dir, fn))

    ev_path = os.path.join(pack_dir, "事件列表.json")
    if os.path.exists(ev_path):
        pack.events = [_parse_event(e) for e in read_json(ev_path)]

    edge_path = os.path.join(pack_dir, "事件因果关系列表.json")
    if os.path.exists(edge_path):
        for raw in read_json(edge_path):
            a, b = _edge_pair(raw)
            if not a or not b:
                continue
            pack.edges.append(Edge(
                cause=a, result=b,
                causal_type=raw.get("causal_type", "") or raw.get("relation_type", ""),
                confidence=str(raw.get("confidence_level", "") or raw.get("confidence", "")),
            ))

    q_path = os.path.join(pack_dir, "问题.json")
    if os.path.exists(q_path):
        pack.questions = [_parse_question(q) for q in read_json(q_path)]

    if with_gold:
        g_path = os.path.join(pack_dir, "gold", "问答对_答案.json")
        if os.path.exists(g_path):
            for raw in read_json(g_path):
                g = _parse_gold(raw)
                pack.gold[g.sample_id] = g

    return pack


def iter_pack_dirs(root):
    """递归收集所有 pack 目录（自身含 问题.json 的目录）。"""
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "问题.json" in filenames:
            out.append(dirpath)
    return sorted(out)


# --------------------------------------------------------------------------- #
# 问题文本解析（通用：实体 ID + 括号内类型提示）
# --------------------------------------------------------------------------- #
def mentions(text):
    """按出现顺序返回去重后的事件 ID 列表。"""
    seen, out = set(), []
    for m in EVENT_ID_RE.findall(text):
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def type_hint(text, event_id):
    """取 'D005（调查报告）' 括号里的类型提示，没有则 None。"""
    for m in re.finditer(re.escape(event_id), text):
        tail = text[m.end():m.end() + 40]
        pm = PAREN_RE.match(tail.lstrip())
        if pm:
            return pm.group(1).strip()
    return None
