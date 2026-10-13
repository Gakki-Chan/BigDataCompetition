# -*- coding: utf-8 -*-
"""候选链的监督标签：硬标签（是否命中金标链）与软标签（与金标链的对齐程度）。"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from scorer import lcs_len


def hard_label(candidate, gold_chains: List[Sequence[str]]) -> int:
    if not gold_chains:
        return 1 if not candidate else 0
    return 1 if tuple(candidate) in {tuple(c) for c in gold_chains} else 0


def soft_label(candidate, gold_chains: List[Sequence[str]]) -> float:
    """与评测口径一致的连续目标：0.625·节点F1 + 0.375·顺序分。"""
    cand = list(candidate)
    if not gold_chains:
        return 1.0 if not cand else 0.0
    best = 0.0
    for gold in gold_chains:
        gold = list(gold)
        if not cand and not gold:
            return 1.0
        inter = len(set(cand) & set(gold))
        precision = inter / len(cand) if cand else 0.0
        recall = inter / len(gold) if gold else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        if cand and cand == gold:
            order = 1.0
        else:
            order = 0.5 * (lcs_len(cand, gold) / max(1, len(gold)))
        best = max(best, 0.625 * f1 + 0.375 * order)
    return best
