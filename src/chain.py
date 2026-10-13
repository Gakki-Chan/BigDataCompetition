# -*- coding: utf-8 -*-
"""链解码层：候选路径生成（通用图算法）+ 候选特征（供学习式排序器使用）。

设计要点（这是"通用、不写规则"的关键）：
  * 生成器**不按意图分支**决定"该生成什么形状"，而是把一组通用形状全部枚举出来
    （通向锚点的路径 / 从锚点出发的路径 / 锚点之间的路径 / 长度 2 的直连边 /
     单节点 / 最新锚点之后的序列 / 空链），由排序器根据学习到的特征选形状。
  * 长度上限、候选上限都是赛题层面的常量，不含领域知识。
  * 特征分两类：与问题文本的相关性 + 通用图统计量。
"""
from __future__ import annotations

from collections import defaultdict

from scorer import containment, cosine
from schema import EVENT_ID_RE, PAREN_RE, mentions, type_hint

MAX_PATH_LEN = 8          # 证据链跳数上限（训练集里最长见过 8 节点）
MAX_SEQ_LEN = 14          # "序列型"候选的长度上限（整段事件序列可能很长）
PER_FAMILY_CAP = 60       # 每个"形状族"的候选上限，保证各形状都有代表
MAX_CANDIDATES = 260


# --------------------------------------------------------------------------- #
# 图
# --------------------------------------------------------------------------- #
class Graph:
    def __init__(self, node_ids, edge_scores):
        self.nodes = list(node_ids)
        self.adj = defaultdict(list)
        self.rev = defaultdict(list)
        for (a, b), _w in edge_scores.items():
            self.adj[a].append(b)
            self.rev[b].append(a)
        self.edge_scores = edge_scores

    def successors(self, n):
        return list(self.adj.get(n, []))

    def predecessors(self, n):
        return list(self.rev.get(n, []))

    def edge_score(self, a, b):
        return self.edge_scores.get((a, b), 0.0)

    def paths_between(self, s, t, max_len=MAX_PATH_LEN, limit=40):
        if s == t:
            return [[s]]
        out, queue = [], [(s, [s])]
        while queue and len(out) < limit:
            node, path = queue.pop(0)
            if len(path) >= max_len:
                continue
            for nxt in self.adj.get(node, []):
                if nxt in path:
                    continue
                np_ = path + [nxt]
                if nxt == t:
                    out.append(np_)
                else:
                    queue.append((nxt, np_))
        out.sort(key=len)
        return out

    def paths_ending_at(self, t, max_len=MAX_PATH_LEN, limit=60):
        out, stack = [], [(t, [t])]
        while stack and len(out) < limit:
            node, path = stack.pop()
            preds = [p for p in self.rev.get(node, []) if p not in path]
            if not preds or len(path) >= max_len:
                if len(path) > 1:
                    out.append(list(reversed(path)))
                continue
            for p in preds:
                stack.append((p, path + [p]))
        out.sort(key=len, reverse=True)
        return out

    def paths_starting_at(self, s, max_len=MAX_PATH_LEN, limit=60):
        out, queue = [], [(s, [s])]
        while queue and len(out) < limit:
            node, path = queue.pop(0)
            if len(path) > 1:
                out.append(path)
            if len(path) >= max_len:
                continue
            for nxt in self.adj.get(node, []):
                if nxt not in path:
                    queue.append((nxt, path + [nxt]))
        out.sort(key=len, reverse=True)
        return out

    def longest_paths_from_roots(self, limit=40):
        roots = [n for n in self.nodes if not self.rev.get(n)]
        out = []
        for r in roots:
            out.extend(self.paths_starting_at(r, limit=20))
        out.sort(key=len, reverse=True)
        return out[:limit]


def build_graph(pack, edge_scores=None):
    if edge_scores is None:
        edge_scores = {(e.cause, e.result): 1.0 for e in pack.edges}
    return Graph(pack.event_ids, edge_scores)


# --------------------------------------------------------------------------- #
# 预计算上下文（推理速度关键：每个 pack 只算一次）
# --------------------------------------------------------------------------- #
class PackContext:
    def __init__(self, pack):
        self.pack = pack
        self.ids = list(pack.event_ids)
        self.idx = pack.event_index
        self.n = max(1, len(self.ids))
        self.ev_of = {e.event_id: e for e in pack.events}
        self.short = {e.event_id: "%s%s%s" % (
            e.event_type, e.trigger, (e.args.get("内容") or "")[:60]) for e in pack.events}

    def serialize(self, chain):
        return " ".join(self.short.get(c, "") for c in chain if c in self.short)

    def type_of(self, eid):
        ev = self.ev_of.get(eid)
        return ev.event_type if ev else ""


def anchors_of(pack, question):
    if IGNORE_ANCHORS:
        return []
    if LINKER is not None:
        from linking import combined_anchors
        return combined_anchors(pack, question, LINKER, LINKER_THRESHOLD, LINKER_TOPK)
    node_set = set(pack.event_ids)
    return [m for m in mentions(question.text) if m in node_set]


# --------------------------------------------------------------------------- #
# 候选生成：通用形状族
# --------------------------------------------------------------------------- #
def _dedup(cands):
    seen, out = set(), []
    for c in cands:
        key = tuple(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(list(c))
    return out


def generate_candidates(pack, question, graph, intent=None, max_candidates=MAX_CANDIDATES,
                        ctx=None):
    """枚举通用形状的候选链。空链 [] 始终保留（对应"无路径/拒答"）。"""
    ctx = ctx or PackContext(pack)
    anchors = anchors_of(pack, question)
    cands = [[]]

    def add(c, seq=False):
        limit = MAX_SEQ_LEN if seq else MAX_PATH_LEN
        if c and 1 <= len(c) <= limit and all(x in ctx.idx for x in c):
            cands.append(c)

    # 族 1：锚点之间的路径（双向）
    if len(anchors) >= 2:
        for s, t in ((anchors[0], anchors[1]), (anchors[1], anchors[0])):
            for p in graph.paths_between(s, t, limit=PER_FAMILY_CAP):
                add(p)

    # 族 2：以锚点结尾的路径
    for a in anchors[:2]:
        for p in graph.paths_ending_at(a, limit=PER_FAMILY_CAP * 2):
            add(p)

    # 族 3：从锚点出发的路径
    for a in anchors[:2]:
        for p in graph.paths_starting_at(a, limit=PER_FAMILY_CAP):
            add(p)

    # 族 4：长度 2 的直连边（含锚点优先，其次全图）
    edges2 = []
    for a in anchors:
        for b in graph.successors(a):
            edges2.append([a, b])
    if len(edges2) < PER_FAMILY_CAP:
        for a in ctx.ids:
            for b in graph.successors(a):
                edges2.append([a, b])
            if len(edges2) >= PER_FAMILY_CAP * 2:
                break
    for c in edges2[:PER_FAMILY_CAP * 2]:
        add(c)

    # 族 5：单节点（锚点优先，补足到一定数量）
    for a in anchors:
        add([a])
    for n in ctx.ids[:30]:
        add([n])

    # 族 6：最新锚点之后的序列（前瞻型问题常见形状；纯位置操作，无领域知识）
    if anchors:
        cut = max(ctx.idx.get(a, 0) for a in anchors)
        tail = [eid for eid in ctx.ids if ctx.idx[eid] > cut]
        if tail:
            add(tail[:MAX_PATH_LEN])
            add(tail)
        # 最后一个可见事件 → 第一个后续事件
        if tail:
            last_anchor = max(anchors, key=lambda a: ctx.idx.get(a, 0))
            add([last_anchor, tail[0]])
    else:
        tail = ctx.ids[1:]
        if tail:
            add(tail[:MAX_PATH_LEN])

    # 族 7：从根节点出发的最长路径（主干链）
    for p in graph.longest_paths_from_roots(limit=20):
        add(p)

    # 族 8：整图的全部短路径（兜底，数量受限）
    if len(cands) < 20:
        for s in ctx.ids:
            for p in graph.paths_starting_at(s, limit=8):
                add(p)

    # 族 9：按"问题里提到的顺序"排列的序列（前瞻型问题的金标常是这个形状）
    if anchors:
        add(list(anchors))
        if len(anchors) > 1:
            add(list(reversed(anchors)))
        # 锚点之间的所有有序对（文本顺序）
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                add([anchors[i], anchors[j]])
        # 锚点（按问题顺序）+ 其后的事件序列
        last = max(ctx.idx.get(a, 0) for a in anchors)
        tail = [eid for eid in ctx.ids if ctx.idx[eid] > last]
        if tail:
            add(list(anchors) + tail, seq=True)
            add(list(anchors) + tail[:MAX_PATH_LEN])

    # 族 10：从最早锚点到最晚锚点之间、按文档顺序排列的序列
    if anchors:
        lo = min(ctx.idx.get(a, 0) for a in anchors)
        hi = max(ctx.idx.get(a, 0) for a in anchors)
        span = ctx.ids[lo:hi + 1]
        add(span, seq=True)
        add(span[:MAX_PATH_LEN])

    # 族 11：按文档顺序排列的整段事件序列（含锚点之后的部分）
    if anchors:
        lo = min(ctx.idx.get(a, 0) for a in anchors)
        add(ctx.ids[lo:], seq=True)
        add(ctx.ids[lo:lo + MAX_PATH_LEN])
    add(ctx.ids[:MAX_PATH_LEN])
    add(ctx.ids, seq=True)
    for k in range(1, min(len(ctx.ids), MAX_SEQ_LEN) + 1):
        add(ctx.ids[:k], seq=True)

    cands = _dedup(cands)
    if len(cands) > max_candidates:
        anchor_set = set(anchors)
        cands.sort(key=lambda c: (-len(anchor_set & set(c)), len(c) if c else 0))
        cands = cands[:max_candidates]
    return cands


# --------------------------------------------------------------------------- #
# 候选特征
# --------------------------------------------------------------------------- #
def candidate_features(pack, question, chain, graph, intent=None, anchor_ids=None,
                       ctx=None):
    ctx = ctx or PackContext(pack)
    qtext = question.text
    chain = list(chain)
    n = len(chain)
    anchors = list(anchor_ids) if anchor_ids is not None else anchors_of(pack, question)
    anchor_set = set(anchors)
    ser = ctx.serialize(chain)

    f = []
    # --- 形状 / 长度 ---
    f.append(1.0 if n == 0 else 0.0)
    f.append(float(min(n, MAX_PATH_LEN)))
    f.append(1.0 if n == 1 else 0.0)
    f.append(1.0 if n == 2 else 0.0)
    f.append(1.0 if n >= 3 else 0.0)
    # --- 与问题的文本相关性 ---
    f.append(cosine(qtext, ser) if ser else 0.0)
    f.append(containment(ser, qtext, 2) if ser else 0.0)
    # --- 锚点覆盖 / 位置 ---
    hit = len(anchor_set & set(chain))
    f.append(hit / len(anchors) if anchors else 0.0)
    f.append(1.0 if anchors and chain and chain[0] == anchors[0] else 0.0)
    f.append(1.0 if anchors and chain and chain[-1] == anchors[-1] else 0.0)
    f.append(1.0 if anchors and chain and chain[0] == anchors[-1] else 0.0)
    f.append(1.0 if anchors and chain and chain[-1] == anchors[0] else 0.0)
    f.append(1.0 if anchors and chain and chain[0] in anchor_set else 0.0)
    f.append(1.0 if anchors and chain and chain[-1] in anchor_set else 0.0)
    # 锚点在链中的相对位置（越靠头/尾越说明形状）
    if chain and anchors:
        pos = [i for i, c in enumerate(chain) if c in anchor_set]
        f.append(pos[0] / n if pos else 0.0)
        f.append(pos[-1] / n if pos else 0.0)
    else:
        f.append(0.0)
        f.append(0.0)
    # --- 括号里的类型提示是否与节点类型一致 ---
    hint_ok, hint_total = 0.0, 0
    for a in anchors:
        h = type_hint(qtext, a)
        et = ctx.type_of(a)
        if h and et:
            hint_total += 1
            if containment(h, et, 2) >= 0.5 or containment(et, h, 2) >= 0.5:
                hint_ok += 1.0
    f.append(hint_ok / hint_total if hint_total else 0.0)
    # --- 边质量 / 时间与编号单调 ---
    if n >= 2:
        scores = [graph.edge_score(a, b) for a, b in zip(chain, chain[1:])]
        f.append(min(scores))
        f.append(sum(scores) / len(scores))
        times = [ctx.ev_of[c].time for c in chain if c in ctx.ev_of]
        mono = 1.0
        for x, y in zip(times, times[1:]):
            if x and y and x > y:
                mono = 0.0
                break
        f.append(mono)
        pos = [ctx.idx.get(c, 0) for c in chain]
        f.append(1.0 if all(b >= a for a, b in zip(pos, pos[1:])) else 0.0)
        f.append(float(pos[-1] - pos[0]))
    else:
        f.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    # --- 枢纽度 ---
    if chain:
        deg = [len(graph.successors(c)) + len(graph.predecessors(c)) for c in chain]
        f.append(sum(deg) / len(deg) / ctx.n)
    else:
        f.append(0.0)
    # --- 未在问题中出现的节点占比 ---
    f.append(1.0 - (hit / n) if n else 1.0)
    # --- 与"最新锚点"的相对位置（前瞻/回溯的分界） ---
    if anchors:
        last = max(ctx.idx.get(a, 0) for a in anchors)
        if chain:
            after = sum(1 for c in chain if ctx.idx.get(c, 0) > last)
            f.append(after / n)
            f.append(1.0 if after == n else 0.0)
        else:
            f.append(0.0)
            f.append(0.0)
        f.append(float(ctx.idx.get(chain[0], 0) - last) / ctx.n if chain else 0.0)
    else:
        f.extend([0.0, 0.0, 0.0])
    # --- 是否覆盖首事件 / 末事件 ---
    f.append(1.0 if chain and chain[0] == ctx.ids[0] else 0.0)
    f.append(1.0 if chain and chain[-1] == ctx.ids[-1] else 0.0)
    return mask_disabled(f)


FEATURE_NAMES = [
    "is_empty", "length", "is_singleton", "is_edge", "is_long",
    "q_chain_cos", "q_chain_cover",
    "anchor_cov", "head_first_anchor", "tail_last_anchor", "head_last_anchor",
    "tail_first_anchor", "head_in_anchors", "tail_in_anchors",
    "anchor_first_pos", "anchor_last_pos", "type_hint_ok",
    "edge_min", "edge_mean", "time_mono", "idx_mono", "idx_span",
    "hub_degree", "novel_ratio",
    "after_last_anchor_ratio", "all_after_last_anchor", "head_vs_last_anchor",
    "head_is_first_event", "tail_is_last_event",
]

# 消融用：置零某组特征（不改向量长度，训练/推理两侧一致）
DISABLED_FEATURES = set()
# 消融用：假装"问题里没有点名任何事件 ID"（真实语料的常态）
IGNORE_ANCHORS = False

# 事件提及链接器（训练脚本注入）。为 None 时退化为"只认问题里出现的 D-id"。
LINKER = None
LINKER_THRESHOLD = 0.5
LINKER_TOPK = 3

# 依赖"文档顺序/编号"的特征组——它们很大程度上是数据生成器的结构性痕迹
ORDER_FEATURES = {
    "idx_mono", "idx_span", "after_last_anchor_ratio", "all_after_last_anchor",
    "head_vs_last_anchor", "head_is_first_event", "tail_is_last_event",
    "time_mono",
}
# 依赖"问题里直接出现事件 ID"的特征组——真实语料里问题通常不会点名事件 ID
ANCHOR_FEATURES = {
    "anchor_cov", "head_first_anchor", "tail_last_anchor", "head_last_anchor",
    "tail_first_anchor", "head_in_anchors", "tail_in_anchors",
    "anchor_first_pos", "anchor_last_pos", "type_hint_ok",
    "after_last_anchor_ratio", "all_after_last_anchor", "head_vs_last_anchor",
}

_FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def mask_disabled(features):
    for name in DISABLED_FEATURES:
        i = _FEATURE_INDEX.get(name)
        if i is not None and i < len(features):
            features[i] = 0.0
    return features
