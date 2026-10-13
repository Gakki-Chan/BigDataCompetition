# -*- coding: utf-8 -*-
"""训练 L0 流水线的全部可训练模块，并把模型存到 artifacts/。

用法：
    python src/train.py --dev-frac 0.15 --out artifacts/l0
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from answer import AnswerBank
from chain import (PackContext, anchors_of, build_graph, candidate_features,
                   generate_candidates)
from models import (AbstainModel, ConfidenceModel, EdgeModel, IntentModel,
                    MoEChainRanker)
from pipeline import meta_features
from schema import iter_pack_dirs, load_pack

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TRAIN_DIR = os.path.join(ROOT, "赛题4", "数据集", "训练集")


def stratified_split(packs, dev_frac, seed=0):
    """按领域分层切分，保证每个领域在 dev 里都有样本。"""
    rnd = random.Random(seed)
    by_domain = {}
    for p in packs:
        by_domain.setdefault(p.domain, []).append(p)
    dev, tr = [], []
    for dom, group in sorted(by_domain.items()):
        rnd.shuffle(group)
        n_dev = max(1, int(round(len(group) * dev_frac)))
        dev.extend(group[:n_dev])
        tr.extend(group[n_dev:])
    return tr, dev


def load_all(root=TRAIN_DIR):
    t0 = time.time()
    packs = [load_pack(d) for d in iter_pack_dirs(root)]
    packs = [p for p in packs if p.questions]
    print("载入 %d 个 pack（%.1fs）" % (len(packs), time.time() - t0))
    return packs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-frac", type=float, default=0.15)
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "l0"))
    ap.add_argument("--sim-bc", type=float, default=0.5,
                    help="训练链排序器时把该比例 pack 当成 B/C 档（用预测边建图）")
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 个 pack 快速试跑")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    packs = load_all()
    if args.limit:
        packs = packs[:args.limit]
    train_packs, dev_packs = stratified_split(packs, args.dev_frac)
    print("训练 %d / 验证 %d" % (len(train_packs), len(dev_packs)))

    # ---------------- 1. 意图分类 ----------------
    t0 = time.time()
    intent = IntentModel()
    intent.fit([q.text for p in train_packs for q in p.questions],
               [q.reasoning_type for p in train_packs for q in p.questions])
    hit = sum(1 for p in dev_packs for q in p.questions
              if intent.predict([q.text])[0] == q.reasoning_type)
    total = max(1, sum(len(p.questions) for p in dev_packs))
    print("意图分类：dev acc=%.4f（%.1fs）" % (hit / total, time.time() - t0))

    # ---------------- 2. 拒答判定 ----------------
    t0 = time.time()
    abstain = AbstainModel()
    abstain.fit([q.text for p in train_packs for q in p.questions
                 if q.sample_id in p.gold],
                [1 if p.gold[q.sample_id].is_unanswerable else 0
                 for p in train_packs for q in p.questions if q.sample_id in p.gold])
    tp = fp = fn = 0
    for p in dev_packs:
        for q in p.questions:
            g = p.gold.get(q.sample_id)
            if g is None:
                continue
            pred = abstain.predict([q.text])[0]
            truth = 1 if g.is_unanswerable else 0
            tp += int(pred == 1 and truth == 1)
            fp += int(pred == 1 and truth == 0)
            fn += int(pred == 0 and truth == 1)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    print("拒答判定：P=%.3f R=%.3f F1=%.3f（%.1fs）"
          % (prec, rec, 2 * prec * rec / max(1e-9, prec + rec), time.time() - t0))

    # ---------------- 3. 因果边模型（B/C 档用） ----------------
    t0 = time.time()
    edge = EdgeModel().fit_priors(train_packs).fit(train_packs)
    print("因果边模型训练完成（%.1fs）" % (time.time() - t0))

    # ---------------- 4. 链排序器 ----------------
    t0 = time.time()
    rnd = random.Random(1)
    X, y, intents_of_rows = [], [], []
    for pack in train_packs:
        use_pred = rnd.random() < args.sim_bc
        scores = edge.predict_scores(pack) if use_pred else None
        graph = build_graph(pack, scores)
        ctx = PackContext(pack)
        for q in pack.questions:
            g = pack.gold.get(q.sample_id)
            if g is None:
                continue
            intent_lab = intent.predict([q.text])[0]
            anchors = anchors_of(pack, q)
            golds = [tuple(c) for c in g.evidence_chains] or [()]
            for c in generate_candidates(pack, q, graph, intent_lab, ctx=ctx):
                X.append(candidate_features(pack, q, c, graph, anchor_ids=anchors, ctx=ctx))
                y.append(1 if tuple(c) in golds else 0)
                intents_of_rows.append(intent_lab)
    print("链排序训练样本 %d 条（正例 %.3f，%.1fs）"
          % (len(y), sum(y) / max(1, len(y)), time.time() - t0))
    t0 = time.time()
    ranker = MoEChainRanker()
    if X:
        ranker.fit(X, y, intents_of_rows)
    print("链排序器训练完成：全局 1 个 + 分意图 %d 个（%.1fs）"
          % (len(ranker.rankers), time.time() - t0))

    # ---------------- 5. 置信度模型 ----------------
    t0 = time.time()
    Xc, yc = [], []
    for pack in train_packs:
        graph = build_graph(pack)
        ctx = PackContext(pack)
        for q in pack.questions:
            g = pack.gold.get(q.sample_id)
            if g is None or g.confidence_level not in ConfidenceModel.VALUES:
                continue
            chain = g.evidence_chains[0] if g.evidence_chains else []
            anchors = anchors_of(pack, q)
            Xc.append(meta_features(pack, q, chain, graph, q.reasoning_type, anchors,
                                    float(g.is_unanswerable), ctx=ctx))
            yc.append(g.confidence_level)
    confidence = ConfidenceModel()
    if Xc:
        confidence.fit(Xc, yc)
    print("置信度模型：%d 条训练完成（%.1fs）" % (len(Xc), time.time() - t0))

    # ---------------- 6. 答案库（检索式组装） ----------------
    t0 = time.time()
    bank = AnswerBank().fit(train_packs)
    print("答案库 %d 条（%.1fs）" % (len(bank.entries), time.time() - t0))

    # ---------------- 保存 ----------------
    bundle = {"intent": intent, "abstain": abstain, "ranker": ranker, "edge": edge,
              "confidence": confidence, "answers": bank,
              "meta": {"dev_packs": [p.name for p in dev_packs],
                       "n_train_packs": len(train_packs),
                       "config": {"abstain_threshold": 0.7, "confidence_prior": 0.7,
                                  "confidence_shrink": 0.75,
                                  "max_path_len": 8, "max_candidates": 260}}}
    path = os.path.join(args.out, "bundle.joblib")
    joblib.dump(bundle, path, compress=3)
    print("已保存：%s（%.1f MB）" % (path, os.path.getsize(path) / 1e6))


if __name__ == "__main__":
    main()
