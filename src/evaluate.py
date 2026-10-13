# -*- coding: utf-8 -*-
"""在 dev 折上评估 L0 流水线，并同时给出"金标上界"用于归一化。

用法：
    python src/evaluate.py --bundle artifacts/l0/bundle.joblib --dev-frac 0.15
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import Pipeline
from schema import load_pack
from scorer import REFUSE_TEXT, Scorer
from train import TRAIN_DIR, load_all, stratified_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CONF_MAP = {"certain": 0.9, "probable": 0.7, "possible": 0.5}


def oracle_pred(gold):
    if gold.is_unanswerable:
        return {"answer": REFUSE_TEXT, "evidence_chain": [], "confidence": None}
    return {"answer": gold.answers,
            "evidence_chain": gold.evidence_chains[0] if gold.evidence_chains else [],
            "confidence": CONF_MAP.get(gold.confidence_level or "", 0.7)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.path.join(ROOT, "artifacts", "l0", "bundle.joblib"))
    ap.add_argument("--dev-frac", type=float, default=0.15)
    ap.add_argument("--split", default="dev", choices=["dev", "train", "test"])
    ap.add_argument("--test-root", default=os.path.join(ROOT, "赛题4", "数据集", "抽样测试集_100"))
    ap.add_argument("--sim-bc", type=float, default=0.5)
    ap.add_argument("--dump", default="", help="把预测写成提交 JSON 的路径")
    ap.add_argument("--cache", default="", help="预测缓存路径（存在则直接复用，跳过推理）")
    ap.add_argument("--strip-edges", action="store_true",
                    help="模拟任务B：抽掉给定因果边，只用事件列表 + 预测边建图")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--abstain-threshold", type=float, default=-1.0)
    args = ap.parse_args()

    bundle = joblib.load(args.bundle)
    pipe = Pipeline(bundle["intent"], bundle["abstain"], bundle["ranker"], bundle["edge"],
                    bundle["confidence"], bundle["answers"])
    cfg = (bundle.get("meta") or {}).get("config") or {}
    if cfg:
        pipe.abstain_threshold = cfg.get("abstain_threshold", pipe.abstain_threshold)
        pipe.confidence_prior = cfg.get("confidence_prior", pipe.confidence_prior)
        pipe.confidence_shrink = cfg.get("confidence_shrink", pipe.confidence_shrink)
    if args.abstain_threshold >= 0:
        pipe.abstain_threshold = args.abstain_threshold
    scorer = Scorer()

    # ---- 选评估集 ----
    if args.split == "test":
        from schema import iter_pack_dirs
        packs = [load_pack(d, with_gold=False) for d in iter_pack_dirs(args.test_root)]
    else:
        packs = load_all()
        if args.limit:
            packs = packs[:args.limit]
        tr, dev = stratified_split(packs, args.dev_frac)
        packs = dev if args.split == "dev" else tr
        if args.split == "dev":
            keep = set(bundle["meta"]["dev_packs"])
            if keep:
                packs = [p for p in packs if p.name in keep]
    print("评估 pack 数（%s）: %d" % (args.split, len(packs)))

    cache = args.cache
    if cache and os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            cached = json.load(f)
        preds = cached["preds"]
        print("复用预测缓存：%s（%d 题）" % (cache, len(preds)))
    else:
        t0 = time.time()
        preds = []
        for pack in packs:
            if args.strip_edges:
                pack.edges = []
            preds.extend(pipe.predict_pack(pack))
        dt = time.time() - t0
        print("推理 %.1fs（%.2f 题/秒，共 %d 题）"
              % (dt, len(preds) / max(1e-9, dt), len(preds)))
        if cache:
            os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
            with open(cache, "w", encoding="utf-8") as f:
                json.dump({"preds": preds}, f, ensure_ascii=False)

    by_id = {p["sample_id"]: p for p in preds}
    rows, oracle_rows = [], []
    for pack in packs:
        for q in pack.questions:
            g = pack.gold.get(q.sample_id)
            if g is None:
                continue
            pred = by_id.get(q.sample_id)
            if pred is None:
                continue
            rows.append(dict({"tier": pack.tier}, **scorer.score_one(pred, g)))
            oracle_rows.append(dict({"tier": pack.tier}, **scorer.score_one(oracle_pred(g), g)))

    if args.dump:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump)), exist_ok=True)
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump([{k: v for k, v in p.items() if not k.startswith("_")} for p in preds],
                      f, ensure_ascii=False, indent=1)
        print("已写出：%s" % args.dump)

    if not rows:
        return
    model_agg = scorer.aggregate(rows)
    oracle_agg = scorer.aggregate(oracle_rows)
    print("\n%-6s %8s %8s %8s %8s | %8s %8s %8s %8s"
          % ("tier", "答案", "证据链", "置信度", "总分", "oracle答", "oracle链", "oracle总", "相对"))
    for tier in ("A", "B", "C", "ALL"):
        if "%s_total" % tier not in model_agg:
            continue
        mt = model_agg["%s_total" % tier]
        ot = oracle_agg.get("%s_total" % tier, 0.0)
        print("%-6s %8.2f %8.2f %8.2f %8.2f | %8.2f %8.2f %8.2f %8.1f%%"
              % (tier, model_agg["%s_answer" % tier], model_agg["%s_chain" % tier],
                 model_agg["%s_confidence" % tier], mt,
                 oracle_agg.get("%s_answer" % tier, 0), oracle_agg.get("%s_chain" % tier, 0),
                 ot, 100.0 * mt / ot if ot else 0.0))
    print("\n加权总分 %.2f ／ oracle 上界 %.2f  →  相对 %.1f%%"
          % (model_agg["weighted"], oracle_agg["weighted"],
             100.0 * model_agg["weighted"] / max(1e-9, oracle_agg["weighted"])))


if __name__ == "__main__":
    main()
