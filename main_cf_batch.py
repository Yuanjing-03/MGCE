import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from utils.config import load_config
from data.dataset_loader import load_dataset
from models.han import HANNodeClassifier
from explainer.cf_generator import MACFCounterfactualExplainer
from evaluation.metrics import compute_single_explanation_metrics, aggregate_metrics, metrics_to_dict


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch run counterfactual explanation on selected nodes of the target node type."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--k", type=int, default=10, help="Run on the first-k nodes of target node type")
    parser.add_argument("--preserve", action="store_true", help="Enable stronger non-key preservation")
    parser.add_argument("--device", type=str, default=None, help="Override device from config")
    parser.add_argument(
        "--only_test_nodes",
        action="store_true",
        help="Only evaluate nodes from test split, and still take the first-k among them",
    )
    parser.add_argument(
        "--all_test_nodes",
        action="store_true",
        help="Run explanation for all nodes in the test split (overrides --k and --only_test_nodes)",
    )
    parser.add_argument(
        "--target_json",
        type=str,
        default=None,
        help=(
            "Path to a JSON file containing the exact target node ids to evaluate. "
            "This overrides --k, --only_test_nodes and --all_test_nodes. "
            "Use this to run MACF on the same nodes as HENCE-X."
        ),
    )
    parser.add_argument(
        "--target_json_first_k",
        type=int,
        default=None,
        help="If --target_json is provided, only use the first K ids from that JSON list.",
    )
    parser.add_argument(
        "--skip_invalid_targets",
        action="store_true",
        help="When using --target_json, skip ids outside [0, num_nodes) instead of raising an error.",
    )
    parser.add_argument(
        "--require_test_nodes",
        action="store_true",
        help="When using --target_json, require every id to belong to target_node_type.test_mask.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue to the next node if explanation/evaluation fails for one target.",
    )
    parser.add_argument("--save_json", type=str, default=None, help="Optional path to save per-node metric dicts as JSON")
    return parser.parse_args()


def build_explainer(model, data, loaded, cfg, device: torch.device, preserve: bool):
    explain_cfg = cfg.get("explain", {})

    pred_coeff = explain_cfg.get("lambda_pred", 1.0)
    meta_coeff = explain_cfg.get("lambda_meta", 1.0)
    spar_coeff = explain_cfg.get("lambda_spar", 0.1)

    preserve_coeff = (
        explain_cfg.get("preserve_coeff_on", 0.6)
        if preserve else explain_cfg.get("preserve_coeff_off", 0.0)
    )
    preserve_ratio_floor = explain_cfg.get("preserve_ratio_floor", 0.40)
    edge_preserve_coeff = (
        explain_cfg.get("edge_preserve_coeff_on", 1.0)
        if preserve else explain_cfg.get("edge_preserve_coeff_off", 0.0)
    )
    edge_preserve_floor = explain_cfg.get("edge_preserve_floor", 0.90)

    use_meta_loss = explain_cfg.get("use_meta_loss", True)
    use_keypath_ranking = explain_cfg.get("use_keypath_ranking", True)
    use_key_sparsity = explain_cfg.get("use_key_sparsity", True)
    uniform_edge_perturbation = explain_cfg.get("uniform_edge_perturbation", False)
    semantic_selection = explain_cfg.get("semantic_selection", True)

    explainer = MACFCounterfactualExplainer(
        model=model,
        full_data=data,
        candidate_metapaths=loaded.candidate_metapaths,
        target_node_type=loaded.target_node_type,
        num_hops=cfg["explain"]["num_hops"],
        topk_paths=cfg["explain"].get("topk_paths", 1),
        mask_mode="sigmoid",
        pred_coeff=pred_coeff,
        meta_coeff=meta_coeff,
        preserve_coeff=preserve_coeff,
        preserve_ratio_floor=preserve_ratio_floor,
        edge_preserve_coeff=edge_preserve_coeff,
        edge_preserve_floor=edge_preserve_floor,
        spar_coeff=spar_coeff,
        ent_coeff=explain_cfg.get("lambda_ent", 0.001),
        lr=explain_cfg.get("mask_lr", 0.03),
        epochs=explain_cfg.get("mask_epochs", 120),
        max_instances=explain_cfg.get("max_instances", 1000),
        device=str(device),
        speedup=explain_cfg.get("speedup", False),
        verbose=explain_cfg.get("verbose", False),
        use_meta_loss=use_meta_loss,
        use_keypath_ranking=use_keypath_ranking,
        use_key_sparsity=use_key_sparsity,
        uniform_edge_perturbation=uniform_edge_perturbation,
        semantic_selection=semantic_selection,
    )
    return explainer


def format_float(x, ndigits=4):
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return "nan"


def _safe_metric_mean(metrics_list, attr_name: str):
    vals = []
    for m in metrics_list:
        if not hasattr(m, attr_name):
            continue
        try:
            v = float(getattr(m, attr_name))
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def _safe_list_mean(values):
    vals = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            vals.append(fv)
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))



def _etype_to_str(etype):
    if isinstance(etype, (list, tuple)) and len(etype) == 3:
        return f"{etype[0]}-{etype[1]}-{etype[2]}"
    return str(etype)


def _metapath_to_str(mp):
    if mp is None:
        return ""
    if not isinstance(mp, (list, tuple)):
        return str(mp)
    return " | ".join(_etype_to_str(e) for e in mp)


def _metapath_to_json(mp):
    out = []
    if mp is None:
        return out

    if not isinstance(mp, (list, tuple)):
        return [{"etype": str(mp)}]

    for etype in mp:
        if isinstance(etype, (list, tuple)) and len(etype) == 3:
            out.append({
                "src": str(etype[0]),
                "rel": str(etype[1]),
                "dst": str(etype[2]),
                "etype_str": _etype_to_str(etype),
            })
        else:
            out.append({"etype": str(etype), "etype_str": str(etype)})
    return out


def _safe_float(x, default):
    try:
        if x is None:
            return default
        fx = float(x)
        if not np.isfinite(fx):
            return default
        return fx
    except Exception:
        return default


def _safe_int_list(xs):
    if xs is None:
        return []
    out: List[int] = []
    try:
        iterator = list(xs)
    except Exception:
        return out
    for x in iterator:
        try:
            out.append(int(x))
        except Exception:
            continue
    return out


def _lookup_metapath_score(mp, key_sensitivities):
    if key_sensitivities is None:
        return None

    mp_str_raw = str(mp)
    mp_str_compact = _metapath_to_str(mp)

    if mp_str_raw in key_sensitivities:
        return _safe_float(key_sensitivities[mp_str_raw], default=None)
    if mp_str_compact in key_sensitivities:
        return _safe_float(key_sensitivities[mp_str_compact], default=None)
    return None


def _serialize_key_sensitivities(result):
    out = {}
    key_metapaths = getattr(result, "key_metapaths", []) or []
    key_sensitivities = getattr(result, "key_sensitivities", {}) or {}

    for mp in key_metapaths:
        mp_str = _metapath_to_str(mp)
        out[mp_str] = _lookup_metapath_score(mp, key_sensitivities)

    for k, v in key_sensitivities.items():
        ks = str(k)
        if ks not in out:
            out[ks] = _safe_float(v, default=None)

    return out


def _serialize_edge_index_dict(edge_dict):
    out = {}
    if not isinstance(edge_dict, dict):
        return out

    for etype, idxs in edge_dict.items():
        out[_etype_to_str(etype)] = _safe_int_list(idxs)
    return out


def _serialize_edge_count_dict(edge_dict):
    out: Dict[str, int] = {}
    if not isinstance(edge_dict, dict):
        return out

    for etype, value in edge_dict.items():
        etype_str = _etype_to_str(etype)
        if isinstance(value, (list, tuple, set)):
            out[etype_str] = int(len(value))
        else:
            try:
                out[etype_str] = int(value)
            except Exception:
                out[etype_str] = 0
    return out


def _serialize_ratio_dict(ratio_dict):
    out = {}
    if not isinstance(ratio_dict, dict):
        return out
    for k, v in ratio_dict.items():
        out[str(k)] = _safe_float(v, default=None)
    return out


def build_semantic_info_for_json(result):
    key_metapaths = getattr(result, "key_metapaths", []) or []
    key_sensitivities = getattr(result, "key_sensitivities", {}) or {}

    key_metapath_records = []
    for rank, mp in enumerate(key_metapaths):
        mp_str = _metapath_to_str(mp)
        score = _lookup_metapath_score(mp, key_sensitivities)
        key_metapath_records.append({
            "rank": int(rank),
            "metapath_str": mp_str,
            "metapath": _metapath_to_json(mp),
            "path_len": int(len(mp)) if isinstance(mp, (list, tuple)) else None,
            "joint_score": score,
        })

    key_edge_sets = getattr(result, "key_edge_sets", {}) or {}
    kept_mask = getattr(result, "kept_mask", {}) or {}
    kept_edges = getattr(result, "kept_edges", {}) or {}
    total_edges = getattr(result, "total_edges", {}) or {}

    return {
        "semantic_json_version": 1,

        "key_metapath_strings": [r["metapath_str"] for r in key_metapath_records],
        "key_metapath_records": key_metapath_records,

        "key_metapaths": [_metapath_to_json(mp) for mp in key_metapaths],
        "key_sensitivities": _serialize_key_sensitivities(result),

        "key_edge_sets": _serialize_edge_index_dict(key_edge_sets),
        "key_edge_counts": _serialize_edge_count_dict(key_edge_sets),
        "kept_mask": _serialize_edge_index_dict(kept_mask),
        "kept_mask_counts": _serialize_edge_count_dict(kept_mask),
        "kept_edges_by_type": _serialize_edge_count_dict(kept_edges),
        "total_edges_by_type": _serialize_edge_count_dict(total_edges),
        "key_ratios": _serialize_ratio_dict(getattr(result, "key_ratios", {}) or {}),
        "nonkey_ratios": _serialize_ratio_dict(getattr(result, "nonkey_ratios", {}) or {}),
    }

def _dedupe_preserve_order(ids: List[int]) -> List[int]:
    seen = set()
    out = []
    for x in ids:
        ix = int(x)
        if ix not in seen:
            out.append(ix)
            seen.add(ix)
    return out


def _extract_ids_from_json_obj(obj: Any) -> List[int]:
    if isinstance(obj, dict):
        for key in ["selected_targets", "targets", "target_ids", "node_ids", "selected_node_ids"]:
            if key in obj:
                obj = obj[key]
                break
        else:
            raise ValueError(
                "Unsupported target JSON dict. Expected one of keys: "
                "selected_targets, targets, target_ids, node_ids, selected_node_ids."
            )

    ids = []
    for item in obj:
        if isinstance(item, dict):
            found = False
            for key in ["node_id", "target", "target_id", "id"]:
                if key in item:
                    ids.append(int(item[key]))
                    found = True
                    break
            if not found:
                raise ValueError(f"Unsupported target item dict: {item}")
        else:
            ids.append(int(item))
    return _dedupe_preserve_order(ids)


def load_target_ids_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    ids = _extract_ids_from_json_obj(obj)
    return ids


def select_node_ids(args, data, target_node_type, num_nodes):
    if args.target_json is not None:
        selected_node_ids = load_target_ids_from_json(args.target_json)
        if args.target_json_first_k is not None:
            if args.target_json_first_k <= 0:
                raise ValueError("--target_json_first_k must be > 0")
            selected_node_ids = selected_node_ids[: args.target_json_first_k]

        invalid = [i for i in selected_node_ids if i < 0 or i >= num_nodes]
        if invalid:
            msg = (
                f"Found {len(invalid)} target ids outside valid range [0, {num_nodes}). "
                f"First invalid ids: {invalid[:20]}"
            )
            if args.skip_invalid_targets:
                print(f"[WARN] {msg}; they will be skipped.")
                selected_node_ids = [i for i in selected_node_ids if 0 <= i < num_nodes]
            else:
                raise ValueError(msg + " Use --skip_invalid_targets to skip them.")

        if args.require_test_nodes:
            if not hasattr(data[target_node_type], "test_mask"):
                raise ValueError(f"{target_node_type} has no test_mask, cannot use --require_test_nodes")
            test_ids = set(torch.where(data[target_node_type].test_mask)[0].cpu().tolist())
            not_test = [i for i in selected_node_ids if i not in test_ids]
            if not_test:
                raise ValueError(
                    f"Found {len(not_test)} ids not in test_mask. First not-test ids: {not_test[:20]}"
                )

        if len(selected_node_ids) == 0:
            raise ValueError("No valid selected target ids remain after filtering.")

        return selected_node_ids, f"target_json:{args.target_json}"

    if args.all_test_nodes:
        if not hasattr(data[target_node_type], "test_mask"):
            raise ValueError(f"{target_node_type} has no test_mask")
        selected_node_ids = torch.where(data[target_node_type].test_mask)[0].cpu().tolist()
        return selected_node_ids, "all_test_nodes"

    if args.only_test_nodes:
        if not hasattr(data[target_node_type], "test_mask"):
            raise ValueError(f"{target_node_type} has no test_mask")
        test_ids = torch.where(data[target_node_type].test_mask)[0].cpu().tolist()
        selected_node_ids = test_ids[: args.k]
        return selected_node_ids, f"first_{args.k}_test_nodes"

    selected_node_ids = list(range(min(args.k, num_nodes)))
    return selected_node_ids, f"first_{min(args.k, num_nodes)}_nodes"


def print_header(title: str, width: int = 160):
    print("=" * width)
    print(title)
    print("=" * width)


def print_subheader(title: str, width: int = 160):
    print("-" * width)
    print(title)
    print("-" * width)


def print_run_config(cfg, loaded, device, preserve, selected_node_ids, selection_mode, target_json=None):
    print_header("Batch Counterfactual Explanation")
    print(f"Dataset name        : {loaded.dataset_name}")
    print(f"Target node type    : {loaded.target_node_type}")
    print(f"Num classes         : {loaded.num_classes}")
    print(f"Device              : {device}")
    print(f"Preserve non-key    : {preserve}")
    print(f"Selection mode      : {selection_mode}")
    if target_json is not None:
        print(f"Target JSON         : {target_json}")
    print(f"Selected nodes      : {len(selected_node_ids)}")
    print(f"Preview             : {selected_node_ids[:20]}")
    print(f"Num hops            : {cfg['explain']['num_hops']}")
    print(f"Model ckpt          : checkpoints/han_{loaded.dataset_name}.pt")
    print("=" * 160)


def print_table_header():
    print(
        f"{'node_id':>8} | {'flip':>5} | {'orig->best':>11} | {'p_target':>8} | "
        f"{'p_orig':>8} | {'del_ratio':>9} | {'kpp':>7} | {'odr':>7} | "
        f"{'cf-eff':>8} | {'f-eff':>8} | {'run(s)':>8}"
    )
    print("-" * 160)


def print_node_row(node_id: int, result, metrics, runtime_sec: float):
    orig_cls = result.orig_class
    best_cls = result.best_pred
    flip_text = "Y" if result.flipped else "N"

    p_target = result.best_probs[result.target_class]
    p_orig = result.best_probs[result.orig_class]

    print(
        f"{node_id:>8} | {flip_text:>5} | {orig_cls}->{best_cls:<8} | "
        f"{format_float(p_target):>8} | {format_float(p_orig):>8} | "
        f"{format_float(metrics.delete_ratio):>9} | {format_float(metrics.kpp):>7} | "
        f"{format_float(metrics.odr):>7} | {format_float(metrics.cf_effect):>8} | "
        f"{format_float(metrics.f_effect):>8} | {format_float(runtime_sec):>8}"
    )

def print_batch_summary(agg: dict):
    print_subheader("Batch Aggregate Metrics")
    keys_in_order = [
        "mean_flip_success",
        "mean_orig_prob_drop",
        "mean_f_effect",
        "mean_delete_ratio",
        "mean_kpp",
        "mean_odr",
        "mean_runtime_sec_per_node",
    ]
    for key in keys_in_order:
        if key in agg:
            print(f"{key:<36}: {agg[key]:.6f}")


def main():
    args = parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device_str = args.device if args.device is not None else cfg.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, fallback to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    loaded = load_dataset(cfg)
    data = loaded.data
    target_node_type = loaded.target_node_type
    num_classes = loaded.num_classes

    model = HANNodeClassifier(
        data=data,
        target_node_type=target_node_type,
        hidden_dim=cfg["model"]["hidden_dim"],
        out_dim=num_classes,
        heads=cfg["model"].get("heads", 8),
        dropout=cfg["model"].get("dropout", 0.5),
    )

    ckpt_path = f"checkpoints/han_{loaded.dataset_name}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    explainer = build_explainer(
        model=model,
        data=data,
        loaded=loaded,
        cfg=cfg,
        device=device,
        preserve=args.preserve,
    )

    num_nodes = data[target_node_type].num_nodes
    selected_node_ids, selection_mode = select_node_ids(
        args=args,
        data=data,
        target_node_type=target_node_type,
        num_nodes=num_nodes,
    )

    print_run_config(
        cfg,
        loaded,
        device,
        args.preserve,
        selected_node_ids=selected_node_ids,
        selection_mode=selection_mode,
        target_json=args.target_json,
    )

    metrics_list = []
    json_rows = []
    runtime_secs = []
    failed_rows = []

    print_subheader("Per-node Results")
    print_table_header()

    program_t0 = time.perf_counter()

    for idx, node_id in enumerate(selected_node_ids, start=1):
        print(f"[INFO] Running node {node_id} ({idx}/{len(selected_node_ids)})")
        node_t0 = time.perf_counter()

        try:
            result = explainer.explain(target_node_id=node_id)

            runtime_sec = time.perf_counter() - node_t0
            runtime_secs.append(runtime_sec)

            metrics = compute_single_explanation_metrics(
                orig_probs=result.orig_probs,
                cf_probs=result.best_probs,
                orig_class=result.orig_class,
                target_class=result.target_class,
                key_ratios=result.key_ratios,
                nonkey_ratios=result.nonkey_ratios,
                key_metapaths=result.key_metapaths,
                key_edge_sets=result.key_edge_sets,
                kept_mask=result.kept_mask,
                fidelity_plus=result.fidelity_plus,
                fidelity_minus=result.fidelity_minus,
                kept_edges=result.kept_edges,
                total_edges=result.total_edges,
                best_pred=result.best_pred,
                soft_keep_ratio=result.soft_keep_ratio,
            )

            metrics_list.append(metrics)
            print_node_row(node_id=node_id, result=result, metrics=metrics, runtime_sec=runtime_sec)

            semantic_info = build_semantic_info_for_json(result)

            row = {
                "node_id": int(node_id),
                "orig_class": int(result.orig_class),
                "target_class": int(result.target_class),
                "best_pred": int(result.best_pred),
                "flipped": bool(result.flipped),
                "runtime_sec": float(runtime_sec),
                **metrics_to_dict(metrics),

                # New: keep semantic/metapath information for visualization.
                **semantic_info,
            }
            json_rows.append(row)

        except Exception as e:
            runtime_sec = time.perf_counter() - node_t0
            msg = f"[ERROR] node {node_id} failed after {runtime_sec:.4f}s: {repr(e)}"
            print(msg)
            failed_rows.append({"node_id": int(node_id), "runtime_sec": float(runtime_sec), "error": repr(e)})
            if not args.continue_on_error:
                raise
            continue

    total_runtime_sec = time.perf_counter() - program_t0

    agg = aggregate_metrics(metrics_list)
    agg["mean_cf_effect"] = _safe_metric_mean(metrics_list, "cf_effect")
    agg["mean_f_effect"] = _safe_metric_mean(metrics_list, "f_effect")
    agg["mean_runtime_sec_per_node"] = _safe_list_mean(runtime_secs)
    agg["total_runtime_sec"] = float(total_runtime_sec)
    agg["num_nodes_selected"] = int(len(selected_node_ids))
    agg["num_nodes_evaluated"] = int(len(json_rows))
    agg["num_nodes_failed"] = int(len(failed_rows))

    print_batch_summary(agg)

    print_subheader("Runtime Summary")
    print(f"{'num_nodes_selected':<36}: {len(selected_node_ids)}")
    print(f"{'num_nodes_evaluated':<36}: {len(json_rows)}")
    print(f"{'num_nodes_failed':<36}: {len(failed_rows)}")
    print(f"{'mean_runtime_sec_per_node':<36}: {agg['mean_runtime_sec_per_node']:.6f}")
    print(f"{'total_runtime_sec':<36}: {agg['total_runtime_sec']:.6f}")

    print_subheader("Notes")
    print("cf_effect : probability drop of original class under the FINAL kept counterfactual graph (same convention as orig_prob_drop)")
    print("f_effect  : original-class score on the DELETED-EDGE explanation subgraph (Scheme B)")
    print("kpp       : deleted-edge proportion covered by selected key meta-path instances; higher is better")
    print("odr       : deleted-edge proportion outside selected key meta-path instances; lower is better")
    print("target_json mode runs MACF on exactly the same node ids selected by HENCE-X, after optional validation/filtering.")
    print("=" * 160)

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "selection_mode": selection_mode,
                    "target_json": args.target_json,
                    "selected_node_ids": [int(x) for x in selected_node_ids],
                    "semantic_json_version": 1,
                    "semantic_fields": [
                        "key_metapath_strings",
                        "key_metapath_records",
                        "key_sensitivities",
                        "key_edge_sets",
                        "key_edge_counts",
                        "kept_mask",
                        "kept_mask_counts",
                        "key_ratios",
                        "nonkey_ratios",
                    ],
                    "per_node": json_rows,
                    "failed": failed_rows,
                    "aggregate": agg,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[OK] Saved metric json to: {args.save_json}")


if __name__ == "__main__":
    main()
