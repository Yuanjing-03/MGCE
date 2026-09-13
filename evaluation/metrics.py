from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

EdgeType = Tuple[str, str, str]


@dataclass
class SingleExplanationMetrics:
    flip_success: float         
    orig_prob_drop: float        
    target_prob_gain: float      
    num_edges_total: int
    num_edges_kept: int
    num_edges_removed: int
    keep_ratio: float
    delete_ratio: float          
    kpp: float                   
    odr: float                   
    fidelity_plus: float
    fidelity_minus: float
    cf_effect: float             
    f_effect: float             

def _safe_mean(vals, default):
    cleaned = []
    for v in vals:
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isnan(fv):
            continue
        cleaned.append(fv)
    if len(cleaned) == 0:
        return default
    return float(sum(cleaned) / len(cleaned))


def _sum_edge_counts(d):
    return int(sum(int(v) for v in d.values()))


def _compute_kpp_odr_by_relation(
    kept_edges,
    total_edges,
    key_metapaths,
):
    key_rel_set = set()
    for mp in key_metapaths or []:
        for etype in mp:
            key_rel_set.add(tuple(etype))

    key_removed = 0
    offkey_removed = 0
    removed_total = 0
    for etype, total in total_edges.items():
        total_i = int(total)
        kept_i = int(kept_edges.get(etype, 0))
        removed_i = max(total_i - kept_i, 0)
        removed_total += removed_i
        if tuple(etype) in key_rel_set:
            key_removed += removed_i
        else:
            offkey_removed += removed_i

    if removed_total <= 0:
        return {"kpp": 0.0, "odr": 0.0}
    return {
        "kpp": float(key_removed / removed_total),
        "odr": float(offkey_removed / removed_total),
    }


def _compute_kpp_odr_by_instance(
    kept_mask,
    total_edges,
    key_edge_sets
):
    key_removed = 0
    offkey_removed = 0
    removed_total = 0

    for etype, total in total_edges.items():
        total_i = int(total)
        mask = kept_mask.get(etype, None)
        key_set = set(int(x) for x in key_edge_sets.get(etype, []))

        if mask is None:
            continue
        if len(mask) < total_i:
            mask = mask + [1] * (total_i - len(mask))
        elif len(mask) > total_i:
            mask = mask[:total_i]

        for eid in range(total_i):
            removed = 1 - int(mask[eid])
            if removed <= 0:
                continue
            removed_total += 1
            if eid in key_set:
                key_removed += 1
            else:
                offkey_removed += 1

    if removed_total <= 0:
        return {"kpp": 0.0, "odr": 0.0}
    return {
        "kpp": float(key_removed / removed_total),
        "odr": float(offkey_removed / removed_total),
    }


def compute_single_explanation_metrics(
    *,
    orig_probs: List[float],
    cf_probs: List[float],
    orig_class: int,
    target_class: int,
    key_ratios: Dict[str, float],                 
    nonkey_ratios: Optional[Dict[str, float]],   
    key_metapaths: List[List[EdgeType]],
    key_edge_sets: Optional[Dict[EdgeType, List[int]]] = None,
    kept_mask: Optional[Dict[EdgeType, List[int]]] = None,
    fidelity_plus: Optional[float] = None,
    fidelity_minus: Optional[float] = None,
    kept_edges: Dict[EdgeType, int] = None,
    total_edges: Dict[EdgeType, int] = None,
    best_pred: int = None,
    soft_keep_ratio: Optional[float] = None,      
) -> SingleExplanationMetrics:
    orig_probs = [float(x) for x in orig_probs]
    cf_probs = [float(x) for x in cf_probs]
    kept_edges = kept_edges or {}
    total_edges = total_edges or {}

    flip_success = 1.0 if int(best_pred) != int(orig_class) else 0.0
    orig_prob_drop = float(orig_probs[orig_class] - cf_probs[orig_class])
    target_prob_gain = float(cf_probs[target_class] - orig_probs[target_class])

    num_edges_total = _sum_edge_counts(total_edges)
    num_edges_kept = _sum_edge_counts(kept_edges)
    num_edges_removed = int(max(num_edges_total - num_edges_kept, 0))
    keep_ratio = float(num_edges_kept / max(num_edges_total, 1))
    delete_ratio = float(num_edges_removed / max(num_edges_total, 1))

    if key_edge_sets is not None and kept_mask is not None:
        semantic_info = _compute_kpp_odr_by_instance(
            kept_mask=kept_mask,
            total_edges=total_edges,
            key_edge_sets=key_edge_sets,
        )
    else:
        semantic_info = _compute_kpp_odr_by_relation(
            kept_edges=kept_edges,
            total_edges=total_edges,
            key_metapaths=key_metapaths,
        )

    fidelity_plus_val = float(fidelity_plus) if fidelity_plus is not None else float("nan")
    fidelity_minus_val = float(fidelity_minus) if fidelity_minus is not None else float("nan")

    cf_effect = orig_prob_drop
    f_effect = fidelity_minus_val

    return SingleExplanationMetrics(
        flip_success=flip_success,
        orig_prob_drop=orig_prob_drop,
        target_prob_gain=target_prob_gain,
        num_edges_total=num_edges_total,
        num_edges_kept=num_edges_kept,
        num_edges_removed=num_edges_removed,
        keep_ratio=keep_ratio,
        delete_ratio=delete_ratio,
        kpp=float(semantic_info["kpp"]),
        odr=float(semantic_info["odr"]),
        fidelity_plus=fidelity_plus_val,
        fidelity_minus=fidelity_minus_val,
        cf_effect=cf_effect,
        f_effect=f_effect,
    )


def metrics_to_dict(m):
    return {
        "flip_success": m.flip_success,
        "orig_prob_drop": m.orig_prob_drop,
        "target_prob_gain": m.target_prob_gain,
        "num_edges_total": float(m.num_edges_total),
        "num_edges_kept": float(m.num_edges_kept),
        "num_edges_removed": float(m.num_edges_removed),
        "keep_ratio": m.keep_ratio,
        "delete_ratio": m.delete_ratio,
        "kpp": m.kpp,
        "odr": m.odr,
        "cf_effect": m.cf_effect,
        "f_effect": m.f_effect,
    }


def aggregate_metrics(metrics_list):
    if len(metrics_list) == 0:
        return {}
    keys = list(metrics_to_dict(metrics_list[0]).keys())
    out = {}
    for k in keys:
        vals = [metrics_to_dict(m)[k] for m in metrics_list]
        out[f"mean_{k}"] = _safe_mean(vals, default=0.0)
    return out


def print_single_metrics(m):
    print("=" * 100)
    print("Single Explanation Metrics")
    print("=" * 100)
    for k, v in metrics_to_dict(m).items():
        if isinstance(v, float) and math.isnan(v):
            print(f"{k:<28}: N/A")
        else:
            print(f"{k:<28}: {v:.6f}" if isinstance(v, float) else f"{k:<28}: {v}")
    print("=" * 100)


def print_aggregate_metrics(agg: Dict[str, float]) -> None:
    print("=" * 100)
    print("Aggregate Metrics")
    print("=" * 100)
    for k, v in agg.items():
        print(f"{k:<34}: {v:.6f}")
    print("=" * 100)
