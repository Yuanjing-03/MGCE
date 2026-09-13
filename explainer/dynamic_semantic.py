from typing import Dict, List, Tuple, Any

import torch

from explainer.metapath_utils import collect_metapath_instances_from_target


EdgeType = Tuple[str, str, str]
MetaPath = List[EdgeType]


def compute_instance_mask_product(instance,mask_dict,eps: float = 1e-12):
    value = None

    for edge_step in instance["edge_path"]:
        etype = edge_step["etype"]
        src_local = edge_step["src_local"]
        dst_local = edge_step["dst_local"]

        edge_index = edge_step["edge_index"]
        mask_value = mask_dict[etype][edge_index]

        if value is None:
            value = mask_value
        else:
            value = value * mask_value

    if value is None:
        value = torch.tensor(eps, dtype=torch.float32)

    return value


def attach_edge_indices_to_instances(instances,local_edge_lookup):
    new_instances = []

    for ins in instances:
        new_ins = {
            "node_path": ins["node_path"].copy(),
            "edge_path": [],
        }

        valid = True
        for edge_step in ins["edge_path"]:
            etype = edge_step["etype"]
            src_local = edge_step["src_local"]
            dst_local = edge_step["dst_local"]

            if etype not in local_edge_lookup:
                valid = False
                break

            edge_lookup = local_edge_lookup[etype]
            edge_key = (src_local, dst_local)

            if edge_key not in edge_lookup:
                valid = False
                break

            edge_index = edge_lookup[edge_key]

            new_edge_step = dict(edge_step)
            new_edge_step["edge_index"] = edge_index
            new_ins["edge_path"].append(new_edge_step)

        if valid:
            new_instances.append(new_ins)

    return new_instances


def compute_metapath_dynamic_strength(
    target_local_id,
    target_node_type,
    metapath,
    local_edge_index_dict,
    local_edge_lookup,
    mask_dict,
    max_instances,
    allow_revisit_nodes,
    eps: float = 1e-12
):
    raw_instances = collect_metapath_instances_from_target(
        target_local_id=target_local_id,
        target_node_type=target_node_type,
        metapath=metapath,
        local_edge_index_dict=local_edge_index_dict,
        max_instances=max_instances,
        allow_revisit_nodes=allow_revisit_nodes,
    )

    instances = attach_edge_indices_to_instances(raw_instances, local_edge_lookup)

    num_instances = len(instances)
    s_orig = float(num_instances)

    if num_instances == 0:
        s_mask = torch.tensor(0.0, dtype=torch.float32)
        retention_ratio = torch.tensor(0.0, dtype=torch.float32)
    else:
        instance_values = []
        for ins in instances:
            v = compute_instance_mask_product(ins, mask_dict=mask_dict, eps=eps)
            instance_values.append(v)

        s_mask = torch.stack(instance_values).sum()
        retention_ratio = s_mask / (torch.tensor(s_orig, device=s_mask.device) + eps)

    return {
        "metapath": metapath,
        "instances": instances,
        "num_instances": num_instances,
        "strength_orig": s_orig,
        "strength_masked": s_mask,
        "retention_ratio": retention_ratio,
    }


def compute_all_metapath_dynamic_strengths(
    target_local_id,
    target_node_type,
    metapaths,
    local_edge_index_dict,
    local_edge_lookup,
    mask_dict,
    max_instances,
    allow_revisit_nodes,
    eps: float = 1e-12
):
    results = []

    for mp in metapaths:
        result = compute_metapath_dynamic_strength(
            target_local_id=target_local_id,
            target_node_type=target_node_type,
            metapath=mp,
            local_edge_index_dict=local_edge_index_dict,
            local_edge_lookup=local_edge_lookup,
            mask_dict=mask_dict,
            max_instances=max_instances,
            allow_revisit_nodes=allow_revisit_nodes,
            eps=eps,
        )
        results.append(result)

    return results


def summarize_dynamic_strengths(
    dynamic_results,
    topk: int = 10,
    sort_by: str = "strength_orig",
):
    def _score(x):
        v = x[sort_by]
        if isinstance(v, torch.Tensor):
            return float(v.detach().cpu().item())
        return float(v)

    ranked = sorted(dynamic_results, key=_score, reverse=True)

    print("=" * 80)
    print("Dynamic Meta-path Semantic Strength Summary")
    print("=" * 80)

    for i, item in enumerate(ranked[:topk]):
        s_orig = item["strength_orig"]
        s_mask = item["strength_masked"]
        rho = item["retention_ratio"]

        if isinstance(s_mask, torch.Tensor):
            s_mask = float(s_mask.detach().cpu().item())
        if isinstance(rho, torch.Tensor):
            rho = float(rho.detach().cpu().item())

        print(
            f"[{i}] "
            f"orig={s_orig:.6f} "
            f"masked={s_mask:.6f} "
            f"ratio={rho:.6f} "
            f"num_instances={item['num_instances']} "
            f"metapath={item['metapath']}"
        )

    print("=" * 80)


def rank_key_metapaths_by_original_strength(
    dynamic_results,
    topk
):
    ranked = sorted(dynamic_results, key=lambda x: x["strength_orig"], reverse=True)
    return ranked[:topk]