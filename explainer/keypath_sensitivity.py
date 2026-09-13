from collections import defaultdict
from typing import Dict, List, Tuple, Any

import torch

from explainer.metapath_utils import collect_metapath_instances_from_target
from explainer.dynamic_semantic import attach_edge_indices_to_instances


EdgeType = Tuple[str, str, str]
MetaPath = List[EdgeType]


def collect_unique_edge_indices_for_metapath(
    target_local_id,
    target_node_type,
    metapath,
    local_edge_index_dict,
    local_edge_lookup,
    max_instances,
    allow_revisit_nodes,
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

    edge_sets = defaultdict(set)

    for ins in instances:
        for edge_step in ins["edge_path"]:
            etype = edge_step["etype"]
            edge_idx = int(edge_step["edge_index"])
            edge_sets[etype].add(edge_idx)

    edge_indices_dict = {}
    num_unique_edges = 0

    for etype, idx_set in edge_sets.items():
        idx_list = sorted(list(idx_set))
        edge_indices_dict[etype] = torch.tensor(idx_list, dtype=torch.long)
        num_unique_edges += len(idx_list)

    return {
        "metapath": metapath,
        "num_instances": len(instances),
        "edge_indices_dict": edge_indices_dict,
        "num_unique_edges": num_unique_edges,
    }


def prune_local_edge_index_dict(
    local_edge_index_dict,
    edge_indices_to_remove
):
    pruned = {}

    for etype, edge_index in local_edge_index_dict.items():
        num_edges = edge_index.size(1)

        if etype not in edge_indices_to_remove:
            pruned[etype] = edge_index
            continue

        remove_idx = edge_indices_to_remove[etype]
        keep_mask = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)
        keep_mask[remove_idx.to(edge_index.device)] = False

        if keep_mask.sum().item() == 0:
            pruned[etype] = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        else:
            pruned[etype] = edge_index[:, keep_mask]

    return pruned


def summarize_sensitivity_results(
    sensitivity_results,
    topk
):
    ranked = sorted(sensitivity_results, key=lambda x: x["prob_drop"], reverse=True)

    print("=" * 100)
    print("Meta-path Sensitivity Ranking")
    print("=" * 100)

    for i, item in enumerate(ranked[:topk]):
        print(
            f"[{i}] "
            f"drop={item['prob_drop']:.6f} | "
            f"base_prob={item['base_orig_prob']:.6f} | "
            f"pruned_prob={item['pruned_orig_prob']:.6f} | "
            f"instances={item['num_instances']} | "
            f"unique_edges={item['num_unique_edges']} | "
            f"metapath={item['metapath']}"
        )

    print("=" * 100)