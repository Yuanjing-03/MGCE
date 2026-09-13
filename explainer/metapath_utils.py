from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import torch
from explainer import _verbosity as _v


EdgeType = Tuple[str, str, str]
MetaPath = List[EdgeType]


def is_valid_metapath(
    metapath,
    available_edge_types,
    start_node_type,
):
    if len(metapath) == 0:
        return False

    edge_type_set = set(available_edge_types)

    for etype in metapath:
        if etype not in edge_type_set:
            return False

    if start_node_type is not None:
        if metapath[0][0] != start_node_type:
            return False

    for i in range(len(metapath) - 1):
        _, _, cur_dst = metapath[i]
        next_src, _, _ = metapath[i + 1]
        if cur_dst != next_src:
            return False

    return True


def filter_valid_metapaths(
    candidate_metapaths,
    available_edge_types,
    start_node_type
):
    valid = []
    for mp in candidate_metapaths:
        if is_valid_metapath(mp, available_edge_types, start_node_type=start_node_type):
            valid.append(mp)
    return valid


def build_typed_local_adjacency(local_edge_index_dict):

    typed_adj = {}

    for etype, edge_index in local_edge_index_dict.items():
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()

        adj = defaultdict(list)
        for s, d in zip(src, dst):
            adj[s].append(d)

        typed_adj[etype] = dict(adj)

    return typed_adj


def collect_metapath_instances_from_target(
    target_local_id,
    target_node_type,
    metapath,
    local_edge_index_dict,
    max_instances,
    allow_revisit_nodes
):
    if len(metapath) == 0:
        return []

    if metapath[0][0] != target_node_type:
        return []

    typed_adj = build_typed_local_adjacency(local_edge_index_dict)
    instances = []

    start_state = [(target_node_type, target_local_id)]

    def dfs(step: int, cur_local_id: int, node_path, edge_path, visited_nodes):
        if len(instances) >= max_instances:
            return

        if step == len(metapath):
            instances.append(
                {
                    "node_path": node_path.copy(),
                    "edge_path": edge_path.copy(),
                }
            )
            return

        etype = metapath[step]
        src_type, _, dst_type = etype

        if node_path[-1][0] != src_type:
            return

        next_nodes = typed_adj.get(etype, {}).get(cur_local_id, [])

        for nxt in next_nodes:
            node_key = (dst_type, nxt)
            if (not allow_revisit_nodes) and (node_key in visited_nodes):
                continue

            node_path.append((dst_type, nxt))
            edge_path.append(
                {
                    "etype": etype,
                    "src_local": cur_local_id,
                    "dst_local": nxt,
                }
            )

            added = False
            if not allow_revisit_nodes:
                visited_nodes.add(node_key)
                added = True

            dfs(step + 1, nxt, node_path, edge_path, visited_nodes)

            if added:
                visited_nodes.remove(node_key)

            node_path.pop()
            edge_path.pop()

    visited_nodes = set(start_state) if not allow_revisit_nodes else set()
    dfs(
        step=0,
        cur_local_id=target_local_id,
        node_path=start_state.copy(),
        edge_path=[],
        visited_nodes=visited_nodes,
    )

    return instances


def summarize_metapath_instances(
    metapath,
    instances,
    max_show,
):

    if not _v.VERBOSE:
        return

    print("=" * 80)
    print("Meta-path Instance Summary")
    print("=" * 80)
    print(f"Meta-path       : {metapath}")
    print(f"Num instances   : {len(instances)}")
    print("-" * 80)

    for i, ins in enumerate(instances[:max_show]):
        print(f"[Instance {i}]")
        print(f"  node_path = {ins['node_path']}")
        print(f"  edge_path = {ins['edge_path']}")
        print("-" * 80)

    print("=" * 80)


def collect_edges_used_by_instances(instances):

    edge_usage = defaultdict(list)

    for ins in instances:
        for edge_step in ins["edge_path"]:
            etype = edge_step["etype"]
            src_local = edge_step["src_local"]
            dst_local = edge_step["dst_local"]
            edge_usage[etype].append((src_local, dst_local))

    return dict(edge_usage)


def count_edge_frequency_in_instances(instances):
    freq = defaultdict(lambda: defaultdict(int))

    for ins in instances:
        for edge_step in ins["edge_path"]:
            etype = edge_step["etype"]
            edge_key = (edge_step["src_local"], edge_step["dst_local"])
            freq[etype][edge_key] += 1

    return {etype: dict(v) for etype, v in freq.items()}