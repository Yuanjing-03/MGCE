from typing import Dict, List, Tuple, Any, Set, Optional
import torch

EdgeType = Tuple[str, str, str]
MetaPath = List[EdgeType]


def extract_k_hop_hetero_subgraph(
    target_node_type,
    target_node_id,
    edge_index_dict,
    num_hops
):

    visited_nodes: Dict[str, Set[int]] = {target_node_type: {int(target_node_id)}}
    frontier: Dict[str, Set[int]] = {target_node_type: {int(target_node_id)}}

    for _ in range(num_hops):
        new_frontier: Dict[str, Set[int]] = {}

        for etype, edge_index in edge_index_dict.items():
            src_type, _, dst_type = etype
            src, dst = edge_index[0].tolist(), edge_index[1].tolist()

            if src_type in frontier:
                frontier_nodes = frontier[src_type]
                for s, d in zip(src, dst):
                    if s in frontier_nodes:
                        visited_nodes.setdefault(dst_type, set()).add(int(d))
                        new_frontier.setdefault(dst_type, set()).add(int(d))

            if dst_type in frontier:
                frontier_nodes = frontier[dst_type]
                for s, d in zip(src, dst):
                    if d in frontier_nodes:
                        visited_nodes.setdefault(src_type, set()).add(int(s))
                        new_frontier.setdefault(src_type, set()).add(int(s))

        frontier = new_frontier

    node_dict = {}
    local_id_maps = {}

    for ntype, global_ids in visited_nodes.items():
        sorted_ids = sorted(list(global_ids))
        gids = torch.tensor(sorted_ids, dtype=torch.long)
        node_dict[ntype] = {"global_nids": gids}
        local_id_maps[ntype] = {gid: i for i, gid in enumerate(sorted_ids)}

    local_edge_index_dict = {}

    for etype, edge_index in edge_index_dict.items():
        src_type, _, dst_type = etype
        if src_type not in node_dict or dst_type not in node_dict:
            continue

        src_all = edge_index[0].tolist()
        dst_all = edge_index[1].tolist()

        kept_src = []
        kept_dst = []

        src_map = local_id_maps[src_type]
        dst_map = local_id_maps[dst_type]

        for s, d in zip(src_all, dst_all):
            if s in src_map and d in dst_map:
                kept_src.append(src_map[s])
                kept_dst.append(dst_map[d])

        if len(kept_src) > 0:
            local_edge_index_dict[etype] = torch.tensor([kept_src, kept_dst], dtype=torch.long)
        else:
            local_edge_index_dict[etype] = torch.empty((2, 0), dtype=torch.long)

    target_local_id = local_id_maps[target_node_type][int(target_node_id)]

    return {
        "node_dict": node_dict,
        "edge_index_dict": local_edge_index_dict,
        "target_local_id": target_local_id,
        "local_id_maps": local_id_maps,
    }


def build_local_edge_lookup(local_edge_index_dict):
    lookup = {}
    for etype, edge_index in local_edge_index_dict.items():
        mapping = {}
        for idx in range(edge_index.size(1)):
            s = int(edge_index[0, idx].item())
            d = int(edge_index[1, idx].item())
            mapping[(s, d)] = idx
        lookup[etype] = mapping
    return lookup


def _collect_instances_in_local_graph(
    target_local_id,
    target_node_type,
    metapath,
    local_edge_index_dict,
    max_instances,
    allow_revisit_nodes):
    results = []

    def dfs(step, cur_type, cur_local_id, node_path, edge_path, visited):
        if len(results) >= max_instances:
            return
        if step == len(metapath):
            results.append(
                {
                    "node_path": list(node_path),
                    "edge_path": list(edge_path),
                }
            )
            return

        etype = metapath[step]
        src_type, _, dst_type = etype
        if cur_type != src_type:
            return
        if etype not in local_edge_index_dict:
            return

        edge_index = local_edge_index_dict[etype]
        for eidx in range(edge_index.size(1)):
            s = int(edge_index[0, eidx].item())
            d = int(edge_index[1, eidx].item())
            if s != cur_local_id:
                continue

            next_node = (dst_type, d)
            if (not allow_revisit_nodes) and next_node in visited:
                continue

            edge_info = {
                "etype": etype,
                "src_local": s,
                "dst_local": d,
                "edge_index": eidx,
            }

            node_path.append(next_node)
            edge_path.append(edge_info)

            if not allow_revisit_nodes:
                visited.add(next_node)

            dfs(
                step + 1,
                dst_type,
                d,
                node_path,
                edge_path,
                visited,
            )

            if not allow_revisit_nodes:
                visited.remove(next_node)

            node_path.pop()
            edge_path.pop()

    start_node = (target_node_type, int(target_local_id))
    dfs(
        0,
        target_node_type,
        int(target_local_id),
        [start_node],
        [],
        {start_node},
    )
    return results


def extract_metapath_guided_hetero_subgraph(
    target_node_type,
    target_node_id,
    edge_index_dict,
    candidate_metapaths,
    num_hops,
    max_instances_per_metapath,
    min_instances_to_keep_path
):
    base = extract_k_hop_hetero_subgraph(
        target_node_type=target_node_type,
        target_node_id=target_node_id,
        edge_index_dict=edge_index_dict,
        num_hops=num_hops,
    )

    base_node_dict = base["node_dict"]
    base_edge_index_dict = base["edge_index_dict"]
    target_local_id = base["target_local_id"]

    selected_nodes: Dict[str, Set[int]] = {target_node_type: {int(target_local_id)}}
    selected_edges: Dict[EdgeType, Set[int]] = {}

    valid_paths_found = 0

    for mp in candidate_metapaths:
        instances = _collect_instances_in_local_graph(
            target_local_id=target_local_id,
            target_node_type=target_node_type,
            metapath=mp,
            local_edge_index_dict=base_edge_index_dict,
            max_instances=max_instances_per_metapath,
            allow_revisit_nodes=True,
        )

        if len(instances) < min_instances_to_keep_path:
            continue

        valid_paths_found += 1

        for ins in instances:
            for ntype, nid in ins["node_path"]:
                selected_nodes.setdefault(ntype, set()).add(int(nid))

            for edge_step in ins["edge_path"]:
                etype = edge_step["etype"]
                eidx = int(edge_step["edge_index"])
                selected_edges.setdefault(etype, set()).add(eidx)

    if valid_paths_found == 0:
        return base

    new_node_dict = {}
    new_local_maps = {}

    for ntype, local_ids in selected_nodes.items():
        sorted_local_ids = sorted(list(local_ids))
        old_global = base_node_dict[ntype]["global_nids"][sorted_local_ids]
        new_node_dict[ntype] = {"global_nids": old_global.clone()}
        new_local_maps[ntype] = {old_lid: new_lid for new_lid, old_lid in enumerate(sorted_local_ids)}

    new_edge_index_dict = {}


    for etype, edge_index in base_edge_index_dict.items():
        src_type, _, dst_type = etype
        chosen = selected_edges.get(etype, set())

        if len(chosen) == 0:
            new_edge_index_dict[etype] = torch.empty((2, 0), dtype=torch.long)
            continue

        new_src = []
        new_dst = []

        src_map = new_local_maps.get(src_type, {})
        dst_map = new_local_maps.get(dst_type, {})

        for old_eidx in sorted(list(chosen)):
            if old_eidx >= edge_index.size(1):
                continue

            old_s = int(edge_index[0, old_eidx].item())
            old_d = int(edge_index[1, old_eidx].item())

            if old_s in src_map and old_d in dst_map:
                new_src.append(src_map[old_s])
                new_dst.append(dst_map[old_d])

        if len(new_src) > 0:
            new_edge_index_dict[etype] = torch.tensor([new_src, new_dst], dtype=torch.long)
        else:
            new_edge_index_dict[etype] = torch.empty((2, 0), dtype=torch.long)

    new_target_local_id = new_local_maps[target_node_type][int(target_local_id)]

    return {
        "node_dict": new_node_dict,
        "edge_index_dict": new_edge_index_dict,
        "target_local_id": new_target_local_id,
        "local_id_maps": new_local_maps,
        "base_subgraph_info": base,
    }