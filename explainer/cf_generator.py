from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Set
import random
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from explainer.subgraph_utils import (
    extract_k_hop_hetero_subgraph,
    extract_metapath_guided_hetero_subgraph,
    build_local_edge_lookup,
)
from explainer.metapath_utils import (
    filter_valid_metapaths,
    collect_metapath_instances_from_target,
)
from explainer.mask_module import EdgeMaskModule
from explainer.dynamic_semantic import attach_edge_indices_to_instances
from explainer.keypath_sensitivity import (
    collect_unique_edge_indices_for_metapath,
    prune_local_edge_index_dict,
)

EdgeType = Tuple[str, str, str]
MetaPath = List[EdgeType]


@dataclass
class CFExplainResult:
    target_node_id: int
    target_node_type: str
    orig_class: int
    target_class: int
    orig_probs: List[float]
    best_probs: List[float]
    best_pred: int
    flipped: bool
    best_epoch: int
    best_score: float
    key_metapaths: List[MetaPath]
    key_ratios: Dict[str, float]
    nonkey_ratios: Dict[str, float]
    key_sensitivities: Dict[str, float]
    kept_edges: Dict[EdgeType, int]
    total_edges: Dict[EdgeType, int]
    soft_keep_ratio: float
    # Instance-level info for more precise evaluation
    key_edge_sets: Dict[EdgeType, List[int]]
    kept_mask: Dict[EdgeType, List[int]]
    fidelity_plus: float
    fidelity_minus: float


class MACFCounterfactualExplainer:
    def __init__(
        self,
        model,
        full_data: HeteroData,
        candidate_metapaths: List[MetaPath],
        target_node_type: str,
        num_hops: int = 2,
        topk_paths: int = 3,
        mask_mode: str = "sigmoid",
        pred_coeff: float = 1.0,
        meta_coeff: float = 1.0,
        preserve_coeff: float = 0.0,
        preserve_ratio_floor: float = 0.4,
        edge_preserve_coeff: float = 0.0,
        edge_preserve_floor: float = 0.85,
        spar_coeff: float = 0.1,
        ent_coeff: float = 0.001,
        lr: float = 0.05,
        epochs: int = 200,
        max_instances: int = 1000,
        reinforce_baseline_momentum: float = 0.9,
        device: str = "cuda",
        verbose: bool = False,
        speedup: bool = False,
        use_meta_loss: bool = True,
        use_keypath_ranking: bool = True,
        use_key_sparsity: bool = True,
        uniform_edge_perturbation: bool = False,
        semantic_selection: bool = True,
    ):
        self.model = model
        self.full_data = full_data
        self.candidate_metapaths = candidate_metapaths
        self.target_node_type = target_node_type

        self.num_hops = num_hops
        self.topk_paths = topk_paths
        self.mask_mode = mask_mode

        self.pred_coeff = pred_coeff
        self.meta_coeff = meta_coeff
        self.preserve_coeff = preserve_coeff
        self.preserve_ratio_floor = preserve_ratio_floor
        self.edge_preserve_coeff = edge_preserve_coeff
        self.edge_preserve_floor = edge_preserve_floor
        self.spar_coeff = spar_coeff
        self.ent_coeff = ent_coeff

        self.lr = lr
        self.epochs = epochs
        self.max_instances = max_instances
        self.reinforce_baseline_momentum = reinforce_baseline_momentum

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        self.verbose = verbose
        self.speedup = bool(speedup)
        self.use_meta_loss = bool(use_meta_loss)
        self.use_keypath_ranking = bool(use_keypath_ranking)
        self.use_key_sparsity = bool(use_key_sparsity)
        self.uniform_edge_perturbation = bool(uniform_edge_perturbation)
        self.semantic_selection = bool(semantic_selection)
        try:
            from explainer import _verbosity as _v

            _v.VERBOSE = bool(self.verbose)
        except Exception:
            pass

        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.full_data = self.full_data.to(self.device)
        self.node_semantic_cache = {}

    @torch.no_grad()
    def _get_original_prediction(self, target_node_id: int) -> Dict[str, Any]:
        logits = self.model(self.full_data)
        probs = F.softmax(logits, dim=-1)

        target_probs = probs[target_node_id]
        orig_class = int(target_probs.argmax().item())

        sorted_idx = torch.argsort(target_probs, descending=True)
        target_class = int(sorted_idx[1].item()) if len(sorted_idx) >= 2 else orig_class

        return {
            "logits": logits[target_node_id],
            "probs": target_probs,
            "orig_class": orig_class,
            "target_class": target_class,
        }

    def _build_local_data(self,subgraph_info,masked_edge_index_dict):
        local_data = HeteroData()

        node_dict = subgraph_info["node_dict"]
        edge_index_dict = masked_edge_index_dict or subgraph_info["edge_index_dict"]
        for ntype in self.full_data.node_types:
            if ntype in node_dict:
                global_nids = node_dict[ntype]["global_nids"].to(self.device)
            else:
                global_nids = torch.empty((0,), dtype=torch.long, device=self.device)

            local_data[ntype].global_nids = global_nids
            local_data[ntype].num_nodes = int(global_nids.numel())

            if hasattr(self.full_data[ntype], "x") and self.full_data[ntype].x is not None:
                if global_nids.numel() > 0:
                    local_data[ntype].x = self.full_data[ntype].x[global_nids].to(self.device)
                else:
                    feat_dim = self.full_data[ntype].x.size(-1)
                    local_data[ntype].x = torch.empty(
                        (0, feat_dim),
                        dtype=self.full_data[ntype].x.dtype,
                        device=self.device,
                    )

            if hasattr(self.full_data[ntype], "y") and self.full_data[ntype].y is not None:
                if global_nids.numel() > 0:
                    local_data[ntype].y = self.full_data[ntype].y[global_nids].to(self.device)
                else:
                    local_data[ntype].y = torch.empty(
                        (0,),
                        dtype=self.full_data[ntype].y.dtype,
                        device=self.device,
                    )

        for etype in self.full_data.edge_types:
            if etype in edge_index_dict:
                local_data[etype].edge_index = edge_index_dict[etype].to(self.device)
            else:
                local_data[etype].edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

        return local_data

    def _sample_hard_masks_and_logprob(self,mask_module):
        mask_logits_dict = mask_module.get_mask_logits_dict()

        sampled_mask_dict = {}
        total_logprob = torch.tensor(0.0, device=self.device)
        total_entropy = torch.tensor(0.0, device=self.device)

        for etype, logits in mask_logits_dict.items():
            logits = logits.to(self.device)
            probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)

            bern = torch.distributions.Bernoulli(probs=probs)
            sample = bern.sample()
            logprob = bern.log_prob(sample).sum()
            entropy = bern.entropy().mean()

            sampled_mask_dict[etype] = sample
            total_logprob = total_logprob + logprob
            total_entropy = total_entropy + entropy

        return sampled_mask_dict, total_logprob, total_entropy

    def _build_masked_edge_index_dict(self,local_edge_index_dict,hard_mask_dict):
        masked_edge_index_dict = {}

        for etype, edge_index in local_edge_index_dict.items():
            hard_mask = hard_mask_dict[etype].to(edge_index.device)
            keep = hard_mask > 0.5

            if keep.sum().item() == 0:
                masked_edge_index_dict[etype] = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
            else:
                masked_edge_index_dict[etype] = edge_index[:, keep]

        return masked_edge_index_dict

    @torch.no_grad()
    def _eval_local_prediction(self,local_data,target_local_id):
        logits = self.model(local_data)
        probs = F.softmax(logits, dim=-1)[target_local_id]
        pred = int(probs.argmax().item())
        return {
            "probs": probs,
            "pred": pred,
        }

    def _prediction_loss_from_probs(
        self,
        probs,
        orig_class,
        target_class,
        margin,
    ):
        return F.relu(probs[orig_class] - probs[target_class] + margin) - 1.2 * probs[target_class]

    def _rank_key_metapaths_by_sensitivity(
        self,
        subgraph_info,
        local_edge_index_dict,
        local_edge_lookup,
        valid_metapaths,
        target_local_id,
        orig_class,
        alpha,
        beta,
        min_instances,
    ):
        base_local_data = self._build_local_data(subgraph_info, local_edge_index_dict)
        base_pred_info = self._eval_local_prediction(base_local_data, target_local_id)
        base_orig_prob = float(base_pred_info["probs"][orig_class].item())

        eps = 1e-12
        metapath_stats = []

        total_fair_strength = 0.0
        total_instances = 0

        for mp in valid_metapaths:
            edge_info = collect_unique_edge_indices_for_metapath(
                target_local_id=target_local_id,
                target_node_type=self.target_node_type,
                metapath=mp,
                local_edge_index_dict=local_edge_index_dict,
                local_edge_lookup=local_edge_lookup,
                max_instances=self.max_instances,
                allow_revisit_nodes=True,
            )

            num_instances = int(edge_info["num_instances"])
            edge_indices_dict = edge_info["edge_indices_dict"]
            num_unique_edges = int(edge_info["num_unique_edges"])
            path_len = len(mp)

            fair_strength = 0.0
            if num_instances >= min_instances:
                fair_strength = float(torch.log1p(torch.tensor(float(num_instances))).item()) / (
                    (float(path_len) ** beta) + eps
                )
                total_fair_strength += fair_strength
                total_instances += num_instances

            metapath_stats.append(
                {
                    "metapath": mp,
                    "num_instances": num_instances,
                    "edge_indices_dict": edge_indices_dict,
                    "num_unique_edges": num_unique_edges,
                    "path_len": path_len,
                    "fair_strength": fair_strength,
                }
            )

        total_fair_strength = max(total_fair_strength, eps)
        total_instances = max(total_instances, 1)

        sensitivity_results = []

        cache = {} if self.speedup else None

        for stat in metapath_stats:
            mp = stat["metapath"]
            num_instances = stat["num_instances"]
            edge_indices_dict = stat["edge_indices_dict"]
            num_unique_edges = stat["num_unique_edges"]
            path_len = stat["path_len"]
            fair_strength = stat["fair_strength"]

            if num_instances == 0 or num_unique_edges == 0:
                prob_drop = 0.0
                pruned_orig_prob = base_orig_prob
            else:
                if cache is not None:
                    key_items = []
                    for et, idxs in sorted(edge_indices_dict.items(), key=lambda x: str(x[0])):
                        key_items.append((str(et), tuple(sorted(idxs))))
                    cache_key = tuple(key_items)
                    cached = cache.get(cache_key, None)
                    if cached is not None:
                        pruned_orig_prob = cached
                    else:
                        pruned_edge_index_dict = prune_local_edge_index_dict(
                            local_edge_index_dict=local_edge_index_dict,
                            edge_indices_to_remove=edge_indices_dict,
                        )
                        pruned_local_data = self._build_local_data(subgraph_info, pruned_edge_index_dict)
                        pruned_pred_info = self._eval_local_prediction(pruned_local_data, target_local_id)
                        pruned_orig_prob = float(pruned_pred_info["probs"][orig_class].item())
                        cache[cache_key] = pruned_orig_prob
                else:
                    pruned_edge_index_dict = prune_local_edge_index_dict(
                        local_edge_index_dict=local_edge_index_dict,
                        edge_indices_to_remove=edge_indices_dict,
                    )
                    pruned_local_data = self._build_local_data(subgraph_info, pruned_edge_index_dict)
                    pruned_pred_info = self._eval_local_prediction(pruned_local_data, target_local_id)
                    pruned_orig_prob = float(pruned_pred_info["probs"][orig_class].item())
                prob_drop = max(0.0, base_orig_prob - pruned_orig_prob)

            edge_norm_drop = prob_drop / (float(num_unique_edges) + eps)
            inst_norm_drop = prob_drop / (float(num_instances) + eps) if num_instances > 0 else 0.0
            fair_local_strength_ratio = fair_strength / total_fair_strength

            path_uniqueness = (float(num_instances) / float(total_instances)) / ((float(path_len) ** 1.0) + eps)

            joint_score = (
                alpha * edge_norm_drop
                + 0.08 * fair_local_strength_ratio
                + 0.02 * path_uniqueness
            )


            sensitivity_results.append(
                {
                    "metapath": mp,
                    "base_orig_prob": base_orig_prob,
                    "pruned_orig_prob": pruned_orig_prob,
                    "prob_drop": prob_drop,
                    "edge_norm_drop": edge_norm_drop,
                    "inst_norm_drop": inst_norm_drop,
                    "fair_local_strength_ratio": fair_local_strength_ratio,
                    "path_uniqueness": path_uniqueness,
                    "joint_score": joint_score,
                    "num_instances": num_instances,
                    "num_unique_edges": num_unique_edges,
                    "path_len": path_len,
                }
            )

        ranked = sorted(
            sensitivity_results,
            key=lambda x: (x["joint_score"], x["edge_norm_drop"], x["prob_drop"]),
            reverse=True,
        )
        return ranked

    def _build_metapath_instance_cache(
        self,
        target_local_id,
        valid_metapaths,
        local_edge_index_dict,
        local_edge_lookup,
    ):
        cache = {}

        for mp in valid_metapaths:
            raw_instances = collect_metapath_instances_from_target(
                target_local_id=target_local_id,
                target_node_type=self.target_node_type,
                metapath=mp,
                local_edge_index_dict=local_edge_index_dict,
                max_instances=self.max_instances,
                allow_revisit_nodes=True,
            )
            instances = attach_edge_indices_to_instances(raw_instances, local_edge_lookup)
            cache[str(mp)] = instances

        return cache

    def _prepare_semantic_cache(
        self,
        target_local_id,
        valid_metapaths,
        local_edge_index_dict,
        local_edge_lookup,
    ):

        mp_cache = self._build_metapath_instance_cache(
            target_local_id=target_local_id,
            valid_metapaths=valid_metapaths,
            local_edge_index_dict=local_edge_index_dict,
            local_edge_lookup=local_edge_lookup,
        )

        meta_index_map = {}

        for mp in valid_metapaths:
            mp_str = str(mp)
            instances = mp_cache.get(mp_str, [])
            num_instances = len(instances)
            if num_instances == 0:
                meta_index_map[mp_str] = []
                continue

            path_len = len(mp)
            per_step = []
            for step in range(path_len):
                etype = instances[0]["edge_path"][step]["etype"]
                idxs = [int(inst["edge_path"][step]["edge_index"]) for inst in instances]
                idx_tensor = torch.tensor(idxs, dtype=torch.long, device=self.device)
                per_step.append((etype, idx_tensor))

            meta_index_map[mp_str] = per_step

        return {"metapath_instance_cache": mp_cache, "meta_index_map": meta_index_map}

    def _build_key_edge_index_sets(
        self,
        key_metapaths,
        metapath_instance_cache,
    ):
        key_edge_sets: Dict[EdgeType, Set[int]] = {}

        for mp in key_metapaths:
            instances = metapath_instance_cache[str(mp)]
            for ins in instances:
                for edge_step in ins["edge_path"]:
                    etype = edge_step["etype"]
                    edge_idx = int(edge_step["edge_index"])
                    if etype not in key_edge_sets:
                        key_edge_sets[etype] = set()
                    key_edge_sets[etype].add(edge_idx)

        return key_edge_sets

    def _build_nonkey_edge_masks(
        self,
        local_edge_index_dict,
        key_edge_sets,
    ):
        nonkey_edge_masks = {}

        for etype, edge_index in local_edge_index_dict.items():
            num_edges = edge_index.size(1)
            mask = torch.ones(num_edges, dtype=torch.bool, device=self.device)

            key_set = key_edge_sets.get(etype, set())
            if len(key_set) > 0:
                idx = torch.tensor(sorted(list(key_set)), dtype=torch.long, device=self.device)
                mask[idx] = False

            nonkey_edge_masks[etype] = mask

        return nonkey_edge_masks

    def _compute_macro_losses_from_cache(
        self,
        valid_metapaths,
        key_metapaths,
        soft_mask_dict,
        metapath_instance_cache,
        meta_index_map,
        eps: float = 1e-12,
    ):
        key_set = {tuple(mp) for mp in key_metapaths}

        key_ratios = {}
        nonkey_ratios = {}
        key_loss_items = []
        preserve_loss_items = []

        for mp in valid_metapaths:
            mp_str = str(mp)
            instances = metapath_instance_cache.get(mp_str, [])
            num_instances = len(instances)

            if num_instances == 0:
                rho = torch.tensor(0.0, device=self.device)
            else:
                if meta_index_map is not None and mp_str in meta_index_map and len(meta_index_map[mp_str]) > 0:
                    per_step = meta_index_map[mp_str]
                    gathered = []
                    for etype, idx_tensor in per_step:
                        m = soft_mask_dict[etype]
                        gathered.append(m[idx_tensor])
                    stacked = torch.stack(gathered, dim=0)
                    prod_per_instance = torch.prod(stacked, dim=0)
                    s_mask = prod_per_instance.sum()
                    s_orig = float(num_instances)
                    rho = s_mask / (torch.tensor(s_orig, device=self.device) + eps)
                else:
                    vals = []
                    for ins in instances:
                        v = None
                        for edge_step in ins["edge_path"]:
                            etype = edge_step["etype"]
                            edge_index = edge_step["edge_index"]
                            m = soft_mask_dict[etype][edge_index]
                            v = m if v is None else v * m
                        vals.append(v)

                    s_mask = torch.stack(vals).sum()
                    s_orig = float(num_instances)
                    rho = s_mask / (torch.tensor(s_orig, device=self.device) + eps)

            if tuple(mp) in key_set:
                key_loss_items.append(rho)
                key_ratios[str(mp)] = float(rho.detach().cpu().item())
            else:
                nonkey_ratios[str(mp)] = float(rho.detach().cpu().item())

                if self.preserve_coeff > 0:
                    if num_instances >= 1:
                        preserve_loss_items.append(
                            torch.relu(torch.tensor(self.preserve_ratio_floor, device=self.device) - rho)
                        )

        loss_meta = (
            torch.stack(key_loss_items).mean()
            if len(key_loss_items) > 0
            else torch.tensor(0.0, device=self.device)
        )

        if self.preserve_coeff > 0 and len(preserve_loss_items) > 0:
            loss_preserve = torch.stack(preserve_loss_items).mean()
        else:
            loss_preserve = torch.tensor(0.0, device=self.device)

        return loss_meta, loss_preserve, key_ratios, nonkey_ratios

    def _compute_edge_preserve_loss(
        self,
        soft_mask_dict,
        nonkey_edge_masks,
    ):
        if self.edge_preserve_coeff <= 0:
            return torch.tensor(0.0, device=self.device)

        loss_items = []

        for etype, m in soft_mask_dict.items():
            nonkey_mask = nonkey_edge_masks[etype]
            if nonkey_mask.sum().item() == 0:
                continue

            nonkey_vals = m[nonkey_mask]
            floor = torch.tensor(self.edge_preserve_floor, device=self.device)
            loss_items.append(torch.relu(floor - nonkey_vals).mean())

        if len(loss_items) == 0:
            return torch.tensor(0.0, device=self.device)

        return torch.stack(loss_items).mean()

    def _compute_key_sparsity_loss(
        self,
        soft_mask_dict,
        key_edge_sets,
    ):
        vals = []

        for etype, m in soft_mask_dict.items():
            key_set = key_edge_sets.get(etype, set())
            if len(key_set) == 0:
                continue

            idx = torch.tensor(sorted(list(key_set)), dtype=torch.long, device=self.device)
            key_vals = m[idx]
            vals.append(key_vals.mean())

        if len(vals) == 0:
            return torch.tensor(0.0, device=self.device)

        return torch.stack(vals).mean()

    def _compute_global_soft_keep_ratio(
        self,
        soft_mask_dict,
    ):
        total_sum = 0.0
        total_count = 0
        for _, m in soft_mask_dict.items():
            total_sum += float(m.sum().detach().cpu().item())
            total_count += int(m.numel())
        return float(total_sum / max(total_count, 1))

    def _bucket_random_select_metapaths(
        self,
        sensitivity_ranked,
        topk_paths,
        num_buckets,
        total_numbers,
        unique,
    ):
        candidates = list(sensitivity_ranked)
        if len(candidates) == 0:
            return []
        print(f"Total candidate metapaths: {len(candidates)}")
        candidates = candidates[: min(num_buckets, len(candidates))]
        actual_buckets = len(candidates)

        if actual_buckets == 0:
            return []

        bucket_size = total_numbers // num_buckets

        selected = []
        used_idx = set()

        need = min(topk_paths, actual_buckets) if unique else topk_paths
        max_trials = 10000

        while len(selected) < need and max_trials > 0:
            max_trials -= 1

            r = random.randint(0, total_numbers - 1)
            bucket_idx = min(r // bucket_size, num_buckets - 1)

            if bucket_idx >= actual_buckets:
                bucket_idx = actual_buckets - 1

            if unique and bucket_idx in used_idx:
                continue

            selected.append(candidates[bucket_idx])
            used_idx.add(bucket_idx)
            print(r, bucket_idx, candidates[bucket_idx]["metapath"])

        if len(selected) < need:
            remaining = [i for i in range(actual_buckets) if i not in used_idx]
            random.shuffle(remaining)
            for idx in remaining:
                selected.append(candidates[idx])
                if len(selected) >= need:
                    break

        return selected
    
    def explain(self, target_node_id):
        orig_info = self._get_original_prediction(target_node_id)
        orig_probs = orig_info["probs"]
        orig_class = orig_info["orig_class"]
        target_class = orig_info["target_class"]

        subgraph_info = extract_metapath_guided_hetero_subgraph(
            target_node_type=self.target_node_type,
            target_node_id=target_node_id,
            edge_index_dict=self.full_data.edge_index_dict,
            candidate_metapaths=self.candidate_metapaths,
            num_hops=self.num_hops,
            max_instances_per_metapath=min(300, self.max_instances),
            min_instances_to_keep_path=1,
        )

        local_edge_index_dict = {
            k: v.to(self.device) for k, v in subgraph_info["edge_index_dict"].items()
        }
        target_local_id = subgraph_info["target_local_id"]

        local_edge_lookup = build_local_edge_lookup(local_edge_index_dict)

        valid_metapaths = filter_valid_metapaths(
            candidate_metapaths=self.candidate_metapaths,
            available_edge_types=list(local_edge_index_dict.keys()),
            start_node_type=self.target_node_type,
        )

        sensitivity_ranked = self._rank_key_metapaths_by_sensitivity(
            subgraph_info=subgraph_info,
            local_edge_index_dict=local_edge_index_dict,
            local_edge_lookup=local_edge_lookup,
            valid_metapaths=valid_metapaths,
            target_local_id=target_local_id,
            orig_class=orig_class,
        )

        if self.use_keypath_ranking:
            key_results = sensitivity_ranked[: self.topk_paths]
        else:
            key_results = self._bucket_random_select_metapaths(
                sensitivity_ranked=sensitivity_ranked,
                topk_paths=self.topk_paths,
                num_buckets=6,
                total_numbers=240,
                unique=True,
            )

        key_metapaths = [item["metapath"] for item in key_results]
        key_sensitivities = {str(item["metapath"]): item["joint_score"] for item in key_results}

        if self.verbose:
            print("=" * 160)
            print("Selected Key Meta-paths by Node-specific Joint Score")
            print("=" * 160)
        for i, item in enumerate(key_results):
            print(
                f"[{i}] "
                f"joint_score={item['joint_score']:.6f} | "
                f"drop={item['prob_drop']:.6f} | "
                f"edge_norm_drop={item['edge_norm_drop']:.6f} | "
                f"fair_local_strength_ratio={item['fair_local_strength_ratio']:.6f} | "
                f"path_uniqueness={item['path_uniqueness']:.6f} | "
                f"inst_norm_drop={item['inst_norm_drop']:.6f} | "
                f"instances={item['num_instances']} | "
                f"unique_edges={item['num_unique_edges']} | "
                f"path_len={item['path_len']} | "
                f"{item['metapath']}"
            )
        print("=" * 160)

        if self.verbose:
            print("[INFO] building meta-path instance cache and index map ...")
        prepared = self._prepare_semantic_cache(
            target_local_id=target_local_id,
            valid_metapaths=valid_metapaths,
            local_edge_index_dict=local_edge_index_dict,
            local_edge_lookup=local_edge_lookup,
        )
        metapath_instance_cache = prepared["metapath_instance_cache"]
        meta_index_map = prepared["meta_index_map"]
        if self.verbose:
            print("[INFO] meta-path instance cache ready.")

        key_edge_sets = self._build_key_edge_index_sets(
            key_metapaths=key_metapaths,
            metapath_instance_cache=metapath_instance_cache,
        )
        nonkey_edge_masks = self._build_nonkey_edge_masks(
            local_edge_index_dict=local_edge_index_dict,
            key_edge_sets=key_edge_sets,
        )

        if self.uniform_edge_perturbation:
            key_edge_sets = {etype: set() for etype in local_edge_index_dict.keys()}
            nonkey_edge_masks = {
                etype: torch.zeros(edge_index.size(1), dtype=torch.bool, device=self.device)
                for etype, edge_index in local_edge_index_dict.items()
            }

        active_edge_mask = {}
        for etype, edge_index in local_edge_index_dict.items():
            num = edge_index.size(1)
            active = torch.zeros(num, dtype=torch.bool, device=self.device)

            for mp_str, instances in metapath_instance_cache.items():
                for ins in instances:
                    for edge_step in ins["edge_path"]:
                        if edge_step["etype"] == etype:
                            idx = int(edge_step["edge_index"])
                            active[idx] = True

            active_edge_mask[etype] = active

        mask_module = EdgeMaskModule(
            local_edge_index_dict=local_edge_index_dict,
            init_strategy="zeros",
            active_edge_mask=active_edge_mask,
        ).to(self.device)

        trainable_params = list(mask_module.trainable_parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=self.lr)

        baseline = None
        best_score = float("inf")
        best_epoch = -1
        best_probs = orig_probs.detach().cpu()
        best_pred = int(orig_class)
        best_key_ratios = {}
        best_nonkey_ratios = {}
        best_hard_mask_dict = None
        best_soft_keep_ratio = float("nan")
        flipped = False

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()

            soft_mask_dict = mask_module.get_mask_dict(mode=self.mask_mode)

            loss_meta, loss_preserve, key_ratios, nonkey_ratios = self._compute_macro_losses_from_cache(
                valid_metapaths=valid_metapaths,
                key_metapaths=key_metapaths,
                soft_mask_dict=soft_mask_dict,
                metapath_instance_cache=metapath_instance_cache,
                meta_index_map=meta_index_map,
            )
            if self.uniform_edge_perturbation:
                loss_edge_preserve = torch.tensor(0.0, device=self.device)
            else:
                loss_edge_preserve = self._compute_edge_preserve_loss(
                    soft_mask_dict=soft_mask_dict,
                    nonkey_edge_masks=nonkey_edge_masks,
                )

            if self.use_key_sparsity and (not self.uniform_edge_perturbation):
                loss_spar = self._compute_key_sparsity_loss(
                    soft_mask_dict=soft_mask_dict,
                    key_edge_sets=key_edge_sets,
                )
            else:
                loss_spar = torch.tensor(0.0, device=self.device)

            hard_mask_dict, total_logprob, total_entropy = self._sample_hard_masks_and_logprob(mask_module)
            masked_edge_index_dict = self._build_masked_edge_index_dict(local_edge_index_dict, hard_mask_dict)
            local_data_masked = self._build_local_data(subgraph_info, masked_edge_index_dict)

            pred_info = self._eval_local_prediction(local_data_masked, target_local_id)
            masked_probs = pred_info["probs"]
            masked_pred = pred_info["pred"]

            pred_loss = self._prediction_loss_from_probs(
                masked_probs,
                orig_class=orig_class,
                target_class=target_class,
                margin=0.05,
            )

            pred_loss_scalar = float(pred_loss.detach().cpu().item())
            if baseline is None:
                baseline = pred_loss_scalar
            else:
                baseline = (
                    self.reinforce_baseline_momentum * baseline
                    + (1.0 - self.reinforce_baseline_momentum) * pred_loss_scalar
                )

            reinforce_loss = (pred_loss.detach() - baseline) * total_logprob
            entropy_reg = -total_entropy

            meta_term = (self.meta_coeff * loss_meta) if self.use_meta_loss else torch.tensor(0.0, device=self.device)

            total_loss = (
                self.pred_coeff * reinforce_loss
                + meta_term
                + self.preserve_coeff * loss_preserve
                + self.edge_preserve_coeff * loss_edge_preserve
                + self.spar_coeff * loss_spar
                + self.ent_coeff * entropy_reg
            )

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                eval_soft_mask_dict = mask_module.get_mask_dict(mode="sigmoid")
                eval_soft_keep_ratio = self._compute_global_soft_keep_ratio(eval_soft_mask_dict)

                eval_hard_mask_dict = {
                    etype: (m > 0.5).float() for etype, m in eval_soft_mask_dict.items()
                }
                eval_edge_index_dict = self._build_masked_edge_index_dict(local_edge_index_dict, eval_hard_mask_dict)
                eval_local_data = self._build_local_data(subgraph_info, eval_edge_index_dict)
                eval_pred_info = self._eval_local_prediction(eval_local_data, target_local_id)

                eval_probs = eval_pred_info["probs"]
                eval_pred = eval_pred_info["pred"]

                eval_pred_loss = float(
                    self._prediction_loss_from_probs(
                        eval_probs,
                        orig_class=orig_class,
                        target_class=target_class,
                        margin=0.05,
                    ).detach().cpu().item()
                )

                eval_macro_loss, eval_preserve_loss, eval_key_ratios, eval_nonkey_ratios = self._compute_macro_losses_from_cache(
                    valid_metapaths=valid_metapaths,
                    key_metapaths=key_metapaths,
                    soft_mask_dict=eval_soft_mask_dict,
                    metapath_instance_cache=metapath_instance_cache,
                    meta_index_map=meta_index_map,
                )
                if self.use_meta_loss:
                    eval_macro_loss = float(eval_macro_loss.detach().cpu().item())
                else:
                    eval_macro_loss = 0.0
                eval_preserve_loss = float(eval_preserve_loss.detach().cpu().item())
                if self.uniform_edge_perturbation:
                    eval_edge_preserve_loss = 0.0
                else:
                    eval_edge_preserve_loss = float(
                        self._compute_edge_preserve_loss(
                            soft_mask_dict=eval_soft_mask_dict,
                            nonkey_edge_masks=nonkey_edge_masks,
                        ).detach().cpu().item()
                    )

                if self.use_key_sparsity and (not self.uniform_edge_perturbation):
                    eval_spar_loss = float(
                        self._compute_key_sparsity_loss(
                            soft_mask_dict=eval_soft_mask_dict,
                            key_edge_sets=key_edge_sets,
                        ).detach().cpu().item()
                    )
                else:
                    eval_spar_loss = 0.0

                if self.semantic_selection:
                    eval_score = (
                        self.meta_coeff * eval_macro_loss
                        + self.preserve_coeff * eval_preserve_loss
                        + self.edge_preserve_coeff * eval_edge_preserve_loss
                        + self.spar_coeff * eval_spar_loss
                        + 0.1 * eval_pred_loss
                    )
                else:
                    eval_score = eval_pred_loss * 0.1

                eval_flipped = eval_pred != orig_class

                if eval_flipped:
                    if (not flipped) or (eval_score < best_score):
                        flipped = True
                        best_score = eval_score
                        best_epoch = epoch
                        best_probs = eval_probs.detach().cpu()
                        best_pred = int(eval_pred)
                        best_key_ratios = eval_key_ratios
                        best_nonkey_ratios = eval_nonkey_ratios
                        best_soft_keep_ratio = eval_soft_keep_ratio
                        best_hard_mask_dict = {
                            k: v.detach().cpu() for k, v in eval_hard_mask_dict.items()
                        }
                else:
                    if (not flipped) and (eval_score < best_score):
                        best_score = eval_score
                        best_epoch = epoch
                        best_probs = eval_probs.detach().cpu()
                        best_pred = int(eval_pred)
                        best_key_ratios = eval_key_ratios
                        best_nonkey_ratios = eval_nonkey_ratios
                        best_soft_keep_ratio = eval_soft_keep_ratio
                        best_hard_mask_dict = {
                            k: v.detach().cpu() for k, v in eval_hard_mask_dict.items()
                        }

            if self.verbose and (epoch % 20 == 0 or epoch == 1 or epoch == self.epochs):
                print(
                    f"Epoch {epoch:03d} | "
                    f"reinforce={float(reinforce_loss.detach().cpu().item()):.6f} | "
                    f"pred_loss={pred_loss_scalar:.6f} | "
                    f"meta={float(loss_meta.detach().cpu().item()):.6f} | "
                    f"preserve={float(loss_preserve.detach().cpu().item()):.6f} | "
                    f"edge_pres={float(loss_edge_preserve.detach().cpu().item()):.6f} | "
                    f"key_spar={float(loss_spar.detach().cpu().item()):.6f} | "
                    f"sampled_pred={masked_pred} | "
                    f"best_pred={best_pred}"
                )

        if best_hard_mask_dict is None:
            best_hard_mask_dict = {
                etype: torch.ones(edge_index.size(1))
                for etype, edge_index in local_edge_index_dict.items()
            }

        kept_edges = {}
        total_edges = {}
        for etype, edge_index in local_edge_index_dict.items():
            total_edges[etype] = int(edge_index.size(1))
            kept_edges[etype] = int(best_hard_mask_dict[etype].sum().item())

        kept_mask_out = {}
        for etype, edge_index in local_edge_index_dict.items():
            mask_tensor = best_hard_mask_dict[etype]
            mask_list = [int(x) for x in mask_tensor.detach().cpu().tolist()]
            kept_mask_out[etype] = mask_list

        key_edge_sets_out = {etype: sorted(list(s)) for etype, s in key_edge_sets.items()}
        try:
            kept_cf_mask = {}
            deleted_expl_mask = {}

            for etype, edge_index in local_edge_index_dict.items():
                final_mask = best_hard_mask_dict[etype].to(self.device).float()
                kept_cf_mask[etype] = final_mask
                deleted_expl_mask[etype] = 1.0 - final_mask

            kept_cf_edge_index = self._build_masked_edge_index_dict(
                local_edge_index_dict, kept_cf_mask
            )
            kept_cf_local_data = self._build_local_data(subgraph_info, kept_cf_edge_index)
            kept_cf_pred = self._eval_local_prediction(kept_cf_local_data, target_local_id)

            deleted_expl_edge_index = self._build_masked_edge_index_dict(
                local_edge_index_dict, deleted_expl_mask
            )
            deleted_expl_local_data = self._build_local_data(subgraph_info, deleted_expl_edge_index)
            deleted_expl_pred = self._eval_local_prediction(deleted_expl_local_data, target_local_id)

            p_full_orig = float(orig_probs[orig_class].detach().cpu().item())
            p_kept_cf_orig = float(
                kept_cf_pred["probs"][orig_class].detach().cpu().item()
            )
            p_deleted_expl_orig = float(
                deleted_expl_pred["probs"][orig_class].detach().cpu().item()
            )

            fidelity_plus = p_full_orig - p_kept_cf_orig
            fidelity_minus = p_deleted_expl_orig
        except Exception:
            fidelity_plus = float("nan")
            fidelity_minus = float("nan")

        return CFExplainResult(
            target_node_id=target_node_id,
            target_node_type=self.target_node_type,
            orig_class=int(orig_class),
            target_class=int(target_class),
            orig_probs=orig_probs.detach().cpu().tolist(),
            best_probs=best_probs.tolist(),
            best_pred=int(best_pred),
            flipped=bool(flipped),
            best_epoch=int(best_epoch),
            best_score=float(best_score),
            key_metapaths=key_metapaths,
            key_ratios=best_key_ratios,
            nonkey_ratios=best_nonkey_ratios,
            key_sensitivities=key_sensitivities,
            kept_edges=kept_edges,
            total_edges=total_edges,
            soft_keep_ratio=float(best_soft_keep_ratio),
            key_edge_sets=key_edge_sets_out,
            kept_mask=kept_mask_out,
            fidelity_plus=fidelity_plus,
            fidelity_minus=fidelity_minus,
        )