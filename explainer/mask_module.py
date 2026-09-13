from typing import Dict, Tuple, Any, Optional

import torch
import torch.nn as nn
from explainer import _verbosity as _v


EdgeType = Tuple[str, str, str]


class EdgeMaskModule(nn.Module):

    def __init__(
        self,
        local_edge_index_dict: Dict[EdgeType, torch.Tensor],
        init_strategy: str = "zeros",
        init_value: float = 0.0,
        active_edge_mask: Optional[Dict[EdgeType, torch.Tensor]] = None,
    ):
        super().__init__()

        self.edge_types = list(local_edge_index_dict.keys())


        self.edge_mask_logits = nn.ParameterDict()
        self._trainable_index_map = {}  # etype_key -> list of trainable indices
        self._num_edges_map = {}

        for etype, edge_index in local_edge_index_dict.items():
            etype_key = self._etype_to_key(etype)
            num_edges = edge_index.size(1)
            self._num_edges_map[etype_key] = int(num_edges)

            active_mask = None
            if active_edge_mask is not None:
                active_mask = active_edge_mask.get(etype, None)

            if active_mask is None:
                if init_strategy == "zeros":
                    init_tensor = torch.zeros(num_edges, dtype=torch.float32)
                elif init_strategy == "constant":
                    init_tensor = torch.full((num_edges,), float(init_value), dtype=torch.float32)
                else:
                    raise ValueError(f"Unsupported init_strategy: {init_strategy}")
                self.edge_mask_logits[etype_key] = nn.Parameter(init_tensor)
                self._trainable_index_map[etype_key] = None  # None means full
            else:
                am = active_mask.to(torch.bool)
                if am.numel() != num_edges:
                    raise ValueError(f"active_edge_mask for {etype} has incorrect length")
                trainable_indices = torch.where(am)[0].tolist()
                self._trainable_index_map[etype_key] = trainable_indices
                if len(trainable_indices) == 0:
                    self.edge_mask_logits[etype_key] = nn.Parameter(torch.empty((0,), dtype=torch.float32))
                else:
                    if init_strategy == "zeros":
                        init_tensor = torch.zeros(len(trainable_indices), dtype=torch.float32)
                    elif init_strategy == "constant":
                        init_tensor = torch.full((len(trainable_indices),), float(init_value), dtype=torch.float32)
                    else:
                        raise ValueError(f"Unsupported init_strategy: {init_strategy}")
                    self.edge_mask_logits[etype_key] = nn.Parameter(init_tensor)

    @staticmethod
    def _etype_to_key(etype: EdgeType) -> str:
        return f"{etype[0]}__{etype[1]}__{etype[2]}"

    @staticmethod
    def _key_to_etype(key: str) -> EdgeType:
        src, rel, dst = key.split("__")
        return (src, rel, dst)

    def reset_parameters(self, init_strategy, init_value):
        for _, param in self.edge_mask_logits.items():
            if init_strategy == "zeros":
                nn.init.constant_(param, 0.0)
            elif init_strategy == "constant":
                nn.init.constant_(param, float(init_value))
            else:
                raise ValueError(f"Unsupported init_strategy: {init_strategy}")

    def get_mask_logits_dict(self):
        out = {}
        for key, param in self.edge_mask_logits.items():
            etype = self._key_to_etype(key)
            train_map = self._trainable_index_map.get(key, None)
            if train_map is None:
                logits = param
            else:
                total_num = self._num_edges_map.get(key, 0)
                logits = torch.full((total_num,), 10.0, dtype=param.dtype, device=param.device)
                if param.numel() > 0:
                    for i, edge_idx in enumerate(train_map):
                        logits[edge_idx] = param[i]
            out[etype] = logits
        return out

    def get_mask_dict(
        self,
        mode,
        tau,
        training,
        hard,
    ):
        if training is None:
            training = self.training

        out = {}

        for key, logits in self.edge_mask_logits.items():
            etype = self._key_to_etype(key)
            train_map = self._trainable_index_map.get(key, None)

            if train_map is None:
                logits_full = logits
            else:
                total_num = self._num_edges_map.get(key, 0)
                if total_num == 0:
                    logits_full = torch.full((0,), 10.0, device=logits.device)
                else:
                    logits_full = torch.full((total_num,), 10.0, device=logits.device)
                    for i, edge_idx in enumerate(train_map):
                        logits_full[edge_idx] = logits[i]

            if mode == "sigmoid":
                mask = torch.sigmoid(logits_full)
            elif mode == "gumbel_sigmoid":
                mask = self._gumbel_sigmoid(logits_full, tau=tau, training=training, hard=hard)
            else:
                raise ValueError(f"Unsupported mask mode: {mode}")

            out[etype] = mask

        return out

    def build_edge_weight_dict(
        self,
        mode,
        tau,
        training,
        hard,
    ):

        return self.get_mask_dict(mode=mode, tau=tau, training=training, hard=hard)

    def forward(
        self,
        mode,
        tau,
        training,
        hard,
    ) -> Dict[EdgeType, torch.Tensor]:
        return self.build_edge_weight_dict(
            mode=mode,
            tau=tau,
            training=training,
            hard=hard,
        )

    def trainable_parameters(self):
        for key, param in self.edge_mask_logits.items():
            if param.numel() > 0:
                yield param

    @staticmethod
    def _sample_gumbel(shape, device, eps: float = 1e-20):
        u = torch.rand(shape, device=device)
        return -torch.log(-torch.log(u + eps) + eps)

    def _gumbel_sigmoid(
        self,
        logits,
        tau,
        training,
        hard,
    ):
        if training:
            g1 = self._sample_gumbel(logits.shape, logits.device)
            g2 = self._sample_gumbel(logits.shape, logits.device)
            logistic_noise = g1 - g2
            y = torch.sigmoid((logits + logistic_noise) / tau)
        else:
            y = torch.sigmoid(logits)

        if hard:
            y_hard = (y > 0.5).float()
            y = y_hard.detach() - y.detach() + y

        return y

    def get_mask_statistics(
        self,
        mode,
        tau,
        training,
        hard
    ):
        mask_dict = self.get_mask_dict(mode=mode, tau=tau, training=training, hard=hard)

        total_edges = 0
        total_mask_sum = 0.0
        per_type_stats = {}

        for etype, mask in mask_dict.items():
            num_edges = mask.numel()
            mask_sum = float(mask.sum().item())
            mask_mean = float(mask.mean().item()) if num_edges > 0 else 0.0

            total_edges += num_edges
            total_mask_sum += mask_sum

            per_type_stats[etype] = {
                "num_edges": num_edges,
                "mask_sum": mask_sum,
                "mask_mean": mask_mean,
                "mask_min": float(mask.min().item()) if num_edges > 0 else 0.0,
                "mask_max": float(mask.max().item()) if num_edges > 0 else 0.0,
            }

        global_mean = total_mask_sum / max(total_edges, 1)

        return {
            "total_edges": total_edges,
            "total_mask_sum": total_mask_sum,
            "global_mask_mean": global_mean,
            "per_type_stats": per_type_stats,
        }


def summarize_mask_statistics(stats):

    if _v.VERBOSE:
        print("=" * 80)
        print("Edge Mask Statistics")
        print("=" * 80)
        print(f"Total edges       : {stats['total_edges']}")
        print(f"Total mask sum    : {stats['total_mask_sum']:.6f}")
        print(f"Global mask mean  : {stats['global_mask_mean']:.6f}")
        print("-" * 80)

        for etype, item in stats["per_type_stats"].items():
            print(
                f"{etype}: "
                f"num_edges={item['num_edges']}, "
                f"mean={item['mask_mean']:.6f}, "
                f"min={item['mask_min']:.6f}, "
                f"max={item['mask_max']:.6f}"
            )
        print("=" * 80)