from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import HeteroData
import scipy.io as sio


@dataclass
class LoadedHeteroDataset:
    data: HeteroData
    dataset_name: str
    target_node_type: str
    num_classes: int
    candidate_metapaths: Optional[List[List[tuple]]] = None

def _build_split_masks(num_nodes,labeled_idx,train_ratio,val_ratio,seed):
    # if labeled_idx.numel() == 0:
    #     raise ValueError("No labeled nodes found, cannot build train/val/test masks.")
    g = torch.Generator()
    g.manual_seed(seed)
    perm = labeled_idx[torch.randperm(labeled_idx.numel(), generator=g)]

    n = perm.numel()
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = n - n_train - n_val

    if n_test <= 0:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train = max(1, n_train - 1)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def _normalize_existing_mask(mask, num_nodes, mask_name):
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.dim() == 1:
        return mask
    if mask.dim() == 2:
        return mask[:, 0]
    # raise ValueError(f"{mask_name} must be 1D or 2D tensor, but got shape {tuple(mask.shape)}")


def _ensure_or_create_masks(data, target_node_type, seed):
    node_store = data[target_node_type]

    y = node_store.y
    num_nodes = y.size(0)
    has_train = hasattr(node_store, "train_mask")
    has_val = hasattr(node_store, "val_mask")
    has_test = hasattr(node_store, "test_mask")

    if has_train and has_val and has_test:
        node_store.train_mask = _normalize_existing_mask(node_store.train_mask, num_nodes, "train_mask")
        node_store.val_mask = _normalize_existing_mask(node_store.val_mask, num_nodes, "val_mask")
        node_store.test_mask = _normalize_existing_mask(node_store.test_mask, num_nodes, "test_mask")
        return

    if y.dim() == 1:
        labeled_idx = torch.where(y >= 0)[0]
    else:
        labeled_idx = torch.where((y >= 0).any(dim=1))[0]

    train_mask, val_mask, test_mask = _build_split_masks(
        num_nodes=num_nodes,
        labeled_idx=labeled_idx,
        train_ratio=0.6,
        val_ratio=0.2,
        seed=seed,
    )
    node_store.train_mask = train_mask
    node_store.val_mask = val_mask
    node_store.test_mask = test_mask


def _resolve_dataset_dir(root, dataset_name):
    root_path = Path(root).expanduser().resolve()
    candidates = [
        root_path / dataset_name,
        root_path / dataset_name.upper(),
        root_path / dataset_name.lower(),
        root_path / "data" / dataset_name,
        root_path / "data" / dataset_name.upper(),
        root_path / "data" / dataset_name.lower(),
    ]
    for c in candidates:
        if (c / "raw").exists():
            return c

def _load_original_dblp(root):
    ds_dir = _resolve_dataset_dir(root, "DBLP")
    raw_dir = ds_dir / "raw"

    data = HeteroData()
    node_types = ["author", "paper", "term", "conference"]

    # features
    x_author = sp.load_npz(raw_dir / "features_0.npz")
    data["author"].x = torch.from_numpy(x_author.toarray()).float()
    x_paper = sp.load_npz(raw_dir / "features_1.npz")
    data["paper"].x = torch.from_numpy(x_paper.toarray()).float()
    x_term = np.load(raw_dir / "features_2.npy")
    data["term"].x = torch.from_numpy(x_term).float()

    node_type_idx = np.load(raw_dir / "node_types.npy")
    num_conf = int((node_type_idx == 3).sum())
    data["conference"].num_nodes = num_conf
    data["conference"].x = torch.ones(num_conf, 1, dtype=torch.float32)

    y = np.load(raw_dir / "labels.npy")
    data["author"].y = torch.from_numpy(y).long()

    split = np.load(raw_dir / "train_val_test_idx.npz")
    for name in ["train", "val", "test"]:
        idx = torch.from_numpy(split[f"{name}_idx"]).long()
        mask = torch.zeros(data["author"].num_nodes, dtype=torch.bool)
        mask[idx] = True
        data["author"][f"{name}_mask"] = mask

    N_a = data["author"].num_nodes
    N_p = data["paper"].num_nodes
    N_t = data["term"].num_nodes
    N_c = data["conference"].num_nodes
    slices = {
        "author": (0, N_a),
        "paper": (N_a, N_a + N_p),
        "term": (N_a + N_p, N_a + N_p + N_t),
        "conference": (N_a + N_p + N_t, N_a + N_p + N_t + N_c),
    }

    A = sp.load_npz(raw_dir / "adjM.npz")
    for src in node_types:
        for dst in node_types:
            A_sub = A[slices[src][0]:slices[src][1], slices[dst][0]:slices[dst][1]].tocoo()
            if A_sub.nnz > 0:
                row = torch.from_numpy(A_sub.row).long()
                col = torch.from_numpy(A_sub.col).long()
                data[(src, "to", dst)].edge_index = torch.stack([row, col], dim=0)

    _ensure_or_create_masks(data, "author")

    candidate_metapaths = [
        [("author", "to", "paper"), ("paper", "to", "author")],  # APA
        [("author", "to", "paper"), ("paper", "to", "term")], # APT
        [("author", "to", "paper"), ("paper", "to", "conference")], # APC
        [
            ("author", "to", "paper"),
            ("paper", "to", "term"),
            ("term", "to", "paper"),
            ("paper", "to", "author"),
        ],  # APTPA
        [
            ("author", "to", "paper"),
            ("paper", "to", "conference"),
            ("conference", "to", "paper"),
            ("paper", "to", "author"),
        ],  # APCPA
        [
            ("author", "to", "paper"),
            ("paper", "to", "author"),
            ("author", "to", "paper"),
            ("paper", "to", "author"),
        ],  # APAPA
    ]

    num_classes = int(data["author"].y.max().item()) + 1
    return LoadedHeteroDataset(
        data=data,
        dataset_name="dblp",
        target_node_type="author",
        num_classes=num_classes,
        candidate_metapaths=candidate_metapaths,
    )


def _load_original_imdb(root):
    ds_dir = _resolve_dataset_dir(root, "IMDB")
    raw_dir = ds_dir / "raw"

    data = HeteroData()
    node_types = ["movie", "director", "actor"]

    for i, node_type in enumerate(node_types):
        x = sp.load_npz(raw_dir / f"features_{i}.npz")
        data[node_type].x = torch.from_numpy(x.toarray()).float()

    y = np.load(raw_dir / "labels.npy")
    y = torch.from_numpy(y).long()
    data["movie"].y = y

    split = np.load(raw_dir / "train_val_test_idx.npz")
    for name in ["train", "val", "test"]:
        idx = torch.from_numpy(split[f"{name}_idx"]).long()
        mask = torch.zeros(data["movie"].num_nodes, dtype=torch.bool)
        mask[idx] = True
        data["movie"][f"{name}_mask"] = mask

    N_m = data["movie"].num_nodes
    N_d = data["director"].num_nodes
    N_a = data["actor"].num_nodes
    slices = {
        "movie": (0, N_m),
        "director": (N_m, N_m + N_d),
        "actor": (N_m + N_d, N_m + N_d + N_a),
    }

    A = sp.load_npz(raw_dir / "adjM.npz")
    for src in node_types:
        for dst in node_types:
            A_sub = A[slices[src][0]:slices[src][1], slices[dst][0]:slices[dst][1]].tocoo()
            if A_sub.nnz > 0:
                row = torch.from_numpy(A_sub.row).long()
                col = torch.from_numpy(A_sub.col).long()
                data[(src, "to", dst)].edge_index = torch.stack([row, col], dim=0)

    _ensure_or_create_masks(data, "movie")

    candidate_metapaths = [
        [("movie", "to", "actor"), ("actor", "to", "movie")],  # MAM
        [("movie", "to", "director"), ("director", "to", "movie")],  # MDM
        [("movie", "to", "actor"), ('actor', 'to', 'movie'), ("movie", "to", "director"), ("director", "to", "movie")], # MAMDM
        [("movie", "to", "director"), ('director', 'to', 'movie'), ("movie", "to", "actor"), ("actor", "to", "movie")], # MDMAM
        [
            ("movie", "to", "actor"),
            ("actor", "to", "movie"),
            ("movie", "to", "actor"),
            ("actor", "to", "movie"),
        ],  # MAMAM
        [
            ("movie", "to", "director"),
            ("director", "to", "movie"),
            ("movie", "to", "director"),
            ("director", "to", "movie"),
        ]  # MDMDM
    ]

    num_classes = int(data["movie"].y.max().item()) + 1
    return LoadedHeteroDataset(
        data=data,
        dataset_name="imdb",
        target_node_type="movie",
        num_classes=num_classes,
        candidate_metapaths=candidate_metapaths,
    )

def _row_normalized_sparse_average(assign_mat, feature_mat):
    assign_mat = assign_mat.tocsr().astype(np.float32)
    feature_mat = feature_mat.tocsr().astype(np.float32)
    deg = np.asarray(assign_mat.sum(axis=1)).reshape(-1)
    deg[deg == 0] = 1.0
    out = assign_mat @ feature_mat
    out = out.multiply(1.0 / deg[:, None])
    return torch.from_numpy(out.toarray()).float()


def _load_original_acm(root):
    ds_dir = _resolve_dataset_dir(root, "ACM")
    raw_dir = ds_dir / "raw"
    mat_path = raw_dir / "ACM.mat"

    mat = sio.loadmat(mat_path)
    required = ["PvsA", "PvsL", "PvsT", "PvsC"]
    missing = [k for k in required if k not in mat]
    if missing:
        raise KeyError(f"ACM.mat missing keys: {missing}. Existing keys: {sorted(mat.keys())}")

    p_vs_a = mat["PvsA"].tocsr()  # paper-author
    p_vs_l = mat["PvsL"].tocsr()  # paper-field / paper-subject
    p_vs_t = mat["PvsT"].tocsr()  # paper-term BOW features
    p_vs_c = mat["PvsC"].tocsr()  # paper-conference, labels from conferences

    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(axis=1) != 0).A1.nonzero()[0]

    p_vs_a = p_vs_a[p_selected]
    p_vs_l = p_vs_l[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]

    data = HeteroData()
    data["paper"].x = torch.from_numpy(p_vs_t.toarray()).float()
    data["author"].x = _row_normalized_sparse_average(p_vs_a.transpose(), p_vs_t)
    data["field"].x = torch.eye(p_vs_l.shape[1], dtype=torch.float32)

    pc_p, pc_c = p_vs_c.nonzero()
    y = np.zeros(p_vs_c.shape[0], dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        y[pc_p[pc_c == conf_id]] = label_id
    data["paper"].y = torch.from_numpy(y).long()

    rng = np.random.default_rng(42)
    float_mask = np.zeros(len(pc_p), dtype=np.float32)
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        n = int(pc_c_mask.sum())
        if n > 0:
            float_mask[pc_c_mask] = rng.permutation(np.linspace(0, 1, n, dtype=np.float32))

    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    for name, idx_np in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        idx = torch.from_numpy(idx_np).long()
        mask = torch.zeros(data["paper"].num_nodes, dtype=torch.bool)
        mask[idx] = True
        data["paper"][f"{name}_mask"] = mask

    pa = p_vs_a.tocoo()
    data[("paper", "to", "author")].edge_index = torch.stack(
        [torch.from_numpy(pa.row).long(), torch.from_numpy(pa.col).long()], dim=0
    )
    data[("author", "to", "paper")].edge_index = torch.stack(
        [torch.from_numpy(pa.col).long(), torch.from_numpy(pa.row).long()], dim=0
    )

    pf = p_vs_l.tocoo()
    data[("paper", "to", "field")].edge_index = torch.stack(
        [torch.from_numpy(pf.row).long(), torch.from_numpy(pf.col).long()], dim=0
    )
    data[("field", "to", "paper")].edge_index = torch.stack(
        [torch.from_numpy(pf.col).long(), torch.from_numpy(pf.row).long()], dim=0
    )

    _ensure_or_create_masks(data, "paper")

    candidate_metapaths = [
        [("paper", "to", "author"), ("author", "to", "paper")],  # PAP
        [("paper", "to", "field"), ("field", "to", "paper")],    # PFP / PSP
        [("paper", "to", "author"), ("author", "to", "paper"),
         ("paper", "to", "field"), ("field", "to", "paper")],    # PAPFP
        [("paper", "to", "field"), ("field", "to", "paper"),
         ("paper", "to", "author"), ("author", "to", "paper")],  # PFPPAP
        [("paper", "to", "author"), ("author", "to", "paper"),
         ("paper", "to", "author"), ("author", "to", "paper")],  # PAPAP
        [("paper", "to", "field"), ("field", "to", "paper"),
         ("paper", "to", "field"), ("field", "to", "paper")],    # PFPFP
    ]

    num_classes = int(data["paper"].y.max().item()) + 1
    return LoadedHeteroDataset(
        data=data,
        dataset_name="acm",
        target_node_type="paper",
        num_classes=num_classes,
        candidate_metapaths=candidate_metapaths,
    )

def _default_candidate_metapaths(dataset_name: str) -> List[List[tuple]]:
    name = dataset_name.lower()
    if name == "acm":
        return [
            [("paper", "to", "author"), ("author", "to", "paper")],
            [("paper", "to", "subject"), ("subject", "to", "paper")],
            [("paper", "to", "term"), ("term", "to", "paper")],
            [("paper", "cite", "paper")],
            [("paper", "ref", "paper")],
        ]
    if name == "dblp":
        return [
            [("author", "to", "paper"), ("paper", "to", "author")],
            [("author", "to", "paper"), ("paper", "to", "term"), ("term", "to", "paper"), ("paper", "to", "author")],
            [("author", "to", "paper"), ("paper", "to", "conference"), ("conference", "to", "paper"), ("paper", "to", "author")],
        ]
    if name == "imdb":
        return [
            [("movie", "to", "actor"), ("actor", "to", "movie")],
            [("movie", "to", "director"), ("director", "to", "movie")],
            [("movie", "to", "keyword"), ("keyword", "to", "movie")],
        ]
    return []


def _build_reverse_edges_if_needed(data):
    existing_etypes = set(data.edge_types)
    new_edges = []
    for etype in list(data.edge_types):
        src_type, rel_type, dst_type = etype
        rev_etype = (dst_type, f"rev_{rel_type}", src_type)
        if rev_etype in existing_etypes:
            continue
        edge_index = data[etype].edge_index
        rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        new_edges.append((rev_etype, rev_edge_index))
    for rev_etype, rev_edge_index in new_edges:
        data[rev_etype].edge_index = rev_edge_index
    return data


def _normalize_hgb_to_internal_format(data, dataset_name, target_node_type, seed):
    name = dataset_name.lower()
    if name == "acm":
        tgt = target_node_type or "paper"
    elif name == "imdb":
        tgt = target_node_type or "movie"
    elif name == "dblp":
        tgt = target_node_type or "author"

    _ensure_or_create_masks(data, tgt, seed=seed)

    y = data[tgt].y
    if y.dim() == 1:
        valid_y = y[y >= 0]
        # if valid_y.numel() == 0:
        #     raise ValueError(f"Dataset '{dataset_name}' has no valid labels on node type '{tgt}'.")
        num_classes = int(valid_y.max().item()) + 1
    else:
        num_classes = int(y.size(1))

    return LoadedHeteroDataset(
        data=data,
        dataset_name=dataset_name.lower(),
        target_node_type=tgt,
        num_classes=num_classes,
        candidate_metapaths=_default_candidate_metapaths(dataset_name),
    )


def load_builtin_dataset(dataset_name, root, target_node_type, seed):
    from torch_geometric.datasets import HGBDataset

    name = dataset_name.lower()
    root = str(Path(root))

    if name == "acm":
        dataset = HGBDataset(root=root, name="ACM")
    elif name == "dblp":
        dataset = HGBDataset(root=root, name="DBLP")
    elif name == "imdb":
        dataset = HGBDataset(root=root, name="IMDB")

    data = dataset[0]
    data = _build_reverse_edges_if_needed(data)
    return _normalize_hgb_to_internal_format(data, dataset_name=name, target_node_type=target_node_type, seed=seed)


def load_fake_hetero_dataset():
    data = HeteroData()
    data["paper"].x = torch.randn(100, 16)
    data["author"].x = torch.randn(60, 16)
    data["subject"].x = torch.randn(20, 16)
    data["term"].num_nodes = 30
    y = torch.randint(0, 3, (100,))
    data["paper"].y = y
    train_mask = torch.zeros(100, dtype=torch.bool)
    val_mask = torch.zeros(100, dtype=torch.bool)
    test_mask = torch.zeros(100, dtype=torch.bool)
    train_mask[:60] = True
    val_mask[60:80] = True
    test_mask[80:] = True
    data["paper"].train_mask = train_mask
    data["paper"].val_mask = val_mask
    data["paper"].test_mask = test_mask
    return LoadedHeteroDataset(data=data, dataset_name="fake", target_node_type="paper", num_classes=3, candidate_metapaths=[])


def load_dataset(cfg):
    dataset_cfg = cfg["dataset"]
    task_cfg = cfg["task"]

    dataset_name = dataset_cfg["name"].lower()
    root = dataset_cfg["root"]
    target_node_type = task_cfg.get("target_node_type", None)
    dataset_format = dataset_cfg.get("format", "hgb_builtin").lower()

    if dataset_name in {"fake", "toy", "debug"}:
        return load_fake_hetero_dataset()

    if dataset_format in {"hence", "hence_raw", "original", "raw"}:
        if dataset_name == "dblp":
            loaded = _load_original_dblp(root)
        elif dataset_name == "imdb":
            loaded = _load_original_imdb(root)
        elif dataset_name == "acm":
            loaded = _load_original_acm(root)

    else:
        loaded = load_builtin_dataset(
            dataset_name=dataset_name,
            root=root,
            target_node_type=target_node_type,
            seed=cfg.get("seed", 42),
        )

    if loaded.candidate_metapaths is None:
        loaded.candidate_metapaths = _default_candidate_metapaths(dataset_name)
    return loaded


def summarize_hetero_data(loaded):
    data = loaded.data
    print("=" * 80)
    print("Heterogeneous Dataset Summary")
    print("=" * 80)
    print(f"Dataset name      : {loaded.dataset_name}")
    print(f"Target node type  : {loaded.target_node_type}")
    print(f"Num classes       : {loaded.num_classes}")
    print("-" * 80)
    print("[Node Types]")
    for ntype in data.node_types:
        x = getattr(data[ntype], "x", None)
        num_nodes = x.size(0) if x is not None else data[ntype].num_nodes
        feat_dim = x.size(1) if x is not None and x.dim() == 2 else "Unknown"
        print(f"  - {ntype:<12} num_nodes={num_nodes}, feat_dim={feat_dim}")
    print("-" * 80)
    print("[Edge Types]")
    for etype in data.edge_types:
        edge_index = data[etype].edge_index
        print(f"  - {etype}: num_edges={edge_index.size(1)}")
    print("-" * 80)
    print("[Meta-path candidates]")
    for i, mp in enumerate(loaded.candidate_metapaths or []):
        print(f"  - {i}: {mp}")
    tgt = loaded.target_node_type
    print("-" * 80)
    print("[Masks]")
    print(f"  - train: sum={int(data[tgt].train_mask.sum())}, shape={tuple(data[tgt].train_mask.shape)}")
    print(f"  - val  : sum={int(data[tgt].val_mask.sum())}, shape={tuple(data[tgt].val_mask.shape)}")
    print(f"  - test : sum={int(data[tgt].test_mask.sum())}, shape={tuple(data[tgt].test_mask.shape)}")
    print("=" * 80)
