"""TopoMoE estimation with splitmask test policy (baseline TST_V_FRAC=0.3).

Policy:
  - TRAIN/VAL: point-wise random masks only.
  - TEST tst_v_full: round(TST_V_FRAC * |i_tst|) nodes from i_tst (default 30%),
    full-graph forward, eval nodes 100% masked at all timesteps.
  - TEST tst_a: all-node test split, random point masks.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Optional

import numpy as np
import torch

import pred_maskpredition_GWN_scpt_geo as est
import unseen_nodes

DEFAULT_TST_V_FRAC = 0.3


def _save_policy(path_dir: str, policy: dict) -> None:
    os.makedirs(path_dir, exist_ok=True)
    if os.environ.get("FORECASTING_EVAL_DIR", "").strip():
        frac = os.environ.get("TST_V_FRAC", "").strip()
        tag = "full" if frac and frac.lower() in {"full", "all", "shared", "1", "1.0", "100", "100%"} else (
            frac.replace(".", "p") if frac else f"default{str(DEFAULT_TST_V_FRAC).replace('.', 'p')}"
        )
        fname = f"mask_policy_frac{tag}_eval.json"
    else:
        fname = "mask_policy.json"
    out = os.path.join(path_dir, fname)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2, ensure_ascii=False)
    print(f"[splitmask] saved {out}")


def _split_istst_by_frac(pool_tst: list[int], frac: float, seed: int) -> list[int]:
    """Pick round(frac * |i_tst|) nodes from i_tst for tst_v_full."""
    n_total = len(pool_tst)
    if frac >= 1.0 or frac <= 0:
        return list(pool_tst)
    n_v = max(1, min(n_total, round(n_total * frac)))
    if n_v >= n_total:
        return list(pool_tst)
    rng = np.random.default_rng(int(seed) + 4242)
    shuffled = list(pool_tst)
    rng.shuffle(shuffled)
    return sorted(shuffled[:n_v])


def _parse_tst_v_frac(raw: str) -> float:
    low = raw.lower()
    if low in {"full", "all", "shared", "1", "1.0", "100", "100%"}:
        return 1.0
    if raw.endswith("%"):
        return float(raw[:-1]) / 100.0
    v = float(raw)
    return v / 100.0 if v > 1.0 else v


def _read_tst_v_frac() -> float:
    raw = os.environ.get("TST_V_FRAC", "").strip()
    if not raw:
        return DEFAULT_TST_V_FRAC
    return _parse_tst_v_frac(raw)


def _init_test_policy(spatial_split_unseen, n_nodes: int) -> dict:
    P = est.P
    pool_tst = sorted(int(i) for i in spatial_split_unseen.i_tst if 0 <= int(i) < n_nodes)
    tst_v_frac_env = os.environ.get("TST_V_FRAC", "").strip()
    tst_v_frac = _read_tst_v_frac()
    tst_v_frac_baseline = not tst_v_frac_env
    split_seed = int(P.seed_SS if P.seed_SS != -1 else P.seed)

    tst_v_nodes = _split_istst_by_frac(pool_tst, tst_v_frac, split_seed)
    P.MASK_POLICY = "train_random_test_istst_frac"
    frac_src = "default" if tst_v_frac_baseline else "TST_V_FRAC"
    note = (
        f"tst_v_full uses {len(tst_v_nodes)}/{len(pool_tst)} i_tst nodes "
        f"({frac_src}={tst_v_frac:.4g}); full-graph forward, eval nodes 100% masked"
    )

    P.TST_V_FRAC = tst_v_frac
    P.TST_V_NODES = tst_v_nodes
    P.TST_EVAL_NODES = tst_v_nodes
    policy = {
        "policy": P.MASK_POLICY,
        "seed": int(P.seed),
        "seed_SS": int(P.seed_SS),
        "n_nodes": int(n_nodes),
        "tst_v_frac": tst_v_frac,
        "tst_v_frac_baseline": bool(tst_v_frac_baseline),
        "miss_ratio_train": float(getattr(P, "MISS_RATIO", 0.2)),
        "train_pool_size": len(spatial_split_unseen.i_trn),
        "test_pool_size": len(pool_tst),
        "istst_total": len(pool_tst),
        "tst_v_nodes": tst_v_nodes,
        "note": note,
    }
    P.MASK_POLICY_JSON = policy
    return policy


def _random_point_mask(x: np.ndarray, m: np.ndarray, ratio: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    mask = rng.random(x.shape) < ratio
    x[mask] = 0.0
    m[mask] = 0.0
    return x, m


def _to_arrays(XS: list[np.ndarray], YS: list[np.ndarray], MS: list[np.ndarray]):
    XS_arr, YS_arr, MS_arr = np.array(XS), np.array(YS), np.array(MS)
    XS_arr = XS_arr[:, :, :, np.newaxis].transpose(0, 3, 2, 1)
    YS_arr = YS_arr[:, :, :, np.newaxis]
    MS_arr = MS_arr[:, :, :, np.newaxis].transpose(0, 3, 2, 1)
    return XS_arr, YS_arr, MS_arr


def getXSYS_estimation(data, mode, missing_ratio=0.2, missing_ratio_test=0.2, epoch: Optional[int] = None):
    P = est.P
    train_num = int(data.shape[0] * P.TRAINRATIO)
    XS, YS, MS = [], [], []
    if mode == "TRAIN":
        rng = np.random.default_rng(int(P.seed) + (int(epoch) if epoch is not None else 0))
        row_range = range(train_num)
        ratio = missing_ratio
    elif mode == "TEST":
        rng = np.random.default_rng(int(P.seed_SS if P.seed_SS != -1 else P.seed) + 99991)
        row_range = range(train_num, data.shape[0])
        ratio = missing_ratio_test
    elif mode == "TEST_VFULL":
        rng = np.random.default_rng(0)
        row_range = range(train_num, data.shape[0])
        ratio = 0.0
    else:
        raise ValueError(f"Unknown mode {mode}")

    full_mask_nodes = set(getattr(P, "TST_EVAL_NODES", []) or []) if mode == "TEST_VFULL" else set()
    for i in row_range:
        x = data[i : i + 1, :].copy()
        y = data[i : i + 1, :].copy()
        m = np.ones_like(x)
        if mode == "TEST_VFULL":
            for node in full_mask_nodes:
                if 0 <= node < x.shape[1]:
                    x[:, node] = 0.0
                    m[:, node] = 0.0
        else:
            x, m = _random_point_mask(x, m, ratio, rng)
        XS.append(x)
        YS.append(y)
        MS.append(m)
    return _to_arrays(XS, YS, MS)


def setups_estimation(missing_ratio=0.2):
    P = est.P
    P.SKIP_TST_U = True
    if not os.path.exists(P.PATH):
        os.makedirs(P.PATH)
    if P.seed_SS == -1:
        P.seed_SS = P.seed
    torch.manual_seed(P.seed)
    torch.cuda.manual_seed(P.seed)
    np.random.seed(P.seed)
    if P.IS_EPOCH_1:
        P.EPOCH = 1
        P.PRETRN_EPOCH = 1
    P.MISS_RATIO = missing_ratio
    print(P.KEYWORD, "data splits (tst_v_full frac baseline)", time.ctime())

    data = est.data
    data_ds = est.data_ds
    spatialSplit_unseen = unseen_nodes.SpatialSplit(data.shape[1], r_trn=P.R_TRN, r_val=0.15, r_tst=0.15, seed=P.seed_SS)
    spatialSplit_allNod = unseen_nodes.SpatialSplit(data.shape[1], r_trn=P.R_TRN, r_val=min(1.0, P.R_TRN * 8 / 7), r_tst=1.0, seed=P.seed_SS)
    policy = _init_test_policy(spatialSplit_unseen, data.shape[1])
    _save_policy(P.PATH, policy)
    print(
        "[splitmask] istst_total", policy["istst_total"],
        "tst_v_frac", policy.get("tst_v_frac"),
        "tst_v_frac_baseline", policy.get("tst_v_frac_baseline"),
        "tst_v_nodes", len(P.TST_V_NODES),
    )

    trainXS, trainYS, trainMS = getXSYS_estimation(data, "TRAIN", missing_ratio=missing_ratio)
    testXS, testYS, testMS = getXSYS_estimation(data, "TEST", missing_ratio=missing_ratio, missing_ratio_test=missing_ratio)
    testVFullXS, testVFullYS, testVFullMS = getXSYS_estimation(data, "TEST_VFULL", missing_ratio=missing_ratio, missing_ratio_test=missing_ratio)
    if P.IS_DESEASONED:
        trainXS_ds, _, trainMS_ds = getXSYS_estimation(data_ds, "TRAIN", missing_ratio=missing_ratio)
        testXS_ds, _, testMS_ds = getXSYS_estimation(data_ds, "TEST", missing_ratio=missing_ratio, missing_ratio_test=missing_ratio)
        testVFullXS_ds, _, testVFullMS_ds = getXSYS_estimation(data_ds, "TEST_VFULL", missing_ratio=missing_ratio, missing_ratio_test=missing_ratio)
        trainXS = np.concatenate((trainXS, trainXS_ds), axis=1)
        testXS = np.concatenate((testXS, testXS_ds), axis=1)
        testVFullXS = np.concatenate((testVFullXS, testVFullXS_ds), axis=1)
        trainMS = np.concatenate((trainMS, trainMS_ds), axis=1)
        testMS = np.concatenate((testMS, testMS_ds), axis=1)
        testVFullMS = np.concatenate((testVFullMS, testVFullMS_ds), axis=1)

    P.trainval_size = len(trainXS)
    P.train_size = int(P.trainval_size * (1 - P.TRAINVALSPLIT))
    XS_torch_trn = trainXS[: P.train_size]
    YS_torch_trn = trainYS[: P.train_size]
    MS_torch_trn = trainMS[: P.train_size]
    XS_torch_val = trainXS[P.train_size :]
    YS_torch_val = trainYS[P.train_size :]
    MS_torch_val = trainMS[P.train_size :]

    all_nodes = np.arange(data.shape[1], dtype=int)

    train_data = torch.utils.data.TensorDataset(
        torch.Tensor(XS_torch_trn[:, :, spatialSplit_unseen.i_trn, :]),
        torch.Tensor(YS_torch_trn[:, :, spatialSplit_unseen.i_trn, :]),
        torch.Tensor(MS_torch_trn[:, :, spatialSplit_unseen.i_trn, :]),
    )
    val_u_data = torch.utils.data.TensorDataset(
        torch.Tensor(XS_torch_val[:, :, spatialSplit_unseen.i_val, :]),
        torch.Tensor(YS_torch_val[:, :, spatialSplit_unseen.i_val, :]),
        torch.Tensor(MS_torch_val[:, :, spatialSplit_unseen.i_val, :]),
    )
    val_a_data = torch.utils.data.TensorDataset(
        torch.Tensor(XS_torch_val[:, :, spatialSplit_allNod.i_val, :]),
        torch.Tensor(YS_torch_val[:, :, spatialSplit_allNod.i_val, :]),
        torch.Tensor(MS_torch_val[:, :, spatialSplit_allNod.i_val, :]),
    )
    tst_v_full_data = torch.utils.data.TensorDataset(
        torch.Tensor(testVFullXS[:, :, all_nodes, :]),
        torch.Tensor(testVFullYS[:, :, all_nodes, :]),
        torch.Tensor(testVFullMS[:, :, all_nodes, :]),
    )
    tst_a_data = torch.utils.data.TensorDataset(
        torch.Tensor(testXS[:, :, spatialSplit_allNod.i_tst, :]),
        torch.Tensor(testYS[:, :, spatialSplit_allNod.i_tst, :]),
        torch.Tensor(testMS[:, :, spatialSplit_allNod.i_tst, :]),
    )

    pin_memory = True if est.device.type == "cuda" else False
    train_iter = torch.utils.data.DataLoader(train_data, P.BATCHSIZE, shuffle=True, num_workers=8, pin_memory=pin_memory)
    val_u_iter = torch.utils.data.DataLoader(val_u_data, P.BATCHSIZE, shuffle=False)
    val_a_iter = torch.utils.data.DataLoader(val_a_data, P.BATCHSIZE, shuffle=False)
    tst_v_full_iter = torch.utils.data.DataLoader(tst_v_full_data, P.BATCHSIZE, shuffle=False)
    tst_a_iter = torch.utils.data.DataLoader(tst_a_data, P.BATCHSIZE, shuffle=False)

    adj_mx = est.load_adj(P.ADJPATH, P.ADJTYPE, P.DATANAME)
    adj_train = [torch.tensor(i[spatialSplit_unseen.i_trn, :][:, spatialSplit_unseen.i_trn]).to(est.device) for i in adj_mx]
    adj_val_u = [torch.tensor(i[spatialSplit_unseen.i_val, :][:, spatialSplit_unseen.i_val]).to(est.device) for i in adj_mx]
    adj_val_a = [torch.tensor(i[spatialSplit_allNod.i_val, :][:, spatialSplit_allNod.i_val]).to(est.device) for i in adj_mx]
    adj_tst_a = [torch.tensor(i[spatialSplit_allNod.i_tst, :][:, spatialSplit_allNod.i_tst]).to(est.device) for i in adj_mx]
    adj_tst_v_full = [torch.tensor(i[all_nodes, :][:, all_nodes]).to(est.device) for i in adj_mx]

    pretrn_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.Tensor(XS_torch_trn[:, -1, spatialSplit_unseen.i_trn, 0]).T),
        P.BATCHSIZE,
        shuffle=True,
    )
    preval_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(trainYS[:, -1, spatialSplit_unseen.i_val, 0]).T.float()),
        P.BATCHSIZE,
        shuffle=False,
    )
    pretrn_iterg = random.sample(list(spatialSplit_unseen.i_trn), P.BATCHSIZE)
    preval_iterg = list(spatialSplit_unseen.i_val)

    P.TST_V_FULL_ITER = tst_v_full_iter
    P.TST_V_FULL_ADJ = adj_tst_v_full
    P.TST_V_FULL_NODES = all_nodes
    P.TST_V_FULL_MAPPING = {int(i): int(i) for i in all_nodes}
    P.TST_V_FULL_SPATIALSPLIT = spatialSplit_unseen

    for k, v in vars(P).items():
        print(k, v)

    return (
        pretrn_iter,
        preval_iter,
        spatialSplit_unseen,
        spatialSplit_allNod,
        train_iter,
        val_u_iter,
        val_a_iter,
        None,
        tst_a_iter,
        adj_train,
        adj_val_u,
        adj_val_a,
        None,
        adj_tst_a,
        {},
        pretrn_iterg,
        preval_iterg,
    )


import pred_maskpredition_GWN_scpt_geo_topomoe as topomoe  # noqa: E402

est.getXSYS_estimation = getXSYS_estimation
est.setups_estimation = setups_estimation
est.get_argv = topomoe.get_argv_topomoe_estimation
est.trainModel_estimation_with_pretrain = topomoe.trainModel_estimation_with_pretrain_topomoe
est.testModel_estimation_with_pretrain = topomoe.testModel_estimation_with_pretrain_topomoe


if __name__ == "__main__":
    est.main()
