from __future__ import annotations
import os, sys, time
from datetime import datetime
from typing import List
import numpy as np
import torch

import pred_maskpredition_GWN_scpt_geo as est
from topo_moe_utils import SparseExpertFusion, add_moe_regularization, build_topology_embedding, default_moe_context, dense_adjacency_from_supports, save_alpha_report

_ORIG_GET_ARGV = est.get_argv
_ORIG_TRAIN = est.trainModel_estimation_with_pretrain
_ORIG_TEST = est.testModel_estimation_with_pretrain
MOE_MODES = {'sparse_moe', 'topo_moe'}


def _bool(x): return bool(int(x)) if not isinstance(x, bool) else x

def _arg(name, idx, default, cast):
    if os.environ.get(name, '') != '': return cast(os.environ[name])
    return cast(sys.argv[idx]) if len(sys.argv) > idx else default


_EXPERT_ALIASES = {
    'temporal': 'temporal', 'temp': 'temporal', 'tmp': 'temporal', 'scpt': 'temporal',
    'geometric': 'geometric', 'geo': 'geometric', 'geom': 'geometric',
    'topology': 'topology', 'topo': 'topology', 'lap': 'topology', 'laplacian': 'topology',
}


def _parse_experts(raw):
    parts = str(raw).replace('+', ',').replace('|', ',').split(',')
    out = []
    for p in parts:
        key = p.strip().lower()
        if not key:
            continue
        if key not in _EXPERT_ALIASES:
            raise ValueError(f'Unknown MOE expert {p!r}. Use temporal/geometric/topology, or aliases scpt/geo/topo.')
        val = _EXPERT_ALIASES[key]
        if val not in out:
            out.append(val)
    if not out:
        raise ValueError('MOE_EXPERTS is empty. Example: MOE_EXPERTS=temporal,geometric,topology')
    return out


def _expert_arg(default_spec, argv_idx):
    raw = os.environ.get('MOE_EXPERTS', '').strip()
    if not raw and len(sys.argv) > argv_idx:
        raw = sys.argv[argv_idx].strip()
    if not raw:
        raw = default_spec
    return _parse_experts(raw)



def _fmt_tag_float(x):
    try:
        out = ("%.4g" % float(x))
    except Exception:
        out = str(x)
    return out.replace('-', 'm').replace('.', 'p')


def _safe_tag(s):
    keep = []
    for ch in str(s):
        keep.append(ch if ch.isalnum() or ch in ('_', '-', '.') else '_')
    tag = ''.join(keep).strip('._-')
    return tag or 'run'


def _apply_moe_run_tag(P, task_prefix):
    """Put every MoE setting in its own output directory to avoid overwriting checkpoints/results.

    Environment variables:
      MOE_RUN_TAG=custom_name     explicit tag; recommended for sweeps
      MOE_AUTO_TAG=0              disable automatic subfolder tagging
    """
    explicit = os.environ.get('MOE_RUN_TAG', '').strip()
    auto = os.environ.get('MOE_AUTO_TAG', '0').strip() != '0'
    if explicit:
        tag = explicit
    elif auto:
        tag = est.run_keyword()
    else:
        tag = ''
    # NOTE: The original base scripts set P.PATH later inside main().
    # When called during get_argv(), P.PATH may not exist yet.
    if not hasattr(P, 'PATH'):
        P.MOE_BASE_PATH = None
        P.MOE_RUN_TAG = _safe_tag(tag) if tag else ''
        return
    P.MOE_BASE_PATH = P.PATH
    if tag:
        P.MOE_RUN_TAG = _safe_tag(tag)
        P.PATH = os.path.join(P.PATH, P.MOE_RUN_TAG)
        os.makedirs(P.PATH, exist_ok=True)
        print(f'[TopoMoE] output dir: {P.PATH}')
    else:
        P.MOE_RUN_TAG = ''

def get_argv_topomoe_estimation():
    _ORIG_GET_ARGV()
    P = est.P
    # original pred_maskpredition_GWN_scpt_geo.py consumes argv[1]..argv[20]
    P.FUSION_MODE = _arg('FUSION_MODE', 21, 'topo_moe', str)
    P.GATE_HIDDEN = _arg('GATE_HIDDEN', 22, getattr(P, 'GATE_HIDDEN', 64), int)
    P.TOPO_LAP_K = _arg('TOPO_LAP_K', 23, 16, int)
    P.MOE_TOP_K = _arg('MOE_TOP_K', 24, 2, int)
    P.MOE_TAU = _arg('MOE_TAU', 25, 1.0, float)
    P.MOE_LB_REG = _arg('MOE_LB_REG', 26, 1e-3, float)
    P.MOE_SMOOTH_REG = _arg('MOE_SMOOTH_REG', 27, 1e-3, float)
    P.MOE_DELTA_REG = _arg('MOE_DELTA_REG', 28, getattr(P, 'DELTA_REG', 0.0), float)
    P.MOE_USE_CTX = _arg('MOE_USE_CTX', 29, 1, _bool)
    default_experts = 'temporal,geometric,topology' if P.FUSION_MODE == 'topo_moe' else 'temporal,geometric'
    P.MOE_EXPERTS = _expert_arg(default_experts, 30)
    P.MOE_FORCE_EXPERT = os.environ.get('MOE_FORCE_EXPERT', '').strip()
    P.MOE_INIT_TEMPORAL_BIAS = float(os.environ.get('MOE_INIT_TEMPORAL_BIAS', '1.0'))
    P.TOPO_FORCE_RECOMPUTE = _bool(os.environ.get('TOPO_FORCE_RECOMPUTE', '0'))
    _apply_moe_run_tag(P, 'est')
    print('[TopoMoE estimation]', {k: getattr(P, k) for k in ['FUSION_MODE','MOE_EXPERTS','GATE_HIDDEN','TOPO_LAP_K','MOE_TOP_K','MOE_TAU','MOE_LB_REG','MOE_SMOOTH_REG','MOE_DELTA_REG','MOE_USE_CTX','MOE_RUN_TAG']})


def _enabled():
    return bool(est.P.IS_PRETRN) and est.P.FUSION_MODE in MOE_MODES


def _A_np():
    return dense_adjacency_from_supports(est.load_adj(est.P.ADJPATH, est.P.ADJTYPE, est.P.DATANAME), symmetrize=True)


def _topo_embed(D):
    P = est.P; A = _A_np()
    path = os.path.join(P.PATH, f'topology_embed_D{D}_lap{P.TOPO_LAP_K}.npz')
    Z = build_topology_embedding(A, embed_dim=D, lap_k=P.TOPO_LAP_K, cache_path=path, force_recompute=P.TOPO_FORCE_RECOMPUTE)
    return torch.tensor(Z, dtype=torch.float32, device=est.device).detach()


def _inputs():
    P = est.P
    # We load the two pretrained branches to keep dimensions identical to the original dual-pretraining path.
    encoder = est.Contrastive_FeatureExtractor_conv(P.TEMPERATURE).to(est.device)
    encoder.load_state_dict(torch.load(est.resolve_artifact_path('pretrain_topo'), map_location=est.device))
    encoderg = est.Geometric_Encoder(P.TEMPERATURE, P.FEATURES, P.GRAPH_NORM, P.HIDDEN).to(est.device)
    encoderg.load_state_dict(torch.load(est.resolve_artifact_path('pretrain_geo'), map_location=est.device))
    temp = est._temporal_full_embed_est(encoder).detach()
    geo = est._geometric_full_embed_est(encoderg).detach()
    A = _A_np()

    pool = {
        'temporal': temp,
        'geometric': geo,
    }
    if 'topology' in P.MOE_EXPERTS:
        pool['topology'] = _topo_embed(temp.shape[0])

    experts, names = [], []
    for name in P.MOE_EXPERTS:
        if name not in pool:
            raise ValueError(f'Expert {name!r} is unavailable. Available experts: {sorted(pool)}')
        experts.append(pool[name])
        names.append(name)

    ctx = torch.tensor(default_moe_context(A, has_temporal=('temporal' in names)), dtype=torch.float32, device=est.device) if P.MOE_USE_CTX else None
    adj_dense = torch.tensor(A, dtype=torch.float32, device=est.device)
    return experts, names, ctx, adj_dense


def _make(experts, names, ctx):
    P = est.P
    return SparseExpertFusion(embed_dim=experts[0].shape[0], n_experts=len(experts), ctx_dim=0 if ctx is None else ctx.shape[1],
                              hidden_dim=P.GATE_HIDDEN, top_k=P.MOE_TOP_K, temperature=P.MOE_TAU,
                              expert_names=names, init_temporal_bias=P.MOE_INIT_TEMPORAL_BIAS).to(est.device)


def _run(fusion, experts, ctx, adj_dense, return_aux=True):
    P = est.P
    avail = None
    if P.MOE_FORCE_EXPERT:
        if P.MOE_FORCE_EXPERT not in fusion.expert_names:
            raise ValueError(f'MOE_FORCE_EXPERT={P.MOE_FORCE_EXPERT} not in {fusion.expert_names}')
        idx = fusion.expert_names.index(P.MOE_FORCE_EXPERT)
        avail = torch.zeros((experts[0].shape[1], len(experts)), dtype=torch.float32, device=est.device); avail[:, idx] = 1.0
    smooth_adj = adj_dense if P.MOE_SMOOTH_REG > 0 else None
    return fusion(experts, ctx=ctx, avail_mask=avail, adj_dense=smooth_adj, return_aux=return_aux)


def trainModel_estimation_with_pretrain_topomoe(name, train_iter, val_u_iter, val_a_iter, adj_train, adj_val_u, adj_val_a, spatialSplit_unseen, spatialSplit_allNod, pretrn_iterg, preval_iterg):
    if not _enabled():
        return _ORIG_TRAIN(name, train_iter, val_u_iter, val_a_iter, adj_train, adj_val_u, adj_val_a, spatialSplit_unseen, spatialSplit_allNod, pretrn_iterg, preval_iterg)
    P = est.P
    print('trainModel Estimation with Sparse/Topo MoE Started ...', time.ctime())
    model = est.getModel(name, est.device)
    criterion = est.masked_loss
    experts, names, ctx, adj_dense = _inputs()
    fusion = _make(experts, names, ctx)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(fusion.parameters()), lr=P.LEARN, weight_decay=P.weight_decay)
    with torch.no_grad():
        fe0, aux0 = _run(fusion, experts, ctx, adj_dense, True)
    train_embed = fe0[:, spatialSplit_unseen.i_trn]
    print('experts', names, 'initial alpha mean', aux0['alpha'].mean(0).detach().cpu().numpy())
    min_val = float('inf'); s_time = datetime.now()
    for epoch in range(P.EPOCH):
        st = datetime.now(); model.train(); fusion.train(); epoch_loss = 0.0; n = 0; batch_times: List[float] = []
        # same as original: regenerate training mask every epoch
        trainXS_ep, trainYS_ep, trainMS_ep = est.getXSYS_estimation(est.data, 'TRAIN', missing_ratio=P.MISS_RATIO)
        if P.IS_DESEASONED:
            trainXS_ds_ep, _, trainMS_ds_ep = est.getXSYS_estimation(est.data_ds, 'TRAIN', missing_ratio=P.MISS_RATIO)
            trainXS_ep = np.concatenate((trainXS_ep, trainXS_ds_ep), axis=1)
            trainMS_ep = np.concatenate((trainMS_ep, trainMS_ds_ep), axis=1)
        XS_ep = trainXS_ep[:P.train_size][:, :, spatialSplit_unseen.i_trn, :]
        YS_ep = trainYS_ep[:P.train_size][:, :, spatialSplit_unseen.i_trn, :]
        MS_ep = trainMS_ep[:P.train_size][:, :, spatialSplit_unseen.i_trn, :]
        ds = torch.utils.data.TensorDataset(torch.Tensor(XS_ep), torch.Tensor(YS_ep), torch.Tensor(MS_ep))
        it = torch.utils.data.DataLoader(ds, P.BATCHSIZE, shuffle=True, num_workers=8, pin_memory=True if est.device.type == 'cuda' else False)
        for x, y, mask in it:
            tb = time.perf_counter(); optimizer.zero_grad()
            x = x.to(est.device, non_blocking=True)
            y = y.to(est.device, non_blocking=True).squeeze(-1)
            mask = mask.to(est.device, non_blocking=True)
            full, aux = _run(fusion, experts, ctx, adj_dense, True)
            train_embed = full[:, spatialSplit_unseen.i_trn]
            pred = model(x, adj_train, train_embed)
            task_loss = criterion(pred, y, mask)
            loss = add_moe_regularization(task_loss, aux, P)
            loss.backward(); optimizer.step()
            epoch_loss += task_loss.item(); n += 1; batch_times.append(time.perf_counter() - tb)
        train_loss = epoch_loss / max(n, 1)
        model.eval(); fusion.eval()
        with torch.no_grad():
            full, aux_eval = _run(fusion, experts, ctx, adj_dense, True)
            train_embed = full[:, spatialSplit_unseen.i_trn]
            val_u_embed = full[:, spatialSplit_unseen.i_val]
            val_a_embed = full[:, spatialSplit_allNod.i_val]
            val_u_loss = est.evaluateModel_estimation_with_pretrain(model, val_u_iter, val_u_embed, adj_val_u)
            val_a_loss = est.evaluateModel_estimation_with_pretrain(model, val_a_iter, val_a_embed, adj_val_a)
        if val_u_loss < min_val:
            min_val = val_u_loss
            torch.save(model.state_dict(), est.artifact_path('best'))
            torch.save(fusion.state_dict(), est.artifact_path('fusion_u'))
            save_alpha_report(P.PATH, 'est_u', aux_eval['alpha'], names)
        am = aux_eval['alpha'].mean(0).detach().cpu().numpy()
        print(f'Epoch {epoch}, Time {(datetime.now()-st).seconds}s, Train Loss: {train_loss:.6f}, Val_U Loss: {val_u_loss:.6f}, Val_A Loss: {val_a_loss:.6f}, alpha_mean: {am}, lb={aux_eval["load_balance_loss"].item():.4f}, smooth={aux_eval["smooth_loss"].item():.4f}, delta={aux_eval["delta_l2"].item():.4f}')
        with open(est.artifact_path('train_log'), 'a') as f:
            f.write(f'epoch,{epoch},train_loss,{train_loss:.10f},val_u_loss,{val_u_loss:.10f},val_a_loss,{val_a_loss:.10f},alpha_mean,{am.tolist()}\n')
    print('TRAINING FINISHED. Best val_u loss:', min_val, 'duration:', datetime.now() - s_time)
    score = est.evaluateModel_estimation_with_pretrain(model, train_iter, train_embed, adj_train)
    with open(est.artifact_path('prediction_scores'), 'a') as f:
        f.write(f'{name}, estimation, MAE on train, {score:.10f}, {score:.10f}\n')


def testModel_estimation_with_pretrain_topomoe(name, mode, test_iter, node_indices, scaler, adj_tst_u, mapping, spatialsplit):
    if not _enabled():
        return _ORIG_TEST(name, mode, test_iter, node_indices, scaler, adj_tst_u, mapping, spatialsplit)
    P = est.P
    print('Model Testing Estimation with Sparse/Topo MoE', mode, 'Started ...', time.ctime())
    model = est.getModel(name, est.device)
    model.load_state_dict(torch.load(est.resolve_artifact_path('best'), map_location=est.device)); model.eval()
    experts, names, ctx, adj_dense = _inputs()
    fusion = _make(experts, names, ctx)
    ckpt = est.resolve_artifact_path('fusion_u')
    if os.path.exists(ckpt): fusion.load_state_dict(torch.load(ckpt, map_location=est.device))
    else: print('fusion checkpoint not found:', ckpt)
    fusion.eval()
    with torch.no_grad():
        full, aux = _run(fusion, experts, ctx, adj_dense, True)
        node_embed = full[:, node_indices]
        save_alpha_report(P.PATH, mode, aux['alpha'], names)
    print('alpha mean', aux['alpha'].mean(0).detach().cpu().numpy(), 'experts', names)
    adj_tst_u = [a.to(est.device) for a in adj_tst_u]
    Y_true_list, Y_pred_list, MISS_list = [], [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for x, y, ms in test_iter:
            x = x.to(est.device, non_blocking=True); y = y.to(est.device, non_blocking=True).squeeze(-1); ms = ms.to(est.device, non_blocking=True)
            pred = model(x, adj_tst_u, node_embed)
            Y_pred_list.append(pred.permute(0, 2, 1).cpu().numpy())
            Y_true_list.append(y.permute(0, 2, 1).cpu().numpy())
            MISS_list.append((1.0 - ms[:, 0, :, :]).cpu().numpy())
    print(f'{name}, {mode}, INFERENCE_WALL_SEC, {time.perf_counter() - t0:.6f}')
    Y_true = np.concatenate(Y_true_list, 0); Y_pred = np.concatenate(Y_pred_list, 0); MISS = np.concatenate(MISS_list, 0)
    def inv3(arr, scaler_obj):
        B, N, T = arr.shape
        if hasattr(scaler_obj, 'mean_') and np.ndim(getattr(scaler_obj, 'mean_')) == 1:
            flat = arr.transpose(0, 2, 1).reshape(-1, N)
            flat = scaler_obj.inverse_transform(flat)
            return flat.reshape(B, T, N).transpose(0, 2, 1)
        return scaler_obj.inverse_transform(arr)
    Y_true = inv3(Y_true, scaler); Y_pred = inv3(Y_pred, scaler)
    eps = 1e-6
    MAE = (np.abs(Y_true - Y_pred) * MISS).sum() / (MISS.sum() + eps)
    RMSE = np.sqrt((((Y_true - Y_pred) ** 2) * MISS).sum() / (MISS.sum() + eps))
    MAPE = (np.abs((Y_true - Y_pred) / (np.abs(Y_true) + eps)) * MISS).sum() / (MISS.sum() + eps)
    print('*' * 40); print(f'{name}, {mode}, Masked MAE: {MAE:.6f}, RMSE: {RMSE:.6f}, MAPE: {MAPE:.6f}')
    np.save(est.mode_npy_path(mode, 'prediction'), Y_pred)
    np.save(est.mode_npy_path(mode, 'groundtruth'), Y_true)
    np.save(est.mode_npy_path(mode, 'missmask'), MISS)
    with open(est.artifact_path('prediction_scores'), 'a') as f:
        f.write(f'{name}, {mode}, Masked MAE, {MAE:.10f}, RMSE, {RMSE:.10f}, MAPE, {MAPE:.10f}\n')


est.get_argv = get_argv_topomoe_estimation
est.trainModel_estimation_with_pretrain = trainModel_estimation_with_pretrain_topomoe
est.testModel_estimation_with_pretrain = testModel_estimation_with_pretrain_topomoe

if __name__ == '__main__':
    est.main()
