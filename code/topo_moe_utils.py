from __future__ import annotations
import json, os
from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn as nn


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def dense_adjacency_from_supports(supports, symmetrize: bool = True) -> np.ndarray:
    arrs = [_np(s) for s in supports] if isinstance(supports, (list, tuple)) else [_np(supports)]
    n = arrs[0].shape[0]
    A = np.zeros((n, n), dtype=np.float64)
    for a in arrs:
        a = np.nan_to_num(a.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        A = np.maximum(A, np.abs(a))
    np.fill_diagonal(A, 0.0)
    return np.maximum(A, A.T) if symmetrize else A


def _std(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd


def _pagerank(A: np.ndarray, damping: float = 0.85, steps: int = 80) -> np.ndarray:
    n = A.shape[0]
    P = A.copy().astype(np.float64)
    rs = P.sum(1, keepdims=True)
    dangling = rs.squeeze(-1) <= 1e-12
    rs[rs <= 1e-12] = 1.0
    P = P / rs
    pr = np.ones(n) / max(n, 1)
    teleport = np.ones(n) / max(n, 1)
    for _ in range(steps):
        dm = pr[dangling].sum() / max(n, 1) if dangling.any() else 0.0
        pr = damping * (pr @ P + dm) + (1 - damping) * teleport
    return pr


def _lap_pe(A: np.ndarray, k: int) -> np.ndarray:
    n = A.shape[0]
    if k <= 0 or n < 3:
        return np.zeros((n, 0), dtype=np.float64)
    A = np.maximum(A, A.T)
    d = A.sum(1)
    dinv = 1 / np.sqrt(np.maximum(d, 1e-12))
    try:
        if n > 1500:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla
            L = sp.eye(n, format='csr') - sp.diags(dinv) @ sp.csr_matrix(A) @ sp.diags(dinv)
            vals, vecs = spla.eigsh(L, k=min(k + 1, n - 1), which='SM')
            vecs = vecs[:, np.argsort(vals)]
        else:
            L = np.eye(n) - dinv[:, None] * A * dinv[None, :]
            vals, vecs = np.linalg.eigh(L)
            vecs = vecs[:, np.argsort(vals)]
        pe = vecs[:, 1:k + 1]
    except Exception:
        rng = np.random.default_rng(0)
        pe, _ = np.linalg.qr(rng.normal(size=(n, min(k, n))))
    if pe.shape[1] < k:
        pe = np.pad(pe, ((0, 0), (0, k - pe.shape[1])), mode='constant')
    for j in range(pe.shape[1]):
        idx = np.argmax(np.abs(pe[:, j]))
        if pe[idx, j] < 0:
            pe[:, j] *= -1
    return pe


def _nx_feats(A: np.ndarray):
    n = A.shape[0]
    z = np.zeros(n, dtype=np.float64)
    if n > 1200:
        return z, z, z
    try:
        import networkx as nx
        G = nx.from_numpy_array((A > 0).astype(float))
        k = min(64, n) if n > 64 else None
        bt = nx.betweenness_centrality(G, k=k, seed=0, normalized=True)
        cc = nx.clustering(G)
        core = nx.core_number(G) if G.number_of_edges() > 0 else {i: 0 for i in range(n)}
        return (np.array([bt.get(i, 0.0) for i in range(n)]),
                np.array([cc.get(i, 0.0) for i in range(n)]),
                np.array([core.get(i, 0.0) for i in range(n)]))
    except Exception:
        return z, z, z


def compute_topology_features(A: np.ndarray, lap_k: int = 16) -> np.ndarray:
    A = np.nan_to_num(A.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    Ab = (A > 0).astype(float)
    As = np.maximum(A, A.T)
    Asb = (As > 0).astype(float)
    out_s, in_s, und_s = A.sum(1), A.sum(0), As.sum(1)
    out_d, in_d, und_d = Ab.sum(1), Ab.sum(0), Asb.sum(1)
    two_hop = np.maximum(((Asb @ Asb) > 0).sum(1) - 1, 0)
    pr_d, pr_u = _pagerank(A), _pagerank(As)
    bet, clu, core = _nx_feats(As)
    base = np.stack([out_s, in_s, und_s, out_d, in_d, und_d,
                     np.log1p(out_s), np.log1p(in_s), np.log1p(und_s),
                     two_hop, pr_d, pr_u, bet, clu, core], axis=1)
    return _std(np.concatenate([base, _lap_pe(As, lap_k)], axis=1))


def build_topology_embedding(A: np.ndarray, embed_dim: int = 32, lap_k: int = 16,
                             cache_path: Optional[str] = None, force_recompute: bool = False) -> np.ndarray:
    if cache_path and os.path.exists(cache_path) and not force_recompute:
        obj = np.load(cache_path)
        Z = obj['embedding']
        if Z.shape == (embed_dim, A.shape[0]):
            return Z.astype(np.float32)
    X = compute_topology_features(A, lap_k=lap_k)
    n, f = X.shape
    if f >= embed_dim:
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        Z = U[:, :embed_dim] * S[:embed_dim][None, :]
    else:
        Z = np.pad(X, ((0, 0), (0, embed_dim - f)), mode='constant')
    Z = _std(Z).astype(np.float32).T.copy()
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, embedding=Z, features=X.astype(np.float32))
    return Z


def default_moe_context(A: np.ndarray, has_temporal: bool = True, is_virtual: Optional[np.ndarray] = None) -> np.ndarray:
    A = np.nan_to_num(A.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    As = np.maximum(A, A.T)
    n = A.shape[0]
    if is_virtual is None:
        is_virtual = np.zeros(n)
    cont = _std(np.stack([A.sum(1), A.sum(0), As.sum(1), _pagerank(As)], axis=1))
    flags = np.stack([np.full(n, 1.0 if has_temporal else 0.0), is_virtual.astype(float)], axis=1)
    return np.concatenate([cont, flags], axis=1).astype(np.float32)


class SparseExpertFusion(nn.Module):
    def __init__(self, embed_dim=32, n_experts=3, ctx_dim=0, hidden_dim=64, top_k=2,
                 temperature=1.0, expert_names: Optional[Sequence[str]] = None,
                 init_temporal_bias: float = 1.0):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.n_experts = int(n_experts)
        self.top_k = max(1, min(int(top_k), self.n_experts))
        self.temperature = max(float(temperature), 1e-6)
        self.expert_names = list(expert_names) if expert_names else [f'expert_{i}' for i in range(n_experts)]
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim, bias=False)) for _ in range(n_experts)])
        self.delta = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim)) for _ in range(n_experts)])
        for d in self.delta:
            nn.init.zeros_(d[-1].weight); nn.init.zeros_(d[-1].bias)
        n_pairs = n_experts * (n_experts - 1) // 2
        router_in = n_experts * embed_dim + n_pairs * 2 * embed_dim + ctx_dim
        self.router = nn.Sequential(nn.Linear(router_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_experts))
        if n_experts >= 2:
            with torch.no_grad():
                self.router[-1].bias.zero_()
                if init_temporal_bias and 'temporal' in self.expert_names:
                    self.router[-1].bias[self.expert_names.index('temporal')] = float(init_temporal_bias)

    def _features(self, H, ctx):
        feats = [H.reshape(H.shape[0], -1)]
        pair = []
        for i in range(self.n_experts):
            for j in range(i + 1, self.n_experts):
                pair += [torch.abs(H[:, i] - H[:, j]), H[:, i] * H[:, j]]
        if pair:
            feats.append(torch.cat(pair, -1))
        if ctx is not None:
            feats.append(ctx.to(H.device, H.dtype))
        return torch.cat(feats, -1)

    def forward(self, embeds, ctx=None, avail_mask=None, adj_dense=None, return_aux=True):
        Hs, delta_l2 = [], None
        for e, z in enumerate(embeds):
            h = self.proj[e](z.T)
            d = self.delta[e](h)
            h = h + d
            cur = d.pow(2).mean()
            delta_l2 = cur if delta_l2 is None else delta_l2 + cur
            Hs.append(h)
        H = torch.stack(Hs, 1)  # [N,E,D]
        logits = self.router(self._features(H, ctx))
        if avail_mask is not None:
            logits = logits.masked_fill(avail_mask.to(logits.device) <= 0, -1e9)
        if self.top_k < self.n_experts:
            idx = torch.topk(logits, self.top_k, dim=-1).indices
            m = torch.zeros_like(logits).scatter_(1, idx, 1.0)
            logits = logits.masked_fill(m <= 0, -1e9)
        alpha = torch.softmax(logits / self.temperature, -1)
        fused = torch.einsum('ne,ned->dn', alpha, H).contiguous()
        if not return_aux:
            return fused
        mean_alpha = alpha.mean(0)
        lb = self.n_experts * torch.sum(mean_alpha * mean_alpha)
        ent = -(alpha * torch.log(alpha + 1e-8)).sum(-1).mean()
        smooth = fused.new_tensor(0.0)
        if adj_dense is not None and alpha.shape[0] <= 2500:
            W = adj_dense.to(alpha.device, alpha.dtype).clamp_min(0)
            if W.sum() > 0:
                diff = alpha[:, None, :] - alpha[None, :, :]
                smooth = (W[:, :, None] * diff.pow(2)).sum() / (W.sum() * self.n_experts + 1e-8)
        return fused, {'alpha': alpha, 'logits': logits, 'load_balance_loss': lb,
                       'entropy': ent, 'smooth_loss': smooth, 'delta_l2': delta_l2,
                       'expert_names': self.expert_names}


def add_moe_regularization(task_loss, aux, P):
    loss = task_loss
    for name, key in [('MOE_LB_REG','load_balance_loss'), ('MOE_SMOOTH_REG','smooth_loss'), ('MOE_DELTA_REG','delta_l2')]:
        coef = float(getattr(P, name, 0.0))
        if coef > 0:
            loss = loss + coef * aux[key]
    ent = float(getattr(P, 'MOE_ENTROPY_REG', 0.0))
    if ent != 0:
        loss = loss + ent * aux['entropy']
    return loss


def save_alpha_report(path: str, prefix: str, alpha: torch.Tensor, expert_names: Sequence[str]):
    os.makedirs(path, exist_ok=True)
    arr = alpha.detach().cpu().numpy()
    np.save(os.path.join(path, f'{prefix}_moe_alpha.npy'), arr)
    summary = {'expert_names': list(expert_names), 'mean': arr.mean(0).tolist(),
               'std': arr.std(0).tolist(), 'min': arr.min(0).tolist(), 'max': arr.max(0).tolist()}
    with open(os.path.join(path, f'{prefix}_moe_alpha_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
