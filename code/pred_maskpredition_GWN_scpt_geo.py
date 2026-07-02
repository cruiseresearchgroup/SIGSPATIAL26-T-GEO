import sys
import os
import shutil
import numpy as np
from scipy.fft import dct, idct
import pandas as pd
from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import Metrics
# import Utils
from GWN_SCPT_14_adpAdj_mask_infill import *
import unseen_nodes
from graph import generate_quotient_graph, generate_graphs, feature_extract, load_dataset, get_subgraph, get_additional_info
from torch_geometric.utils.convert import from_networkx
from torch.utils.data import DataLoader, Dataset, TensorDataset
import random
import matplotlib
import networkx as nx
from sklearn.preprocessing import MinMaxScaler # P
import os
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ["NO_PROXY"] = "overpass-api.de"
import matplotlib.pyplot as plt

class StandardScaler: #device
    def __init__(self):
        self.u = None
        self.z = None
    def fit_transform(self, x):
        self.u = x.mean()
        self.z = x.std()
        return (x-self.u)/self.z
    def inverse_transform(self, x):
        return x * self.z + self.u


def getXSYS_estimation(data, mode, missing_ratio=0.2,missing_ratio_test=0.2):
    TRAIN_NUM = int(data.shape[0] * P.TRAINRATIO)
    XS, YS, MS = [], [], []
    if mode == 'TRAIN':
        for i in range(TRAIN_NUM):
            x = data[i:i+1, :].copy()   # current timestep
            y = data[i:i+1, :].copy()   # target is the full current timestep
            m = np.ones_like(x)
            mask = np.random.rand(*x.shape) < missing_ratio
            x[mask] = 0.0
            m[mask] = 0.0
            XS.append(x)
            YS.append(y)
            MS.append(m)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM, data.shape[0]):
            x = data[i:i+1, :].copy()
            y = data[i:i+1, :].copy()
            m = np.ones_like(x)
            mask = np.random.rand(*x.shape) < missing_ratio_test
            x[mask] = 0.0
            m[mask] = 0.0
            XS.append(x)
            YS.append(y)
            MS.append(m)
    XS, YS, MS = np.array(XS), np.array(YS), np.array(MS)
    XS = XS[:, :, :, np.newaxis].transpose(0, 3, 2, 1)  # [B, 1, N, 1]
    YS = YS[:, :, :, np.newaxis]                        # [B, 1, N, 1]
    MS = MS[:, :, :, np.newaxis].transpose(0, 3, 2, 1)  # [B, 1, N, 1]
    return XS, YS, MS

# Custom TensorDataset that returns indices
class TensorDatasetWithIndices(TensorDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)  # Retrieve the original data (features, targets)
        return index, data  # Return the index along with the data

def setups_estimation(missing_ratio=0.4):
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
    print(P.KEYWORD, 'data splits (estimation task)', time.ctime())

    # ========== normalization & deseasonalization ==========
    #if P.IS_DESEASONED:
        #data_dct = dct(data, axis=0)
        #data_dct[P.n_dct_coeff:, :] = 0
        #data_ds = data - idct(data_dct, axis=0)
        #scaler = StandardScaler()
        #data = scaler.fit_transform(data)
        #scaler_ds = StandardScaler()
        #data_ds = scaler_ds.fit_transform(data_ds)
    #else:
        #scaler = StandardScaler()
        #data = scaler.fit_transform(data)

    # ========== temporal split ==========
    trainXS, trainYS, trainMS = getXSYS_estimation(data, 'TRAIN')
    testXS, testYS, testMS = getXSYS_estimation(data, 'TEST')

    if P.IS_DESEASONED:
        trainXS_ds, _, trainMS_ds = getXSYS_estimation(data_ds, 'TRAIN')
        testXS_ds, _, testMS_ds = getXSYS_estimation(data_ds, 'TEST')
        trainXS = np.concatenate((trainXS, trainXS_ds), axis=1)
        testXS = np.concatenate((testXS, testXS_ds), axis=1)
        trainMS = np.concatenate((trainMS, trainMS_ds), axis=1)
        testMS = np.concatenate((testMS, testMS_ds), axis=1)

    P.trainval_size = len(trainXS)
    P.train_size = int(P.trainval_size * (1 - P.TRAINVALSPLIT))
    XS_torch_trn = trainXS[:P.train_size]
    YS_torch_trn = trainYS[:P.train_size]
    MS_torch_trn = trainMS[:P.train_size]
    XS_torch_val = trainXS[P.train_size:]
    YS_torch_val = trainYS[P.train_size:]
    MS_torch_val = trainMS[P.train_size:]

    # ========== spatial split ==========
    spatialSplit_unseen = unseen_nodes.SpatialSplit(data.shape[1], r_trn=P.R_TRN, r_val=0.15, r_tst=0.15, seed=P.seed_SS)
    spatialSplit_allNod = unseen_nodes.SpatialSplit(data.shape[1], r_trn=P.R_TRN, r_val=min(1.0, P.R_TRN * 8 / 7), r_tst=1.0, seed=P.seed_SS)

    # ========== convert to Tensor ==========
    XS_torch_train = torch.Tensor(XS_torch_trn[:, :, spatialSplit_unseen.i_trn, :])
    YS_torch_train = torch.Tensor(YS_torch_trn[:, :, spatialSplit_unseen.i_trn, :])
    MS_torch_train = torch.Tensor(MS_torch_trn[:, :, spatialSplit_unseen.i_trn, :])

    XS_torch_val_u = torch.Tensor(XS_torch_val[:, :, spatialSplit_unseen.i_val, :])
    YS_torch_val_u = torch.Tensor(YS_torch_val[:, :, spatialSplit_unseen.i_val, :])
    MS_torch_val_u = torch.Tensor(MS_torch_val[:, :, spatialSplit_unseen.i_val, :])

    XS_torch_val_a = torch.Tensor(XS_torch_val[:, :, spatialSplit_allNod.i_val, :])
    YS_torch_val_a = torch.Tensor(YS_torch_val[:, :, spatialSplit_allNod.i_val, :])
    MS_torch_val_a = torch.Tensor(MS_torch_val[:, :, spatialSplit_allNod.i_val, :])

    XS_torch_tst_u = torch.Tensor(testXS[:, :, spatialSplit_unseen.i_tst, :])
    YS_torch_tst_u = torch.Tensor(testYS[:, :, spatialSplit_unseen.i_tst, :])
    MS_torch_tst_u = torch.Tensor(testMS[:, :, spatialSplit_unseen.i_tst, :])

    XS_torch_tst_a = torch.Tensor(testXS[:, :, spatialSplit_allNod.i_tst, :])
    YS_torch_tst_a = torch.Tensor(testYS[:, :, spatialSplit_allNod.i_tst, :])
    MS_torch_tst_a = torch.Tensor(testMS[:, :, spatialSplit_allNod.i_tst, :])

    # ========== build Dataset and DataLoader ==========
    train_data = torch.utils.data.TensorDataset(XS_torch_train, YS_torch_train, MS_torch_train)
    val_u_data = torch.utils.data.TensorDataset(XS_torch_val_u, YS_torch_val_u, MS_torch_val_u)
    val_a_data = torch.utils.data.TensorDataset(XS_torch_val_a, YS_torch_val_a, MS_torch_val_a)
    tst_u_data = torch.utils.data.TensorDataset(XS_torch_tst_u, YS_torch_tst_u, MS_torch_tst_u)
    tst_a_data = torch.utils.data.TensorDataset(XS_torch_tst_a, YS_torch_tst_a, MS_torch_tst_a)

    num_workers = 8  # tune to CPU cores; 4-8 is typical
    pin_memory = True if device.type == 'cuda' else False  # pinned memory for faster GPU transfer

    train_iter = torch.utils.data.DataLoader(train_data, P.BATCHSIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_u_iter = torch.utils.data.DataLoader(val_u_data, P.BATCHSIZE, shuffle=False)
    val_a_iter = torch.utils.data.DataLoader(val_a_data, P.BATCHSIZE, shuffle=False)
    tst_u_iter = torch.utils.data.DataLoader(tst_u_data, P.BATCHSIZE, shuffle=False)
    tst_a_iter = torch.utils.data.DataLoader(tst_a_data, P.BATCHSIZE, shuffle=False)

    # ========== adjacency ==========
    adj_mx = load_adj(P.ADJPATH, P.ADJTYPE, P.DATANAME)
    print(adj_mx)
    adj_train = [torch.tensor(i[spatialSplit_unseen.i_trn, :][:, spatialSplit_unseen.i_trn]).to(device) for i in adj_mx]
    adj_val_u = [torch.tensor(i[spatialSplit_unseen.i_val, :][:, spatialSplit_unseen.i_val]).to(device) for i in adj_mx]
    adj_val_a = [torch.tensor(i[spatialSplit_allNod.i_val, :][:, spatialSplit_allNod.i_val]).to(device) for i in adj_mx]
    adj_tst_u = [torch.tensor(i[spatialSplit_unseen.i_tst, :][:, spatialSplit_unseen.i_tst]).to(device) for i in adj_mx]
    adj_tst_a = [torch.tensor(i[spatialSplit_allNod.i_tst, :][:, spatialSplit_allNod.i_tst]).to(device) for i in adj_mx]

    # ========== pretraining node sampling ==========
    pretrn_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            XS_torch_train[:,-1,:,0].T), P.BATCHSIZE, shuffle=True)
    preval_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(trainYS[:,-1,spatialSplit_unseen.i_val,0]).T.float()),
        P.BATCHSIZE, shuffle=False)
    if not hasattr(P, "GEO_PRETRAIN_TRAIN_ONLY"):
        env_geo = os.environ.get("GEO_PRETRAIN_TRAIN_ONLY", "1").strip().lower()
        P.GEO_PRETRAIN_TRAIN_ONLY = env_geo not in ("0", "false", "no")
    if P.GEO_PRETRAIN_TRAIN_ONLY:
        geo_pool = list(spatialSplit_unseen.i_trn)
    else:
        geo_pool = list(range(int(data.shape[1])))
    k_geo = min(P.BATCHSIZE, len(geo_pool))
    pretrn_iterg = random.sample(geo_pool, k_geo)
    preval_iterg = list(spatialSplit_unseen.i_val)
    print(
        f"[geo-pretrain] GEO_PRETRAIN_TRAIN_ONLY={P.GEO_PRETRAIN_TRAIN_ONLY} "
        f"pool={len(geo_pool)} batch={k_geo}"
    )

    # dump hyperparameters
    for k, v in vars(P).items():
        print(k, v)
    mapping_tst_u = {old: new for new, old in enumerate(spatialSplit_unseen.i_tst)}

    return pretrn_iter, preval_iter, spatialSplit_unseen, spatialSplit_allNod, \
        train_iter, val_u_iter, val_a_iter, tst_u_iter, tst_a_iter, \
        adj_train, adj_val_u, adj_val_a, adj_tst_u, adj_tst_a, mapping_tst_u,pretrn_iterg,preval_iterg




def pre_evaluateModel(model, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x in data_iter:
            l = model.contrast(x[0].to(device))
            l_sum += l.item() * x[0].shape[0]
            n += x[0].shape[0]
        return l_sum / n
def network_calls():
    Q, nearest_node, clusters, gdf_nodes, gdf_edges, traffic, hull = generate_quotient_graph(P.QUOTIENT_GRAPH_RADIUS, P.DATANAME)
    info = get_additional_info(hull)
    return

def pretrainModel(name, mode, pretrain_iter, preval_iter):
    print('pretrainModel Started ...', time.ctime())
    model = Contrastive_FeatureExtractor_conv(P.TEMPERATURE).to(device)
    min_val_loss = np.inf
    optimizer = torch.optim.Adam(model.parameters(), lr=P.LEARN, weight_decay=P.weight_decay)
    s_time = datetime.now()
    for epoch in range(P.PRETRN_EPOCH):
        starttime = datetime.now()
        loss_sum, n = 0.0, 0
        model.train()
        for x in pretrain_iter:
            '''for x in pretrain_iter:
                print("type(x):", type(x))
            try:
                print("len(x):", len(x))
            except Exception as e:
                print("no len, err:", e)
            if hasattr(x, "shape"):
                print("x.shape:", x.shape)
            break'''
            optimizer.zero_grad()
            loss = model.contrast(x[0].to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x[0].shape[0]
            n += x[0].shape[0]
        train_loss = loss_sum / n
        val_loss = pre_evaluateModel(model, preval_iter)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), artifact_path(pretrain_artifact_key(name)))
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, "validation loss:", val_loss)
        with open(artifact_path(f"{pretrain_artifact_key(name)}_log"), 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))
    e_time = datetime.now()
    print('PRETIME DURATION:', e_time, '-', s_time, '=', e_time-s_time)
    print('pretrainModel Ended ...', time.ctime())

def pre_evaluateModel_g(model, data_iter, Q1, Q2):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x in data_iter:
            dataset_keys = {i: k for i, k in enumerate(load_dataset(P.DATANAME).keys())}
            Q1_s, Q2_s = get_subgraph(Q1, dataset_keys[x], P.SUBGRAPH_SIZE), get_subgraph(Q2, dataset_keys[x], P.SUBGRAPH_SIZE)
            fQ1, fQ2 = feature_extract(Q1_s, P.FEATURES).float().to(device), feature_extract(Q2_s, P.FEATURES).float().to(device) # 64x4 tensor
            nQ1, nQ2 = from_networkx(Q1_s).to(device), from_networkx(Q2_s).to(device)

            l = model.contrast(fQ1, fQ2, nQ1.edge_index, nQ2.edge_index)
            l_sum += l.item() * P.SUBGRAPH_SIZE
            n += P.SUBGRAPH_SIZE
        return l_sum / n
def pretrainModel_g(name, mode, pretrain_iter, preval_iter):
    print('pretrainModel Started ...', time.ctime())
    # model = Contrastive_FeatureExtractor_conv(P.TEMPERATURE).to(device)
    # this is a 207x4 matrix
    model = Geometric_Encoder(P.TEMPERATURE, P.FEATURES, P.GRAPH_NORM, P.HIDDEN).to(device)
    min_val_loss = np.inf
    optimizer = torch.optim.Adam(model.parameters(), lr=P.PRE_LEARN, weight_decay=P.weight_decay)
    s_time = datetime.now()
    Q, nearest_node, clusters, gdf_nodes, gdf_edges, traffic, hull = generate_quotient_graph(P.QUOTIENT_GRAPH_RADIUS, P.DATANAME)
    info = get_additional_info(hull)
    Q_nearest, _ = generate_graphs(Q, nearest_node, clusters, gdf_nodes, gdf_edges, info, nearest=True)
    scaler = MinMaxScaler()
    scaler.fit(feature_extract(Q_nearest, P.FEATURES))

    for epoch in range(P.PRETRN_EPOCH):
        # unseen stuff trainModel here
        Q1, Q2 = generate_graphs(Q, nearest_node, clusters, gdf_nodes, gdf_edges, info) # gives 2 networkx graphs 
        starttime = datetime.now()
        loss_sum, n = 0.0, 0
        model.train()
        # this used to be the data for BATCH_SIZE nodes (all data)
        # this should now be the features for BATCH_SIZE nodes (all features)
        # slice the 207x4 feature matrix into a BATCH_SIZEx4 feature matrix

        # pretrain_iter = len(0, 7, 108, 34, ...) = 100
        for x in pretrain_iter:
            dataset_keys = {i: k for i, k in enumerate(load_dataset(P.DATANAME).keys())}
            Q1_s, Q2_s = get_subgraph(Q1, dataset_keys[x], P.SUBGRAPH_SIZE), get_subgraph(Q2, dataset_keys[x], P.SUBGRAPH_SIZE)
            # x = len([0 7 108 34 ...]) = 64
            # dataset_keys = {i: k for i, k in enumerate(load_dataset().keys())}
            # indices = list(map(lambda k: dataset_keys[k], x))
            # # [0 -> 734108]
            # Q1_s = Q1.subgraph(indices).copy()
            # Q2_s = Q2.subgraph(indices).copy()
            # print(Q1, Q1_s, Q2, Q2_s)
            fQ1, fQ2 = torch.from_numpy(scaler.transform(feature_extract(Q1_s, P.FEATURES))).float().to(device), \
            torch.from_numpy(scaler.transform(feature_extract(Q2_s, P.FEATURES))).float().to(device) # 64x4 tensor
            # Q1 -> fQ1: feature matrix
            # Q1 -> nQ1: edge index, GCN doesn't like adjacency matrices
            nQ1, nQ2 = from_networkx(Q1_s).to(device), from_networkx(Q2_s).to(device)

            # fig = matplotlib.pyplot.figure()
            # nx.draw(Q1_s, pos=positions1)
            # fig.savefig("graph1.png")
            
            # fig = matplotlib.pyplot.figure()
            # nx.draw(Q2_s, pos=positions2)
            # fig.savefig("graph2.png")

            # return

            # print(fQ1, fQ2)
            # print(nQ1, nQ2)
            # x = [0, 15, 32, 79]
            # fQ1[x] = [[0.7, 0.3, 0.8, 0.5], [0.6, 0.3, 0.8, 0.5], ...] [64 x 4]
            optimizer.zero_grad()
            # loss = model.contrast([0.7, 0.3, 0.8, 0.5], [0.7, 0.3, 0.8, 0.5])
            loss = model.contrast(fQ1, fQ2, nQ1.edge_index, nQ2.edge_index)
            # loss = model.contrast(x[0].to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * P.SUBGRAPH_SIZE
            n += P.SUBGRAPH_SIZE
        train_loss = loss_sum / n
        val_loss = pre_evaluateModel_g(model, preval_iter, Q1, Q2)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), artifact_path(pretrain_artifact_key(name)))
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, "validation loss:", val_loss)
        with open(artifact_path(f"{pretrain_artifact_key(name)}_log"), 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))
    e_time = datetime.now()
    print('PRETIME DURATION:', e_time, '-', s_time, '=', e_time-s_time)
    print('pretrainModel Ended ...', time.ctime())

def getModel(name, device):
    model = gwnet(device, num_nodes=P.N_NODE, in_dim=P.CHANNEL, adp_adj=P.adp_adj, sga=P.is_SGA).to(device)
    return model


def masked_loss(y_pred, y_true, mask):
    # mask: [B, 1, N, T]  (1=observed, 0=missing)
    miss_mask = 1 - mask[:, 0, :, :].permute(0, 2, 1)  # -> [B, T, N], 1=missing
    loss = (y_pred - y_true) ** 2
    loss = loss * miss_mask
    return loss.sum() / (miss_mask.sum() + 1e-6)


def graph_constructor_helper():
    Q, nearest_node, clusters, gdf_nodes, gdf_edges, traffic, hull = generate_quotient_graph(P.QUOTIENT_GRAPH_RADIUS, P.DATANAME)
    info = get_additional_info(hull)
    Q1, _ = generate_graphs(Q, nearest_node, clusters, gdf_nodes, gdf_edges, info, nearest=True) # gives 2 networkx graphs 
    dataset_keys = {i: k for i, k in enumerate(load_dataset(P.DATANAME).keys())}
    fQ1 = feature_extract(Q1, P.FEATURES).float().to(device)
    # Q1 -> fQ1: feature matrix
    # Q1 -> nQ1: edge index, GCN doesn't like adjacency matrices
    nQ1 = from_networkx(Q1)
    return fQ1, nQ1


class TemporalBaseDeltaFusion(nn.Module):
    """
    fused = z_temporal + gamma * delta(z_temporal, z_geometric)
    Same as temporal_delta in pred_GWN_16_adpAdj.py.
    """
    def __init__(self, embed_dim=32, hidden_dim=64, use_projection=True):
        super().__init__()
        self.use_projection = use_projection
        if use_projection:
            self.proj_t = nn.Linear(embed_dim, embed_dim, bias=False)
            self.proj_g = nn.Linear(embed_dim, embed_dim, bias=False)
        else:
            self.proj_t = None
            self.proj_g = None

        self.delta_net = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        nn.init.zeros_(self.delta_net[-1].weight)
        nn.init.zeros_(self.delta_net[-1].bias)

        self.gamma_net = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.gamma_net[-1].bias, -4.0)

    def forward(self, z_t, z_g):
        ztt = z_t.T
        zgg = z_g.T
        if self.use_projection:
            ztt = self.proj_t(ztt)
            zgg = self.proj_g(zgg)
        feat = torch.cat([ztt, zgg, ztt - zgg, ztt * zgg], dim=1)
        delta = self.delta_net(feat).T
        gamma_logits = self.gamma_net(feat).T
        gamma = torch.sigmoid(gamma_logits)
        fused = z_t + gamma * delta
        return fused, delta


def _normalize_embeddings_fuse(embed_temporal, embed_geometric):
    if not getattr(P, 'FUSION_NORM', True):
        return embed_temporal, embed_geometric
    d = embed_temporal.shape[0]
    z_t = F.layer_norm(embed_temporal.T, (d,)).T
    z_g = F.layer_norm(embed_geometric.T, (d,)).T
    return z_t, z_g


def _fuse_embeddings_temporal_delta(embed_temporal, embed_geometric, gate_module, return_gate=False):
    embed_temporal, embed_geometric = _normalize_embeddings_fuse(embed_temporal, embed_geometric)
    fused, delta = gate_module(embed_temporal, embed_geometric)
    return (fused, delta) if return_gate else fused


def _temporal_full_embed_est(encoder):
    encoder.eval()
    data_ = data_ds if P.IS_DESEASONED else data
    with torch.no_grad():
        return encoder(torch.Tensor(data_[:P.trainval_size, :]).to(device).float().T).T.detach()


def _geometric_full_embed_est(encoderg):
    encoderg.eval()
    with torch.no_grad():
        fQ1, nQ1 = graph_constructor_helper()
        return encoderg(fQ1.to(device), nQ1.edge_index.to(device)).T.detach()


def trainModel_estimation_with_pretrain(name,
                                        train_iter, val_u_iter, val_a_iter,
                                        adj_train, adj_val_u, adj_val_a,
                                        spatialSplit_unseen, spatialSplit_allNod,pretrn_iterg,preval_iterg ):
    print('trainModel (Estimation + Pretrain) Started ...', time.ctime())
    model = getModel(name, device)
    criterion = masked_loss
    s_time = datetime.now()
    gate_module = None
    temp_full_embed = geo_full_embed = None

    # === pretrain: full-graph topo/geo embed + learnable temporal_delta fusion ===
    if P.IS_PRETRN:
        encoder = Contrastive_FeatureExtractor_conv(P.TEMPERATURE).to(device)
        encoder.load_state_dict(torch.load(resolve_artifact_path("pretrain_topo"), map_location=device))
        encoderg = Geometric_Encoder(P.TEMPERATURE, P.FEATURES, P.GRAPH_NORM, P.HIDDEN).to(device)
        encoderg.load_state_dict(torch.load(resolve_artifact_path("pretrain_geo"), map_location=device))
        temp_full_embed = _temporal_full_embed_est(encoder)
        geo_full_embed = _geometric_full_embed_est(encoderg)
        gate_module = TemporalBaseDeltaFusion(
            embed_dim=temp_full_embed.shape[0],
            hidden_dim=getattr(P, 'GATE_HIDDEN', 64),
            use_projection=True,
        ).to(device)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(gate_module.parameters()),
            lr=P.LEARN, weight_decay=P.weight_decay)
        print('Using temporal_delta fusion (TemporalBaseDeltaFusion), embed_dim=', temp_full_embed.shape[0])
        gate_module.eval()
        with torch.no_grad():
            fe0, _ = _fuse_embeddings_temporal_delta(
                temp_full_embed, geo_full_embed, gate_module, return_gate=True)
        train_embed = fe0[:, spatialSplit_unseen.i_trn]
        val_u_embed = fe0[:, spatialSplit_unseen.i_val]
        val_a_embed = fe0[:, spatialSplit_allNod.i_val]
        gate_module.train()
    else:
        optimizer = torch.optim.Adam(list(model.parameters()), lr=P.LEARN, weight_decay=P.weight_decay)
        train_embed = torch.zeros(32, train_iter.dataset.tensors[0].shape[2]).to(device).detach()
        val_u_embed = torch.zeros(32, val_u_iter.dataset.tensors[0].shape[2]).to(device).detach()
        val_a_embed = torch.zeros(32, val_a_iter.dataset.tensors[0].shape[2]).to(device).detach()

    print('train_embed', train_embed.shape)
    print('val_u_embed', val_u_embed.shape)
    print('val_a_embed', val_a_embed.shape)

    min_val_loss = float('inf')

    for epoch in range(P.EPOCH):
        model.train()
        epoch_loss, n = 0.0, 0
        start_time = datetime.now()
        # ===== regenerate TRAIN random masks each epoch (val/test fixed) =====
        trainXS_ep, trainYS_ep, trainMS_ep = getXSYS_estimation(
            data, 'TRAIN', missing_ratio=P.MISS_RATIO
        )

        if P.IS_DESEASONED:
            trainXS_ds_ep, _, trainMS_ds_ep = getXSYS_estimation(
                data_ds, 'TRAIN', missing_ratio=P.MISS_RATIO
            )
            trainXS_ep = np.concatenate((trainXS_ep, trainXS_ds_ep), axis=1)   # C=2
            trainMS_ep = np.concatenate((trainMS_ep, trainMS_ds_ep), axis=1)

        # train split only (exclude val), spatial subset (unseen train nodes)
        XS_ep = trainXS_ep[:P.train_size][:, :, spatialSplit_unseen.i_trn, :]
        YS_ep = trainYS_ep[:P.train_size][:, :, spatialSplit_unseen.i_trn, :]
        MS_ep = trainMS_ep[:P.train_size][:, :, spatialSplit_unseen.i_trn, :]

        train_data_ep = torch.utils.data.TensorDataset(
            torch.Tensor(XS_ep),
            torch.Tensor(YS_ep),
            torch.Tensor(MS_ep)
        )

        train_iter_epoch = torch.utils.data.DataLoader(
            train_data_ep,
            P.BATCHSIZE,
            shuffle=True,
            num_workers=8,
            pin_memory=True if device.type == 'cuda' else False
        )
        if gate_module is not None:
            gate_module.train()
        batch_times: list[float] = []
        for x, y, mask in train_iter_epoch:
            t_b = time.perf_counter()
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            y = y.squeeze(-1)
            if gate_module is not None:
                train_full, tr_delta = _fuse_embeddings_temporal_delta(
                    temp_full_embed, geo_full_embed, gate_module, return_gate=True)
                train_embed = train_full[:, spatialSplit_unseen.i_trn]
                y_pred = model(x, adj_train, train_embed)
                loss = criterion(y_pred, y, mask)
                dr = getattr(P, 'DELTA_REG', 0.0)
                if dr > 0 and tr_delta is not None:
                    d = tr_delta[:, spatialSplit_unseen.i_trn]
                    loss = loss + dr * d.pow(2).mean()
            else:
                y_pred = model(x, adj_train, train_embed)
                loss = criterion(y_pred, y, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n += 1
            batch_times.append(time.perf_counter() - t_b)

        train_loss = epoch_loss / n
        n_it = len(batch_times)
        mean_it = (sum(batch_times) / n_it) if n_it else 0.0
        if gate_module is not None:
            gate_module.eval()
            with torch.no_grad():
                fe_ep, full_delta = _fuse_embeddings_temporal_delta(
                    temp_full_embed, geo_full_embed, gate_module, return_gate=True)
            train_embed = fe_ep[:, spatialSplit_unseen.i_trn]
            val_u_embed = fe_ep[:, spatialSplit_unseen.i_val]
            val_a_embed = fe_ep[:, spatialSplit_allNod.i_val]
            if full_delta is not None:
                print('delta(epoch) mean/std/min/max',
                      full_delta.mean().item(), full_delta.std().item(),
                      full_delta.min().item(), full_delta.max().item())
        val_u_loss = evaluateModel_estimation_with_pretrain(model, val_u_iter, val_u_embed, adj_val_u)
        val_a_loss = evaluateModel_estimation_with_pretrain(model, val_a_iter, val_a_embed, adj_val_a)

        if val_u_loss < min_val_loss:
            min_val_loss = val_u_loss
            torch.save(model.state_dict(), artifact_path("best"))
            if gate_module is not None:
                torch.save(gate_module.state_dict(), artifact_path("fusion_u"))

        epoch_time = (datetime.now() - start_time).seconds
        print(f"Epoch {epoch}, Time {epoch_time}s, Train Loss: {train_loss:.6f}, "
              f"Val_U Loss: {val_u_loss:.6f}, Val_A Loss: {val_a_loss:.6f}")
        with open(artifact_path("train_log"), 'a') as f:
            f.write(f"epoch,{epoch},time,{epoch_time},train_loss,{train_loss:.10f},"
                    f"val_u_loss,{val_u_loss:.10f},val_a_loss,{val_a_loss:.10f}\n")
            f.write(
                f"iter_epoch,{epoch},mean_batch_sec,{mean_it:.10f},n_batches,{n_it}\n"
            )

    print("TRAINING FINISHED. Best val_u loss:", min_val_loss)
    print('MODEL TRAINING DURATION:', datetime.now() - s_time)

    train_score = evaluateModel_estimation_with_pretrain(model, train_iter, train_embed, adj_train)
    print(f"{name}, MAE on train: {train_score:.6f}")
    with open(artifact_path("prediction_scores"), 'a') as f:
        f.write(f"{name}, estimation, MAE on train, {train_score:.10f}, {train_score:.10f}\n")


def testModel_estimation_with_pretrain(name, mode, test_iter, node_indices, scaler, adj_tst_u, mapping,spatialsplit):
    """
    node_indices: original node IDs
    mapping: {original_id -> subgraph index}
    """
    print('Model Testing (Estimation)', mode, 'Started ...', time.ctime())

    # === load model ===
    model = getModel(name, device)
    model.load_state_dict(torch.load(resolve_artifact_path("best"), map_location=device))
    model.to(device)
    model.eval()
    node_embed = torch.zeros(32, test_iter.dataset.tensors[0].shape[2]).to(device).detach()

    # === node embed: full-graph topo + geo, temporal_delta fuse, then test columns ===
    if P.IS_PRETRN:
        encoder = Contrastive_FeatureExtractor_conv(P.TEMPERATURE).to(device)
        encoder.load_state_dict(torch.load(resolve_artifact_path("pretrain_topo"), map_location=device))
        encoderg = Geometric_Encoder(P.TEMPERATURE, P.FEATURES, P.GRAPH_NORM, P.HIDDEN).to(device)
        encoderg.load_state_dict(torch.load(resolve_artifact_path("pretrain_geo"), map_location=device))
        temp_full_embed = _temporal_full_embed_est(encoder)
        geo_full_embed = _geometric_full_embed_est(encoderg)
        gate_module = TemporalBaseDeltaFusion(
            embed_dim=temp_full_embed.shape[0],
            hidden_dim=getattr(P, 'GATE_HIDDEN', 64),
            use_projection=True,
        ).to(device)
        fusion_ckpt = resolve_artifact_path("fusion_u")
        if os.path.exists(fusion_ckpt):
            gate_module.load_state_dict(torch.load(fusion_ckpt, map_location=device))
        else:
            print('fusion checkpoint not found, fallback to randomly initialized module:', fusion_ckpt)
        gate_module.eval()
        with torch.no_grad():
            full_fused = _fuse_embeddings_temporal_delta(temp_full_embed, geo_full_embed, gate_module)
            node_embed = full_fused[:, node_indices]
        print("Using temporal_delta fused embeddings for testing...")
    else:
        print("No pre-training: using zero embeddings for testing.")
        hidden_dim = 32
        node_embed = torch.zeros((hidden_dim, len(node_indices)), device=device)

    adj_tst_u = [a.to(device) for a in adj_tst_u]

    Y_true_list, Y_pred_list, MISS_list = [], [], []

    inf_t0 = time.perf_counter()
    with torch.no_grad():
        for batch in test_iter:
            if len(batch) != 3:
                print("[WARN] unexpected batch format:", type(batch), len(batch))
                continue
            x, y, ms = batch
            x = x.to(device, non_blocking=True)    # [B, C, N, T]
            y = y.to(device, non_blocking=True)    # [B, T, N, 1]
            ms = ms.to(device, non_blocking=True)  # [B, 1, N, T]

            y = y.squeeze(-1)                      # [B, T, N]
            y_pred = model(x, adj_tst_u, node_embed)  # [B, T, N]

            # [B, N, T]
            y_pred = y_pred.permute(0, 2, 1)
            y      = y.permute(0, 2, 1)

            miss_mask = 1.0 - ms[:, 0, :, :]       # [B, N, T]

            Y_true_list.append(y.cpu().numpy())
            Y_pred_list.append(y_pred.cpu().numpy())
            MISS_list.append(miss_mask.cpu().numpy())

    inf_sec = time.perf_counter() - inf_t0
    print(f"{name}, {mode}, INFERENCE_WALL_SEC, {inf_sec:.6f}")
    with open(artifact_path("timing_infer"), "a", encoding="utf-8") as tf:
        tf.write(f"{mode},{inf_sec:.10f}\n")

    Y_true = np.concatenate(Y_true_list, axis=0)
    Y_pred = np.concatenate(Y_pred_list, axis=0)
    MISS   = np.concatenate(MISS_list,   axis=0)

    print("[DBG] shapes  Y_true:", Y_true.shape, "  Y_pred:", Y_pred.shape, "  MISS:", MISS.shape)
    assert Y_true.shape == Y_pred.shape == MISS.shape

    # === inverse scaling ===
    def safe_inverse_transform_3d(arr, scaler):
        B, N, T = arr.shape
        if hasattr(scaler, "mean_") and np.ndim(getattr(scaler, "mean_")) == 1:
            flat = arr.transpose(0, 2, 1).reshape(-1, N)
            flat = scaler.inverse_transform(flat)
            return flat.reshape(B, T, N).transpose(0, 2, 1)
        else:
            return scaler.inverse_transform(arr)

    Y_true = safe_inverse_transform_3d(Y_true, scaler)
    Y_pred = safe_inverse_transform_3d(Y_pred, scaler)

    # === metrics (masked positions only) ===
    eps = 1e-6
    abs_err = np.abs(Y_true - Y_pred) * MISS
    sq_err  = ((Y_true - Y_pred) ** 2) * MISS
    den     = MISS.sum() + eps

    MAE  = abs_err.sum() / den
    RMSE = np.sqrt(sq_err.sum() / den)
    MAPE = (np.abs((Y_true - Y_pred) / (np.abs(Y_true) + eps)) * MISS).sum() / den

    print('*' * 40)
    print(f"{name}, {mode}, Masked MAE: {MAE:.6f}, RMSE: {RMSE:.6f}, MAPE: {MAPE:.6f}")

    # === save outputs ===
    np.save(mode_npy_path(mode, "prediction"), Y_pred)
    np.save(mode_npy_path(mode, "groundtruth"), Y_true)
    np.save(mode_npy_path(mode, "missmask"), MISS)

    with open(artifact_path("prediction_scores"), 'a') as f:
        f.write(f"{name}, {mode}, Masked MAE, {MAE:.10f}, RMSE, {RMSE:.10f}, MAPE, {MAPE:.10f}\n")

    # === heatmap helper ===
    def plot_heatmap(matrix, nodes, title, save_path):
        plt.figure(figsize=(10, 6))
        plt.imshow(matrix.T, aspect="auto", cmap="viridis", origin="lower")
        plt.colorbar(label="Speed")
        plt.xlabel("Time step")
        plt.ylabel("Node ID")
        plt.yticks(range(len(nodes)), nodes)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    # === plot a few nodes (show original IDs, index via mapping) ===
    subset_nodes_raw = list(node_indices[:10])
    subset_nodes = [mapping[n] for n in subset_nodes_raw if n in mapping]

    B, N, T = Y_pred.shape
    pred_sub = Y_pred[:, subset_nodes, :].reshape(B, len(subset_nodes))
    true_sub = Y_true[:, subset_nodes, :].reshape(B, len(subset_nodes))

    plot_heatmap(true_sub, subset_nodes_raw, "Ground Truth Speeds", f"{P.PATH}/heatmap_true_{mode}.png")
    plot_heatmap(pred_sub, subset_nodes_raw, "Predicted Speeds", f"{P.PATH}/heatmap_pred_{mode}.png")
    plot_heatmap(np.abs(true_sub - pred_sub), subset_nodes_raw, "Prediction Error", f"{P.PATH}/heatmap_error_{mode}.png")

    print('Model Testing Ended ...', time.ctime())

def evaluateModel_estimation_with_pretrain(model, data_iter, node_embed, adj):
    """
    Evaluate loss only at masked (missing) positions:
      - MS: 1=observed, 0=missing
      - miss_mask = 1 - MS for missing-only evaluation
    Dataset-wide weighted average:
      loss = sum((y_pred - y_true)^2 * miss_mask) / sum(miss_mask)
    """
    model.eval()
    num_sum = 0.0   # numerator: accumulated weighted error
    den_sum = 0.0   # denominator: count of valid mask entries
    with torch.no_grad():
        for x, y, ms in data_iter:
            x, y, ms = x.to(device), y.to(device), ms.to(device)

            # y: [B, 1, N, 1] -> [B, T, N], T=1 here
            y = y.squeeze(-1)            # [B, 1, N]
            # model output: [B, T, N]
            y_pred = model(x, adj, node_embed)  # [B, T, N]

            # miss_mask: channel 0; [B, 1, N, 1] -> [B, T, N]
            miss_mask = 1.0 - ms[:, 0, :, :]      # [B, N, T]
            miss_mask = miss_mask.permute(0, 2, 1)  # -> [B, T, N]

            # accumulate globally
            sq_err = (y_pred - y) ** 2            # [B, T, N]
            num_sum += (sq_err * miss_mask).sum().item()
            den_sum += miss_mask.sum().item()

    return num_sum / (den_sum + 1e-6)

################# Parameter Setting #######################
P = type('Parameters', (object,), {})()
P.TIMESTEP_IN =1
P.TIMESTEP_OUT = 1
P.CHANNEL = 1
P.BATCHSIZE = 64 # 64
P.LEARN = 0.0003
P.PRETRN_EPOCH = 100
P.EPOCH = 100# 100
P.TRAINRATIO = 0.8 # TRAIN + VAL  USE_MASK
P.TRAINVALSPLIT = 0.125 # val_ratio = 0.8 * 0.125 = 0.1
P.ADJTYPE = 'doubletransition'
P.MODELNAME = 'GraphWaveNet'
P.FEATURES = 4
P.SUBGRAPH_SIZE = 64
P.QUOTIENT_GRAPH_RADIUS = 0.01
P.NETWORK_CALLS = 0
P.PRE_LEARN = 0.0001
P.GRAPH_NORM = False
P.HIDDEN = 320
P.MISS_RATIO = 0.2
P.GATE_HIDDEN = 64
P.FUSION_NORM = True
P.DELTA_REG = 0.0
data = None
data_ds = None
scaler = None
###########################################################
def get_argv():
    ''' # ARGV
    0: .py file
    1: IS_PRETRN
    2: R_TRN
    3: IS_EPOCH_1
    4: seed
    5: TEMPERATURE
    6: dataset
    7: seed_ss # spatial split
    8: IS_DESEASONED
    9: weight_decay
    10: adp_adj
    11: is_SGA
    12: FEATURES
    '''
    print('sys.argv', sys.argv)
    P.IS_PRETRN = bool(int(sys.argv[1])) if len(sys.argv) >= 2 else True #True
    P.R_TRN = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.7
    P.IS_EPOCH_1 = bool(int(sys.argv[3])) if len(sys.argv) >= 4 else False
    P.seed = int(sys.argv[4]) if len(sys.argv) >= 5 else 100#27#83#42#100 
    P.TEMPERATURE = float(sys.argv[5]) if len(sys.argv) >= 6 else 1.0
    P.DATANAME = sys.argv[6] if len(sys.argv) >= 7 else 'PEMSBAY'
    P.seed_SS = int(sys.argv[7]) if len(sys.argv) >= 8 else -1#100
    P.IS_DESEASONED = bool(int(sys.argv[8])) if len(sys.argv) >= 9 else True
    P.weight_decay = float(sys.argv[9]) if len(sys.argv) >= 10 else 0.0001
    P.adp_adj = bool(int(sys.argv[10])) if len(sys.argv) >= 11 else True
    P.is_SGA = bool(int(sys.argv[11])) if len(sys.argv) >= 12 else True
    P.FEATURES = int(sys.argv[12]) if len(sys.argv) >= 13 else 4
    P.SUBGRAPH_SIZE = int(sys.argv[13]) if len(sys.argv) >= 14 else 64
    P.QUOTIENT_GRAPH_RADIUS = float(sys.argv[14]) if len(sys.argv) >= 15 else 0.01
    P.PRETRN_EPOCH = int(sys.argv[15]) if len(sys.argv) >= 16 else 100
    P.EPOCH = int(sys.argv[16]) if len(sys.argv) >= 17 else 100
    P.NETWORK_CALLS = bool(int(sys.argv[17])) if len(sys.argv) >= 18 else 0
    P.PRE_LEARN = float(sys.argv[18]) if len(sys.argv) >= 19 else P.LEARN
    P.GRAPH_NORM = bool(int(sys.argv[19])) if len(sys.argv) >= 20 else True
    P.HIDDEN = int(sys.argv[20]) if len(sys.argv) >= 21 else 320

device = torch.device('cuda:0') 
#device = torch.device("cpu")
###########################################################

def run_keyword():
    """Save dir name, e.g. est_METRLA_GraphWaveNet_2606061540_3596005"""
    return (
        "est_" + P.DATANAME + "_" + P.MODELNAME + "_"
        + datetime.now().strftime("%y%m%d%H%M") + "_" + str(os.getpid())
    )


_ARTIFACT_NAMES = {
    "best": "best.pt",
    "fusion_u": "fusion_u.pt",
    "train_log": "train_log.txt",
    "prediction_scores": "prediction_scores.txt",
    "timing_infer": "timing_infer.txt",
    "pretrain_topo": "pretrain_topo.pt",
    "pretrain_geo": "pretrain_geo.pt",
    "pretrain_topo_log": "pretrain_topo_log.txt",
    "pretrain_geo_log": "pretrain_geo_log.txt",
}

_ARTIFACT_LEGACY = {
    "best": ("GraphWaveNet_best.pt",),
    "fusion_u": ("GraphWaveNet_fusion_u.pt",),
    "train_log": ("GraphWaveNet_log.txt",),
    "prediction_scores": ("GraphWaveNet_prediction_scores.txt",),
    "timing_infer": ("GraphWaveNet_timing_infer.txt",),
    "pretrain_topo": ("pretrain_temporal.pt", "encoder.pt"),
    "pretrain_geo": ("encoderg.pt",),
    "pretrain_topo_log": ("pretrain_temporal_log.txt", "encoder_log.txt",),
    "pretrain_geo_log": ("encoderg_log.txt",),
}

_PRETRAIN_NAME_TO_KEY = {"encoder": "pretrain_topo", "encoderg": "pretrain_geo"}


def artifact_path(key: str) -> str:
    return os.path.join(P.PATH, _ARTIFACT_NAMES[key])


def resolve_artifact_path(key: str) -> str:
    primary = artifact_path(key)
    if os.path.isfile(primary):
        return primary
    for legacy in _ARTIFACT_LEGACY.get(key, ()):
        legacy_path = os.path.join(P.PATH, legacy)
        if os.path.isfile(legacy_path):
            return legacy_path
    return primary


def pretrain_artifact_key(name: str) -> str:
    return _PRETRAIN_NAME_TO_KEY.get(name, name)


def mode_npy_path(mode: str, kind: str) -> str:
    return os.path.join(P.PATH, f"{mode}_{kind}.npy")

def main():
    script_start_time = datetime.now()
    get_argv()

    # === paths ===
    eval_dir = os.environ.get("FORECASTING_EVAL_DIR", "").strip()
    eval_mode = os.environ.get("FORECASTING_EVAL_MODE", "").strip()
    # Backward compatible shortcut: only-eval tst_a
    eval_tst_a_dir = os.environ.get("FORECASTING_EVAL_TST_A_DIR", "").strip()
    if eval_tst_a_dir and not eval_dir:
        eval_dir = eval_tst_a_dir
        eval_mode = eval_mode or "tst_a"

    if eval_dir:
        P.PATH = os.path.abspath(eval_dir)
        P.KEYWORD = os.path.basename(P.PATH.rstrip("/"))
    else:
        P.KEYWORD = run_keyword()
        P.PATH = '../save/' + P.KEYWORD
    print('PATH', os.path.abspath(P.PATH))
    print('KEYWORD', P.KEYWORD)

    global data
    global data_ds
    global scaler_main   # scaler for raw series
    global scaler_ds     # scaler for deseasonalized component (optional)

    scaler = None
    scaler_ds = None

    # === load data ===
    if P.DATANAME == 'METRLA':
        P.FLOWPATH = '../METRLA/metr-la.h5'
        P.n_dct_coeff = 3918
        P.ADJPATH = '../METRLA/adj_mx.pkl'
        P.N_NODE = 207
        data = pd.read_hdf(P.FLOWPATH).values
    elif P.DATANAME == 'PEMSBAY':
        P.FLOWPATH = '../PEMSBAY/pems-bay.h5'
        P.n_dct_coeff = 4107
        P.ADJPATH = '../PEMSBAY/adj_mx_bay.pkl'
        P.N_NODE = 325
        data = pd.read_hdf(P.FLOWPATH).values
    elif P.DATANAME == 'PEMSD7M':
        P.FLOWPATH = '../PEMSD7M/V_228.csv'
        P.n_dct_coeff = 860
        P.ADJPATH = '../PEMSD7M/W_228.csv'
        P.N_NODE = 228
        data = pd.read_csv(P.FLOWPATH, index_col=[0]).values
    elif P.DATANAME == 'EXPYTKY':
        P.BATCHSIZE = 8
        P.FLOWPATH = '../EXPYTKY/expy-tky.h5'
        P.n_dct_coeff = 1514
        P.ADJPATH = '../EXPYTKY/adj_mx.pkl'
        P.N_NODE = 1843
        data = pd.read_hdf(P.FLOWPATH).values
    elif P.DATANAME == 'PEMS07':
        P.BATCHSIZE = 32
        P.FLOWPATH = '../PEMS07/pems07_speed.h5'
        P.n_dct_coeff = 3904
        P.ADJPATH = '../PEMS07/adj_mx.pkl'
        P.N_NODE = 400
        data = pd.read_hdf(P.FLOWPATH).values
    elif P.DATANAME == 'PEMS11160':
        P.BATCHSIZE = 16
        P.EPOCH = 20
        P.FLOWPATH = '../PEMS11160/pems12kSPEED2m.npy'
        P.n_dct_coeff = 2179
        P.ADJPATH = '../PEMS11160/adj_mat.pkl'
        P.N_NODE = 11160
        with open(P.FLOWPATH, 'rb') as f:
            data = np.load(f)
    else:
        raise ValueError("Unsupported dataset name")

    # === optional network topology viz ===
    if P.NETWORK_CALLS:
        network_calls()
        return

    # === deseasonalization (optional) ===
    if P.IS_DESEASONED:
        P.CHANNEL = 2
        data_ = dct(data, axis=0)
        data_[P.n_dct_coeff:, :] = 0
        data_ds = data - idct(data_, axis=0)
    else:
        data_ds = None

    # === standardization (keep separate scalers) ===
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    if P.IS_DESEASONED:
        scaler = StandardScaler()
        data_ds = scaler.fit_transform(data_ds)

    print('data.shape', data.shape)
    if P.IS_DESEASONED:
        print('[DEBUG] use deseasoned branch: CHANNEL=2')

    # === build estimation datasets ===
    # pass missing_ratio through so train masks use the intended rate (e.g. 0.2)
    pretrn_iter, preval_iter, spatialSplit_unseen, spatialSplit_allNod, \
    train_iter, val_iter, val_a_iter, test_iter, tst_a_iter, \
    adj_train, adj_val, adj_val_a, adj_test, adj_tst_a,mapping_tst_u,pretrn_iterg,preval_iterg = setups_estimation(missing_ratio=0.2)
    print(adj_train)
    print(adj_val)

    mapping_tst_a = {old: new for new, old in enumerate(spatialSplit_allNod.i_tst)}

    if eval_dir:
        mode = eval_mode or "tst_a"
        print(P.KEYWORD, f"eval only mode={mode} (FORECASTING_EVAL_DIR)", time.ctime())

        if mode == "tst_a":
            test_iter = tst_a_iter
            node_indices = spatialSplit_allNod.i_tst
            adj = adj_tst_a
            mapping = mapping_tst_a
            ssplit = spatialSplit_allNod
        elif mode == "tst_u":
            if getattr(P, "SKIP_TST_U", False):
                raise ValueError("tst_u is disabled for this entrypoint; use tst_v_full")
            test_iter = test_iter
            node_indices = getattr(P, "TST_U_NODE_INDICES", spatialSplit_unseen.i_tst)
            adj = adj_test
            mapping = mapping_tst_u
            ssplit = spatialSplit_unseen
        elif mode == "tst_v_full":
            # Provided by custom setups_estimation (e.g. splitmask entrypoint with shared i_tst).
            test_iter = getattr(P, "TST_V_FULL_ITER")
            node_indices = getattr(P, "TST_V_FULL_NODES")
            adj = getattr(P, "TST_V_FULL_ADJ")
            mapping = getattr(P, "TST_V_FULL_MAPPING")
            ssplit = getattr(P, "TST_V_FULL_SPATIALSPLIT", spatialSplit_allNod)
        elif mode == "shared_eval":
            eval_splits = (
                ("tst_v_full", getattr(P, "TST_V_FULL_ITER"), getattr(P, "TST_V_FULL_NODES"), getattr(P, "TST_V_FULL_ADJ"), getattr(P, "TST_V_FULL_MAPPING"), getattr(P, "TST_V_FULL_SPATIALSPLIT", spatialSplit_allNod)),
            )
            if not getattr(P, "SKIP_TST_U", False):
                eval_splits = (
                    ("tst_u", test_iter, getattr(P, "TST_U_NODE_INDICES", spatialSplit_unseen.i_tst), adj_test, mapping_tst_u, spatialSplit_unseen),
                ) + eval_splits
            for emode, eiter, enodes, eadj, emap, essplit in eval_splits:
                testModel_estimation_with_pretrain(
                    name=P.MODELNAME,
                    mode=emode,
                    test_iter=eiter,
                    node_indices=enodes,
                    scaler=scaler,
                    adj_tst_u=eadj,
                    mapping=emap,
                    spatialsplit=essplit,
                )
            print('SCRIPT DURATION', datetime.now() - script_start_time)
            return
        else:
            raise ValueError(f"Unsupported FORECASTING_EVAL_MODE={mode!r}")

        testModel_estimation_with_pretrain(
            name=P.MODELNAME,
            mode=mode,
            test_iter=test_iter,
            node_indices=node_indices,
            scaler=scaler,
            adj_tst_u=adj,
            mapping=mapping,
            spatialsplit=ssplit,
        )
        print('SCRIPT DURATION', datetime.now() - script_start_time)
        return

    if P.IS_PRETRN:
        print(P.KEYWORD, 'pretraining started', time.ctime())
        pretrainModel('encoder', 'pretrain', pretrn_iter, preval_iter)
        pretrainModel_g('encoderg', 'pretrain', pretrn_iterg, preval_iterg)

    else:
        print(P.KEYWORD, 'No pre-training')

    # === estimation training ===
    print(P.KEYWORD, 'training started', time.ctime())
    trainModel_estimation_with_pretrain(
        P.MODELNAME,
        train_iter,        # unseen train nodes
        val_iter,          # unseen val nodes
        val_a_iter,        # all-node val
        adj_train, adj_val, adj_val_a,
        spatialSplit_unseen, spatialSplit_allNod ,pretrn_iterg,preval_iterg
    )

    # === estimation testing ===
    print(P.KEYWORD, 'testing started', time.ctime())
    if getattr(P, "SKIP_TST_U", False):
        testModel_estimation_with_pretrain(
            name=P.MODELNAME,
            mode="tst_v_full",
            test_iter=getattr(P, "TST_V_FULL_ITER"),
            node_indices=getattr(P, "TST_V_FULL_NODES"),
            scaler=scaler,
            adj_tst_u=getattr(P, "TST_V_FULL_ADJ"),
            mapping=getattr(P, "TST_V_FULL_MAPPING"),
            spatialsplit=getattr(P, "TST_V_FULL_SPATIALSPLIT", spatialSplit_allNod),
        )
    else:
        testModel_estimation_with_pretrain(
            name=P.MODELNAME,
            mode="tst_u",
            test_iter=test_iter,
            node_indices=getattr(P, "TST_U_NODE_INDICES", spatialSplit_unseen.i_tst),
            scaler=scaler,
            adj_tst_u=adj_test,
            mapping=mapping_tst_u,
            spatialsplit=spatialSplit_unseen,
        )

    testModel_estimation_with_pretrain(
        name=P.MODELNAME,
        mode="tst_a",
        test_iter=tst_a_iter,
        node_indices=spatialSplit_allNod.i_tst,
        scaler=scaler,
        adj_tst_u=adj_tst_a,
        mapping=mapping_tst_a,
        spatialsplit=spatialSplit_allNod,
    )

    print('SCRIPT DURATION', datetime.now() - script_start_time)

if __name__ == '__main__':
    main()