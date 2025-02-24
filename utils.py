import os

import numpy as np
import pandas as pd
from karateclub import Graph2Vec, FGSD
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from torch_geometric.nn import MLP

from model import GCN, GraphTransformer, GAT


def get_pred_types(subgraph_list, subg_funcs, g_funcs, task_names):
    get_num_classes = lambda feature_idx: [len(np.unique([g[feature_idx] for g in subgraph_list]))]
    if len(subg_funcs) > 0:
        pred_types = [None]
    else:
        pred_types = []

    if len(g_funcs) > 0:
        for i in range(len(g_funcs)):
            num_classes = get_num_classes(2 + len(subg_funcs) + i)
            if num_classes[0] == 1:
                task_names.pop(len(subg_funcs) + i)
            if num_classes[0] > 1:
                pred_types += num_classes
    return pred_types


def create_res_dfs(filename, task_names, k):
    result_dfs = []
    result_dir_paths = []
    for task_name in task_names:
        result_df = pd.DataFrame(columns=['prediction_layer', 'score'])
        result_dfs.append(result_df)
        result_dir_path = os.path.join('.', 'results', filename, task_name, str(k))
        result_dir_paths.append(result_dir_path)
        os.makedirs(result_dir_path, exist_ok=True)


def scale_encode_targets(subgraph_list, pred_types):
    y = np.array([g[2:] for g in subgraph_list])
    for i, pred_type in enumerate(pred_types):
        if pred_type is None:
            scaler = MinMaxScaler()
            y[:, i] = scaler.fit_transform(y[:, i].reshape(-1, 1))[:, 0]
            for idx, subgraph in enumerate(subgraph_list):
                subgraph[2 + i] = float(y[idx, i])
        else:
            if pred_type > 6:
                a, b = np.unique(y[:, i], return_counts=True)
                total_sum = np.sum(b)
                cumulative_sum = np.cumsum(b)
                half_sum = total_sum / 2
                cut_index = np.searchsorted(cumulative_sum, half_sum)
                v = a[cut_index]
                y[:, i] = (y[:, i] <= v).astype(int)
            else:
                encoder = OrdinalEncoder()
                y[:, i] = encoder.fit_transform(y[:, i].reshape(-1, 1))[:, 0]
            for idx, subgraph in enumerate(subgraph_list):
                subgraph[2 + i] = int(y[idx, i])

    return scaler


def init_embedding_component(embedding_name, num_features, embedding_size, train_data):
    # Init Embedding model
    if embedding_name == 'GAT':
        embedding_layer = GAT(num_layers=2, num_features=num_features, hidden_dim=16, target_size=embedding_size,
                              heads=4)
    elif embedding_name == 'GraphTransformer':
        embedding_layer = GraphTransformer(num_layers=2, num_features=num_features, hidden_dim=16,
                                           target_size=embedding_size,
                                           heads=4)
    elif embedding_name == 'GCN':
        embedding_layer = GCN(num_layers=2, num_features=num_features, hidden_dim=16, target_size=embedding_size)
    elif embedding_name == 'Graph2Vec':
        embedding_layer = Graph2Vec(dimensions=embedding_size)
        embedding_layer.fit([t.graph for t in train_data])
    elif embedding_name == 'FGSD':
        embedding_layer = FGSD(hist_bins=embedding_size)
        embedding_layer.fit([t.graph for t in train_data])
    return embedding_layer


def init_pred_component(regressor, classifier, embedding_size, mlp_hidden_dim, mlp_num_layers, pred_types):
    # Init predictors
    nn_preds = []
    for pred_type in pred_types:
        if pred_type is None:
            target_dim = 1
            mlp = MLP(in_channels=embedding_size, hidden_channels=mlp_hidden_dim, out_channels=target_dim,
                      num_layers=mlp_num_layers)
        else:
            if pred_type > 6:
                target_dim = 2
            else:
                target_dim = pred_type
            mlp = MLP(in_channels=embedding_size, hidden_channels=mlp_hidden_dim, out_channels=target_dim,
                      num_layers=mlp_num_layers)
        nn_preds.append(mlp)

    ml_preds = []
    for pred_type in pred_types:
        if pred_type is None:
            target_dim = 1
            ml = regressor()
        else:
            target_dim = pred_type
            ml = classifier()
        ml_preds.append(ml)
    return nn_preds, ml_preds


def set_new_ml_preds(regressor, classifier, pred_types):
    # Tests on Random forest models
    new_ml_preds = []
    for pred_type in pred_types:
        if pred_type is None:
            target_dim = 1
            new_ml = regressor()
        else:
            target_dim = pred_type
            new_ml = classifier()
        new_ml_preds.append(new_ml)

    return new_ml_preds
