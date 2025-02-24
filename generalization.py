import os.path

import numpy as np
import pandas as pd

from ocpa.objects.log.importer.csv import factory as csv_import_factory
from ocpa.algo.predictive_monitoring import factory as predictive_monitoring

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP

from model import GAT, CustomPipeline, train_loop, evaluate, GCN, GraphTransformer
from preprocessing import get_process_executions_nx, generate_matrix_dataset, get_subgraphs_labeled

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from utils import get_pred_types


def gen_function(filename, parameters, subg_funcs, g_funcs, no_feat, ks, embedding_name, embedding_size,
                 batch_size, learning_rate, mlp_hidden_dim, mlp_num_layers, subg=True):
    # OCEL file import
    print(f"Importing OCEL {filename}")
    file_path = os.path.join('.', 'ocel', 'csv', filename)
    ocel = csv_import_factory.apply(file_path=file_path + '.csv', parameters=parameters)

    # Feature and target definition
    print("Preprocessing")
    activities = list(set(ocel.log.log["event_activity"].tolist()))

    # Event level targets
    event_target_set = [(predictive_monitoring.EVENT_REMAINING_TIME, ())]

    # Execution level targets (Outcome related)
    execution_target_set = [(predictive_monitoring.EXECUTION_NUM_OBJECT, ()),
                            (predictive_monitoring.EXECUTION_NUM_OF_END_EVENTS, ()),
                            (predictive_monitoring.EXECUTION_NUM_OF_EVENTS, ()),
                            (predictive_monitoring.EXECUTION_UNIQUE_ACTIVITIES, ())]

    if no_feat:
        execution_feature_set = []
        event_feature_set = []
    else:
        # Event level features
        event_feature_set = [(predictive_monitoring.EVENT_ELAPSED_TIME, ()),
                             (predictive_monitoring.EVENT_NUM_OF_OBJECTS, ())] + \
                            [(predictive_monitoring.EVENT_ACTIVITY, (act,)) for act in activities]
        # [(predictive_monitoring.EVENT_PRECEDING_ACTIVITIES, (act,)) for act in activities] + \
        execution_feature_set = []
    PE_nx = get_process_executions_nx(ocel, event_feature_set, event_target_set, execution_feature_set,
                                      execution_target_set)

    for k in ks:
        # Create result dataframe
        if subg:
            task_names = [f.__name__ for f in subg_funcs]
        else:
            task_names = [f.__name__ for f in g_funcs]

        print(f"Using subgraphs of length {k}")
        if subg:
            subgraph_list = get_subgraphs_labeled(PE_nx, k=k, subg_funcs=subg_funcs,
                                                  g_funcs=[])
        else:
            subgraph_list = get_subgraphs_labeled(PE_nx, k=k, subg_funcs=[],
                                                  g_funcs=g_funcs)

        if subg:
            pred_types = get_pred_types(subgraph_list, subg_funcs, [], task_names)
        else:
            pred_types = get_pred_types(subgraph_list, [], g_funcs, task_names)
        # pred_types = [None for i in pred_types]
        result_dfs = []
        result_dir_paths = []
        for task_name in task_names:
            result_df = pd.DataFrame(columns=['prediction_layer', 'score'])
            result_dfs.append(result_df)
            result_dir_path = os.path.join('.', 'results', filename, task_name, str(k))
            result_dir_paths.append(result_dir_path)
            os.makedirs(result_dir_path, exist_ok=True)

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

        idx_feats_to_scale = [0]
        input_dataset = generate_matrix_dataset(subgraph_list, idx_feats_to_scale, k, no_feat)
        num_features = input_dataset[0].x.shape[1]
        # train val test split
        temp_data, test_data = train_test_split(input_dataset, test_size=0.2, random_state=42)
        train_data, val_data = train_test_split(temp_data, test_size=0.2, random_state=42)
        # data loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        # embedding, predictor setup
        print("Model Construction")
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
                ml = RandomForestRegressor()
            else:
                target_dim = pred_type
                ml = RandomForestClassifier()
            ml_preds.append(ml)
        # Init pipeline
        model = CustomPipeline(embedding_layer, nn_preds, ml_preds)
        # Init optimizer
        optimizer = optim.Adam(model.parameters(), learning_rate)
        # Move model and data to device (cuda if exists)
        # model = model.to(device)
        print("Training...")
        train_loop(model, optimizer, train_loader, val_loader, pred_types, num_epochs=20)
        # Tests on linear ml models
        test_nn_scores, test_ml_scores = evaluate(model, test_loader, pred_types, train_loader)
        # Store test scores for nn and ml
        for i, result_df in enumerate(result_dfs):
            if pred_types[i] is None:
                a = scaler.inverse_transform(np.array(test_nn_scores[i]).reshape(-1, 1))[0][0]
                res_row = [f"MLP_{mlp_hidden_dim}_{mlp_num_layers}", a / 86400]
                result_df.loc[len(result_df)] = res_row
                b = scaler.inverse_transform(test_ml_scores[i].reshape(-1, 1))[0][0]
                res_row = ["Random forest models", b / 86400]
                result_df.loc[len(result_df)] = res_row
            else:
                res_row = [f"MLP_{mlp_hidden_dim}_{mlp_num_layers}", test_nn_scores[i]]
                result_df.loc[len(result_df)] = res_row
                res_row = ["Random forest models", test_ml_scores[i]]
                result_df.loc[len(result_df)] = res_row

        for i, result_df in enumerate(result_dfs):
            result_file_path = os.path.join(result_dir_paths[i], f"{embedding_name}_{embedding_size}_{no_feat}.csv")
            result_df.to_csv(result_file_path, mode='a')
            print(result_df)

        embedding_A = model.Embedding
        for param in embedding_A.parameters():
            param.requires_grad = False

        if not subg:
            task_names = [f.__name__ for f in subg_funcs]
        else:
            task_names = [f.__name__ for f in g_funcs]

        print(f"Using subgraphs of length {k}")
        if not subg:
            subgraph_list = get_subgraphs_labeled(PE_nx, k=k, subg_funcs=subg_funcs,
                                                  g_funcs=[])
        else:
            subgraph_list = get_subgraphs_labeled(PE_nx, k=k, subg_funcs=[],
                                                  g_funcs=g_funcs)

        if not subg:
            pred_types = get_pred_types(subgraph_list, subg_funcs, [], task_names)
        else:
            pred_types = get_pred_types(subgraph_list, [], g_funcs, task_names)
        # pred_types = [None for i in pred_types]
        result_dfs = []
        result_dir_paths = []
        for task_name in task_names:
            result_df = pd.DataFrame(columns=['prediction_layer', 'score'])
            result_dfs.append(result_df)
            result_dir_path = os.path.join('.', 'results', filename, task_name, str(k))
            result_dir_paths.append(result_dir_path)
            os.makedirs(result_dir_path, exist_ok=True)

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

        idx_feats_to_scale = [0]
        input_dataset = generate_matrix_dataset(subgraph_list, idx_feats_to_scale, k, no_feat)
        num_features = input_dataset[0].x.shape[1]
        # train val test split
        temp_data, test_data = train_test_split(input_dataset, test_size=0.2, random_state=42)
        train_data, val_data = train_test_split(temp_data, test_size=0.2, random_state=42)
        # data loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

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
                ml = RandomForestRegressor()
            else:
                target_dim = pred_type
                ml = RandomForestClassifier()
            ml_preds.append(ml)
        # Init pipeline
        model = CustomPipeline(embedding_A, nn_preds, ml_preds)
        # Init optimizer
        # optimizer = optim.Adam(model.parameters(), learning_rate)
        # train_loop(model, optimizer, train_loader, val_loader, pred_types, num_epochs=20)
        # Tests on linear ml models

        test_nn_scores, test_ml_scores = evaluate(model, test_loader, pred_types, train_loader)
        # Store test scores for nn and ml
        for i, result_df in enumerate(result_dfs):
            if pred_types[i] is None:
                a = scaler.inverse_transform(np.array(test_nn_scores[i]).reshape(-1, 1))[0][0]
                res_row = [f"MLP_{mlp_hidden_dim}_{mlp_num_layers}", a / 86400]
                result_df.loc[len(result_df)] = res_row
                b = scaler.inverse_transform(test_ml_scores[i].reshape(-1, 1))[0][0]
                res_row = ["Random forest models", b / 86400]
                result_df.loc[len(result_df)] = res_row
            else:
                res_row = [f"MLP_{mlp_hidden_dim}_{mlp_num_layers}", test_nn_scores[i]]
                result_df.loc[len(result_df)] = res_row
                res_row = ["Random forest models", test_ml_scores[i]]
                result_df.loc[len(result_df)] = res_row

        for i, result_df in enumerate(result_dfs):
            result_file_path = os.path.join(result_dir_paths[i], f"{embedding_name}_{embedding_size}_{no_feat}.csv")
            result_df.to_csv(result_file_path, mode='a')
            print(result_df)
