import logging

import numpy as np
import torch

from generalization import gen_function

seed = 11
np.random.seed(seed)
torch.manual_seed(seed)

from preprocessing import num_objects, num_events, num_unique_acts, remaining_time, ocel_to_csv
from major_function import main_function

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    # Create list of filenames and corresponding object types to use
    names_ot_dict = {"recruiting-ocel1": ['applicants', 'applications', 'offers'],
                     "BPI2017-Final": ["application", "offer"],
                     "ocel2-p2p": ['material', 'purchase_requisition', 'quotation', 'purchase_order', 'goods_receipt',
                                   'invoice_receipt', 'payment']
                     }

    # Target extraction functions
    subg_funcs = [remaining_time]
    g_funcs = [num_events]  # num_events, num_objects, num_unique_acts]
    # Set this value to:
    # - True to use the graph structure only as input
    # - False to use event-level features detailed in main_function
    no_feats = [False]
    ks = range(3, 15)
    embedding_name_list = ["GAT", "GCN", "GraphTransformer"]
    embedding_size_list = [32]
    batch_size = 128
    learning_rate = 0.01
    mlp_hidden_dim_list = [16]
    mlp_num_layers_list = [2]
    filename = "BPI2017-Final"
    object_types = names_ot_dict[filename]
    parameters = {"obj_names": object_types,
                  "val_names": [],
                  "act_name": "event_activity",
                  "time_name": "event_timestamp",
                  "sep": ","}
    for embedding_name in embedding_name_list:
        for embedding_size in embedding_size_list:
            for mlp_hidden_dim in mlp_hidden_dim_list:
                for mlp_num_layers in mlp_num_layers_list:
                    for no_feat in no_feats:
                        # main_function(filename, parameters, subg_funcs, g_funcs, no_feat, ks, embedding_name,
                        #              embedding_size, batch_size, learning_rate, mlp_hidden_dim, mlp_num_layers)
                        gen_function(filename, parameters, subg_funcs, g_funcs, no_feat, ks, embedding_name,
                                     embedding_size, batch_size, learning_rate, mlp_hidden_dim, mlp_num_layers, subg=False)
    a = 0
