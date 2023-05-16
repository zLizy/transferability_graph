import numpy as np
import pandas as pd

from utils._util import *
from utils.graph import HGraph

import os
import json
import pickle
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.data import download_url, extract_zip
# We can make use of the `loader.LinkNeighborLoader` from PyG:
from torch_geometric.loader import LinkNeighborLoader

import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def gnn_train(args,df_perf,data_dict,evaluation_dict,setting_dict,batch_size):
    unique_dataset_id = data_dict['unique_dataset_id']
    data_feat = data_dict['data_feat']
    unique_model_id = data_dict['unique_model_id']
    model_feat = data_dict['model_feat']
    edge_index_model_to_dataset = data_dict['edge_index_model_to_dataset']
    edge_attr_model_to_dataset = data_dict['edge_attr_model_to_dataset']
    edge_index_dataset_to_dataset = data_dict['edge_index_dataset_to_dataset']
    
    ## Construct a heterogeneous graph
    graph = HGraph(
        unique_model_id,
        model_feat,
        unique_dataset_id,
        data_feat,
        edge_index_model_to_dataset,
        edge_index_dataset_to_dataset,
        edge_attr_model_to_dataset,
        contain_data_similarity=args.contain_data_similarity,
        contain_dataset_feature=args.contain_dataset_feature,
        contain_model_feature=args.contain_model_feature,
        )
    data = graph.data

    
    ## Define GNN
    print('== Load a GNN ==')
    from utils.gnn import Model
    num_dataset_nodes = data["dataset"].num_nodes
    num_model_nodes = data["model"].num_nodes
    evaluation_dict['num_model'] = num_model_nodes
    evaluation_dict['num_dataset'] = num_dataset_nodes
    if args.embed_dataset_feature:
        dim_dataset_feature = data["dataset"].x.shape[1]
    else:
        dim_dataset_feature = 1
    if args.contain_model_feature:
        dim_model_feature = data["model"].x.shape[1]
    else:
        dim_model_feature = 1
    metadata = data.metadata()
    # print()
    # print(f'==== dim_dataset_feature: {dim_dataset_feature}')
    # print()
    # print(f'==== dim_model_feature: {dim_model_feature}')
    
    # metadata,num_dataset_nodes,num_model_nodes,dim_dataset_feature,hidden_channels
    model = Model(
                metadata,
                num_dataset_nodes,
                num_model_nodes,
                dim_model_feature,
                dim_dataset_feature,
                args.contain_model_feature,
                args.contain_dataset_feature,
                args.embed_model_feature,
                args.embed_dataset_feature,
                args.gnn_method,
                hidden_channels=args.hidden_channels, #64 # 128
                node_types=data.node_types
            )
    print()

    ## Initialize lazy modules
    # with torch.no_grad():  # Initialize lazy modules.
    #     out = model(data)
    print(f'== Begin splitting dataset ==')
    train_data, val_data,test_data = graph.split()
    print(f'len(train_data): {len(train_data)},len(train_data): {len(val_data)}, len(train_data): {len(test_data)}')
    # print(val_data.x_dict.keys())
    # print(val_data.edge_index_dict.keys())
    # print("train_data['model'].x.shape")
    # print(train_data['model'].x.shape)
    # print("train_data['model'].node_id.shape")
    # print(train_data['model'].node_id.shape)
    # print(train_data["model", "trained_on", "dataset"].edge_label_index)
    # print(train_data["model", "trained_on", "dataset"].edge_label)
    print(train_data)
    # print()
    # print("val_data['dataset'].x.shape")
    # print(val_data['dataset'].x.shape)
    # print(val_data)
    # print()
    # print("test_data['dataset'].x.shape")
    # print(test_data['dataset'].x.shape)
    # print(test_data)
    # print()

    print('== Begin training network ==')
    # with torch.no_grad():  # Initialize lazy modules.
    #     out = model(data.x_dict, data.edge_index_dict)
    ## Train model
    if args.gnn_method == 'SAGEConv':
        epochs = 20
    elif args.gnn_method == 'GATConv':
        epochs = 10
    else:
        epochs = 20
    model,loss, train_time = train(model,train_data,batch_size,epochs)
    evaluation_dict['epochs'] = epochs
    evaluation_dict['loss'] = loss
    evaluation_dict['train_time'] = train_time
    evaluation_dict['hidden_channels'] = args.hidden_channels
    print()

    ## Validate model on validation set
    val_AUC = validate(model,val_data,batch_size=batch_size)
    evaluation_dict['val_AUC'] = val_AUC
    ## Evaluate model on test set
    test_AUC = validate(model,test_data,batch_size=batch_size)
    evaluation_dict['test_AUC'] = test_AUC

    print()
    print('========== predict model nodes for the test dataset ========')
    print(unique_dataset_id[unique_dataset_id['dataset']==args.test_dataset])
    
    dataset_index = data_dict['dataset_idx']
    dataset_index = np.repeat(dataset_index,data['model'].num_nodes)
    data["model", "trained_on", "dataset"].edge_label_index = torch.stack([data['model'].node_id,torch.from_numpy(dataset_index).to(torch.int64)],dim=0)
    pred = predict_model_for_dataset(model,data)
    norm = np.linalg.norm(pred)     # To find the norm of the array
    # print(norm)                        # Printing the value of the norm
    normalized_pred = pred/norm 
    print(normalized_pred[:5])
    config_name = '_'.join([('{0}=={1}'.format(k, v)) for k,v in setting_dict.items()])

    # Save graph embedding distance results
    dir_path = os.path.join('./rank',f'{args.test_dataset}',args.dataset_emb_method, args.gnn_method)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    unique_model_id['score'] = normalized_pred.tolist()
    # np.save(os.path.join(dir_path,config_name+'.npy'),pred)
    unique_model_id.to_csv(os.path.join(dir_path,config_name+'.csv'))

     
    df_perf = pd.concat([df_perf,pd.DataFrame(evaluation_dict,index=[0])],ignore_index=True)
    print()
    print('======== save =======')
    # save the 
    df_perf.to_csv(args.path)