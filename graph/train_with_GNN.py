import numpy as np
import pandas as pd

from utils._util import *
from utils.graph import HGraph
from torch_geometric.data import Data
from utils.CustomRandomLinkSplit import RandomLinkSplit

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

SAVE_GRAPH=True
SAVE_AND_BREAK=False

## Train
def train(model,train_data,graph_type='hetero',label_type=[],batch_size=4,epochs=5):
    start = time.time()
    s,r,t = label_type
    # model = model.to(device)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#,capturable=True)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    if graph_type == 'hetero':
        train_loader = get_dataloader(train_data,label_type,batch_size=batch_size,is_train=True)
    elif graph_type == 'homo':
        train_loader = get_homo_dataloader(train_data,batch_size=batch_size,is_train=True)
    # print("Capturing:", torch.cuda.is_current_stream_capturing())
    # torch.cuda.empty_cache()
    for epoch in range(1, epochs+1):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            # print(sampled_data['dataset'].x)
            # sampled_data.to(device)
            # print(sampled_data)
            # print(sampled_data['dataset'].x)
            pred = model(sampled_data)
            if graph_type == 'hetero':
                ground_truth = sampled_data[s,r,t].edge_label
                loss = F.binary_cross_entropy(pred, ground_truth)
            elif graph_type == 'homo':
                ground_truth = sampled_data.edge_label
                loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            # loss = F.cross_entropy(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        # if total_loss < 1: break
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
        # if (total_loss / total_examples) < 0.1: break
    train_time = time.time()-start
    return model,round(total_loss/total_examples,4), train_time

def validate(model,val_data,graph_type='hetero',label_type=[],batch_size=8):
    s,r,t = label_type
    preds = []
    ground_truths = []
    if graph_type == 'hetero':
        val_loader = get_dataloader(val_data,label_type,batch_size=batch_size,is_train=True)
    elif graph_type == 'homo':
        val_loader = get_homo_dataloader(val_data,batch_size=batch_size,is_train=True)
    
    for sampled_data in tqdm.tqdm(val_loader):
        with torch.no_grad():
            sampled_data.to(device)
            preds.append(model(sampled_data))
            if graph_type == 'hetero':
                ground_truths.append(sampled_data[s,r,t].edge_label)
            elif graph_type == 'homo':
                ground_truths.append(sampled_data.edge_label)
    pred = torch.cat(preds, dim=0).cpu().numpy()
    print(f'len(pred): {len(pred)}')#, pred: {pred}')
    mask = pred > 0
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    print(f'len(ground_truth): {len(ground_truth)}: {np.sort(ground_truth)}')
    auc = roc_auc_score(ground_truth, pred, multi_class='ovr')
    pre = precision_score(ground_truth.astype(int),mask)
    print()
    print(f"Validation AUC: {auc:.4f}, pred: {pre:.4f}")
    return auc

def custom_replace(data):
    edge_label = data.edge_label.clone()
    edge_label[data.edge_label==2] = 1
    data.edge_label = edge_label
    return data
    
def get_edge_label(data):
    print()
    print(f'======= homo graph processing ========')
    edge_label_index = data["model", "trained_on", "dataset"].edge_label_index
    edge_label = data["model", "trained_on", "dataset"].edge_label
    return edge_label_index, edge_label

def gnn_train(args,df_perf,data_dict,evaluation_dict,setting_dict,batch_size,custom_negative_sampling=False):
    unique_dataset_id = data_dict['unique_dataset_id']
    data_feat = data_dict['data_feat']
    unique_model_id = data_dict['unique_model_id']
    model_feat = data_dict['model_feat']
    model_idx = data_dict['model_idx']
    edge_index_accu_model_to_dataset = data_dict['edge_index_accu_model_to_dataset']
    edge_attr_accu_model_to_dataset = data_dict['edge_attr_accu_model_to_dataset']
    edge_index_tran_model_to_dataset = data_dict['edge_index_tran_model_to_dataset']
    edge_attr_tran_model_to_dataset = data_dict['edge_attr_tran_model_to_dataset']
    edge_index_dataset_to_dataset = data_dict['edge_index_dataset_to_dataset']
    edge_attr_dataset_to_dataset = data_dict['edge_attr_dataset_to_dataset']
    negative_pairs = data_dict['negative_pairs']
    max_dataset_idx = data_dict['max_dataset_idx']

    # print('')
    # print(f'data_dict: {data_dict}')
    
    ## Construct a heterogeneous graph
    graph = HGraph(
        args.gnn_method,
        max_dataset_idx,
        model_idx,
        unique_model_id,
        model_feat,
        unique_dataset_id,
        data_feat,
        edge_index_accu_model_to_dataset,
        edge_attr_accu_model_to_dataset,
        edge_index_dataset_to_dataset,
        edge_attr_dataset_to_dataset,
        edge_index_tran_model_to_dataset,
        edge_attr_tran_model_to_dataset,
        negative_pairs,
        contain_data_similarity=args.contain_data_similarity,
        contain_dataset_feature=args.contain_dataset_feature,
        contain_model_feature=args.contain_model_feature,
        custom_negative_sampling=custom_negative_sampling
        )
    
    data = graph.data
    print(f'== Begin splitting dataset ==')
    train_data, val_data,test_data = graph.split()
    
    ## Validate (sub)graphs
    # data.validate()
    # train_data.validate()
    # val_data.validate()
    # test_data.validate()
    # print(val_data)
    
    ### !!! label_type ["model", "trained_on", "dataset"] or  ["model", "transfer", "dataset"]
    label_type = graph.label_type
    s,r,t = label_type
    
    unique_values = train_data[s,r,t].edge_label.unique(return_counts=True)
    if len(unique_values[0])>2 or 2 in train_data[s,r,t].edge_label: 
        print(f'=== 2 in train_data')
        train_data = custom_replace(train_data)
    unique_values = val_data[s,r,t].edge_label.unique(return_counts=True)
    if len(unique_values[0])>2 or 2 in val_data[s,r,t].edge_label: 
        print(f'=== 2 in val_data')
        val_data = custom_replace(val_data)
    unique_values = test_data[s,r,t].edge_label.unique(return_counts=True)
    if len(unique_values[0])>2 or 2 in test_data[s,r,t].edge_label: 
        print(f'=== 2 in test_data')
        test_data = custom_replace(test_data)

    ### save thd graph
    setting_dict.pop('gnn_method')
    config_name = '_'.join([('{0}={1}'.format(k[:14], str(v)[:5])) for k,v in setting_dict.items()])
    if SAVE_GRAPH:
        if 'without_transfer' in args.gnn_method:
            gnn = 'gnn_graph_without_transfer'
            config_name = 'without_transfer_'+config_name
        elif 'without_accuracy' in args.gnn_method:
            gnn = 'gnn_graph_without_accuracy'
            config_name = 'without_accuracy_'+config_name
        else:
            gnn = 'gnn_graph'
        if 'homo' in args.gnn_method: gnn = 'homo_'+gnn
            
        _dir = os.path.join('./saved_graph',f'{args.test_dataset}', gnn)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        # if not os.path.exists(os.path.join(_dir,config_name+'.pt')):
        torch.save(data, os.path.join(_dir,config_name+'.pt'))
        if SAVE_AND_BREAK:
            return 0, ''
    
    ## Define GNN
    print(f'== Load {args.gnn_method} ==')
    from utils.gnn import HeteroModel, HomoModel
    num_dataset_nodes = data["dataset"].num_nodes
    num_model_nodes = data["model"].num_nodes
    evaluation_dict['num_model'] = num_model_nodes
    evaluation_dict['num_dataset'] = num_dataset_nodes
    if args.contain_dataset_feature:
        dim_dataset_feature = data["dataset"].x.shape[1]
    else:
        dim_dataset_feature = 1
    if args.contain_model_feature:
        dim_model_feature = data["model"].x.shape[1]
    else:
        dim_model_feature = 1
    metadata = data.metadata()
    print()
    print(f'metadata: {metadata}')
    print('\n',data[metadata[1][0]])

    if 'homo' in args.gnn_method:
        # edge_label_index, edge_label = get_edge_label(train_data)
        
        print(f'\n== data{data}')
        data_hetero = data
        # data = data_hetero.to_homogeneous(add_node_type=False,add_edge_type=False)
        edge_lists = []
        for edge_name in metadata[1]:
            print(edge_name)
            edge_lists.append(data_hetero[edge_name].edge_index)
        
        node_x_lists = []
        dimension = 128
        for node_name in metadata[0]:
            print(node_name)
            try:
                dimension = data_hetero[node_name].x.shape[1]
                print(f'dimension: {dimension}')
                node_x_lists.append(data_hetero[node_name].x)
            except:
                node_x_lists.append(torch.rand((data_hetero[node_name].num_nodes,dimension)))
       
        data = Data(x=torch.cat(node_x_lists,dim=0),edge_index=torch.cat(edge_lists,dim=1))
        data.node_id = torch.from_numpy(np.asarray(data_dict['node_ID']))
        print(f'\n===== homo_data: {data}')

        transform = RandomLinkSplit(#T.Compose([T.ToUndirected(),
            num_val=0.1,  # TODO
            num_test=0.2,  # TODO
            disjoint_train_ratio=0.3,  # TODO
            neg_sampling_ratio=2.0,  # TODO
            add_negative_train_samples=True,  # TODO
            negative_pairs=negative_pairs,
            is_undirected=True,
            # rev_edge_types = self.data.metadata()[1]
            # rev_edge_types=("dataset", "rev_trained_on", "model"), 
            custom_negative_sampling=custom_negative_sampling
        )
        train_data_hetero = train_data
        train_data, val_data, test_data = transform(data)
        # data = data_homo
        print(f'data_homo: {data}')
        print(data.node_id)

        ## Validate (sub)graphs
        data.validate()
        train_data.validate()
        val_data.validate()
        test_data.validate()

        # train_data =train_data.to_homogeneous(add_node_type=False,add_edge_type=False)
        # print(f'train_data: {train_data}')
        # print(f'\n== train_data_hetero:\n{train_data_hetero}')
        # print(f'\n==train_data_hetero.edge_label_index.shape:{edge_label_index.shape}')
        # print(f'edge_label.shape: {edge_label.shape}')
        # print(f'\nnew edge_label.shape: {train_data.edge_label.shape}')
        print(f'\nnew train_data_homo.edge_label.shape: {train_data.edge_label.shape}')
        print(f'train_data_homo.edge_label.shape: {train_data.edge_label.shape}')
        print(f'train_data_homo.edge_label.unique count: {train_data.edge_label.unique(return_counts=True)}')
        
        unique_values = val_data.edge_label.unique(return_counts=True)
        print(f'\nval_data_homo.edge_label.unique count: {unique_values}')
        print(val_data)
        if len(unique_values[0])>2 or 2 in val_data.edge_label: 
            print(f'=== 2 in val_data')
            val_data = custom_replace(val_data)
        unique_values = test_data.edge_label.unique(return_counts=True)
        print(f'\ntest_data_homo.edge_label.unique count: {unique_values}')
        print(test_data)
        if len(unique_values[0])>2 or 2 in test_data.edge_label: 
            print(f'=== 2 in test_data')
            test_data = custom_replace(test_data)

        model = HomoModel(
            metadata,
            args.gnn_method,
            hidden_channels=args.hidden_channels,
        )
    else:
        model = HeteroModel(
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
                label_type,
                hidden_channels=args.hidden_channels, #64 # 128
                node_types=data.node_types
            )
    print()
    print('== Begin training network ==')
    # with torch.no_grad():  # Initialize lazy modules.
    #     out = model(data.x_dict, data.edge_index_dict)
    ## Train model
    if args.gnn_method == 'SAGEConv':
        if args.contain_model_feature:
            epochs = 30
        else: epochs = 30
    elif args.gnn_method == 'GATConv':
        epochs = 30
    else:
        epochs = 30

    if 'homo' in args.gnn_method:
        graph_type = 'homo'
    else:
        graph_type = 'hetero'

    model,loss, train_time = train(model,train_data,graph_type,label_type,batch_size,epochs)

    evaluation_dict['epochs'] = epochs
    evaluation_dict['loss'] = loss
    evaluation_dict['train_time'] = train_time
    evaluation_dict['hidden_channels'] = args.hidden_channels
    print()

    ## Validate model on validation set
    val_AUC = validate(model,val_data,graph_type,label_type,batch_size=batch_size)
    evaluation_dict['val_AUC'] = val_AUC
    ## Evaluate model on test set
    test_AUC = validate(model,test_data,graph_type,label_type,batch_size=batch_size)
    evaluation_dict['test_AUC'] = test_AUC

    print()
    print('========== predict model nodes for the test dataset ========')
    print(unique_dataset_id[unique_dataset_id['dataset']==args.test_dataset])
    
    dataset_index = data_dict['test_dataset_idx']
    dataset_index = np.repeat(dataset_index,len(data_dict['model_idx']))
    
    if graph_type == 'hetero':
        # data['model'].node_id
        data[s,r,t].edge_label_index = torch.stack([
            torch.from_numpy(data_dict['model_idx']).to(torch.int64),torch.from_numpy(dataset_index).to(torch.int64)],dim=0)
    elif graph_type == 'homo':
        data.edge_label_index = torch.stack([torch.from_numpy(data_dict['model_idx']).to(torch.int64),torch.from_numpy(dataset_index).to(torch.int64)],dim=0)
    
    pred = predict_model_for_dataset(model,data)
    # print(f'\npred: {pred}')
    print(f'len(pred): {len(pred)}')
    norm = np.linalg.norm(pred)     # To find the norm of the array
    # print(norm)                        # Printing the value of the norm
    normalized_pred = pred/norm 
    print(normalized_pred[:5])

    # Save graph embedding distance results
    dir_path = os.path.join('./rank',f'{args.test_dataset}', args.gnn_method)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    results = pd.DataFrame(data_dict['model_idx'],columns=['mappedID'])
    results['score'] = normalized_pred.tolist()
    results = pd.merge(results,unique_model_id,how='left',on='mappedID')
    # unique_model_id['score'] = normalized_pred.tolist()
    # np.save(os.path.join(dir_path,config_name+'.npy'),pred)
    # unique_model_id.to_csv(os.path.join(dir_path,config_name+'.csv'))
    save_path = os.path.join(dir_path,config_name+'.csv')
    try:
        results.to_csv(save_path)
    except:
        results.to_csv(os.path.join(dir_path,'results.csv'))

     
    df_perf = pd.concat([df_perf,pd.DataFrame(evaluation_dict,index=[0])],ignore_index=True)
    print()
    print('======== save =======')
    # save the 
    df_perf.to_csv(args.path)

    return results, save_path