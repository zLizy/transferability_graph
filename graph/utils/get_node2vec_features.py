#https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py

import time
import sys
import numpy as np
import pandas as pd

import torch
from utils.graph import Graph
from torch import Tensor
import torch_geometric.transforms as T
import os
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
# from torch_geometric.nn import Node2Vec
from utils.node2vec import Node2Vec
# import torch_cluster
#!pip install torch_cluster -f -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(f"Device: '{device}'")

SAVE_GRAPH = True

class Classifier(torch.nn.Module):
    def forward(self, model,edge_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_0 = model[edge_index[0]]
        edge_feat_1 = model[edge_index[1]]
        # edge_feat_1= x[edge_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_0 * edge_feat_1).sum(dim=-1)

class N2VPlusModel():
    def __init__(
            self,
            edge_index,
            embedding_dim=128,
            walk_length=6,#5,
            context_size=5,
            walks_per_node=5,
            num_negative_samples=1,
            negative_pairs=[],
            extend=True,
            epochs=30,
            ):
        self.epochs = epochs
        self.base = Node2VecPlus(
            edge_index,
            embedding_dim=embedding_dim,
            walk_length= walk_length, #20,
            context_size=context_size,
            walks_per_node=walks_per_node, #10,
            num_negative_samples=num_negative_samples,
            negative_pairs=negative_pairs,
            p=q,
            q=1,
            sparse=False,
        )

    def forward(self,data):
        pred = self.classifier(
            self.base(),
            data
        )
        return pred
        

class N2VModel(torch.nn.Module):
    def __init__(
        self,
        edge_index,
        edge_attr,
        node_IDs,
        embedding_dim=128,
        walk_length=6,#5,
        context_size=5,
        walks_per_node=5,
        num_negative_samples=1,
        negative_pairs=[],
        p=1,
        q=1,
        sparse=False, #True
        epochs=15,
        extend=False,
    ):
        super().__init__()
        self.epochs = epochs
        self.base = Node2Vec(
            edge_index,
            edge_attr,
            node_IDs,
            embedding_dim=embedding_dim,
            walk_length= walk_length, #20,
            context_size=context_size,
            walks_per_node=walks_per_node, #10,
            num_negative_samples=num_negative_samples,
            negative_pairs=negative_pairs,
            p=1,
            q=1,
            sparse=True,
            extend=extend,
        ).to(device)
        self.classifier = Classifier()
    
    def forward(self,data):
        pred = self.classifier(
            self.base(),
            data
        )
        return pred
    
    @torch.no_grad()
    def test(self):
        self.base.eval()
        z = self.base()
        # acc = model.test(z[data.train_mask], data.y[data.train_mask],
        #                  z[data.test_mask], data.y[data.test_mask],
        #                  max_iter=150)
        # return acc
    
    def train(self):
        start = time.time()
        num_workers = 0 if sys.platform.startswith('win') else 4
        loader = self.base.loader(batch_size=128, shuffle=True,
                            num_workers=1)
        optimizer = torch.optim.SparseAdam(list(self.base.parameters()), lr=0.01)
        
        for epoch in range(1, 1+self.epochs):
            self.base.train()
            total_loss = 0
            count = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                if count == 0:
                    # print()
                    # print('----------')
                    # print(f'pos_rw: {pos_rw}')
                    # print(f'neg_rw: {neg_rw}')
                    count += 1
                loss = self.base.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            loss = total_loss / len(loader)
            # acc = test()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}') #, Acc: {acc:.4f}')
        train_time = time.time()-start
        return loss, train_time

def node2vec_train(args,df_perf,data_dict,evaluation_dict,setting_dict,batch_size,extend=False):

    edge_index_accu_model_to_dataset = data_dict['edge_index_accu_model_to_dataset']
    edge_attr_accu_model_to_dataset = data_dict['edge_attr_accu_model_to_dataset']
    edge_index_tran_model_to_dataset = data_dict['edge_index_tran_model_to_dataset']
    edge_attr_tran_model_to_dataset = data_dict['edge_attr_tran_model_to_dataset']
    
    edge_index_dataset_to_dataset = data_dict['edge_index_dataset_to_dataset']
    edge_attr_dataset_to_dataset = data_dict['edge_attr_dataset_to_dataset']
    negative_pairs = data_dict['negative_pairs']
    node_IDs = data_dict['node_ID']
    ## Construct a graph
    if 'without_transfer' in args.gnn_method:
        without_transfer = True
    else:
        without_transfer = False
    graph = Graph(
        data_dict['node_ID'],
        edge_index_accu_model_to_dataset,
        edge_attr_accu_model_to_dataset,
        edge_index_tran_model_to_dataset,
        edge_attr_tran_model_to_dataset,
        edge_index_dataset_to_dataset,
        edge_attr_dataset_to_dataset,
        without_transfer=without_transfer,
        # max_model_id = data_dict['max_model_idx']
        )
    data = graph.data
    # data = T.ToUndirected(data)

    ### save thd graph
    setting_dict.pop('gnn_method')
    config_name = '_'.join([('{0}={1}'.format(k[:14], str(v)[:4])) for k,v in setting_dict.items()])
    if SAVE_GRAPH:
        if 'without_transfer' in args.gnn_method:
            gnn = 'walk_graph_without_transfer'
        else:
            gnn = 'walk_graph'

        if 'without_transfer' in args.gnn_method:
            config_name = 'without_transfer_'+config_name
        _dir = os.path.join('./saved_graph',f'{args.test_dataset}', gnn)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        # if not os.path.exists(os.path.join(_dir,config_name+'.pt')):
        torch.save(data, os.path.join(_dir,config_name+'.pt'))

    print(f'------- data: ----------')
    print(data)

    # settings
    epochs = 30
    evaluation_dict['epochs'] = epochs

    model = N2VModel(
        data.edge_index,
        data.edge_attr,
        node_IDs,
        negative_pairs=negative_pairs,
        epochs=epochs,
        extend=extend
        )
    loss, train_time = model.train()
    evaluation_dict['loss'] = loss
    evaluation_dict['train_time'] = train_time
    
    # save
    dataset_index = data_dict['test_dataset_idx'] #+ data_dict['max_model_idx'] + 1
    print('dataset_index',dataset_index)

    # dataset_index = np.repeat(dataset_index,len(data_dict['unique_model_id']))
    dataset_index = np.repeat(dataset_index, len(data_dict['model_idx']))#len(edge_index_accu_model_to_dataset[0,:]))
    print(f"len(model_index): {len(data_dict['model_idx'])}'")
    # edge_index = torch.stack([torch.from_numpy(data_dict['model_idx']).to(torch.int64),torch.from_numpy(dataset_index).to(torch.int64)],dim=0)
    edge_index = torch.stack([torch.from_numpy(data_dict['model_idx']).to(torch.int64),torch.from_numpy(dataset_index).to(torch.int64)],dim=0)
    # data["model", "trained_on", "dataset"].edge_label_index = torch.stack([data['model'].node_id,torch.from_numpy(dataset_index).to(torch.int64)],dim=0)
    # dataset_emb = model.base(dataset_index)
    # print(f'\nedge_index: {edge_index}')
    # print(f"node_ID: {data_dict['node_ID']}")
    from utils._util import predict_model_for_dataset
    pred = predict_model_for_dataset(model,edge_index,gnn_method='node2vec')
    print(pred[:5])        
    norm = np.linalg.norm(pred)     # To find the norm of the array
    # Printing the value of the norm
    normalized_pred = pred/norm 
    print(normalized_pred[:5])

    config_name = '_'.join([('{0}=={1}'.format(k[:14], str(v)[:5])) for k,v in setting_dict.items()])

    # Save graph embedding distance results
    dir_path = os.path.join('./rank',f'{args.test_dataset}', args.gnn_method)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    results = pd.DataFrame(data_dict['unique_model_id'])
    results['score'] = float('-inf') #normalized_pred.tolist()
    print(f'results: {results}')
    # results = results.reset_index().set_index('mappedID')
    for idx, p in zip(data_dict['model_idx'],pred):
        # print(f'idx: {idx}')
        if idx < len(results):
            results.loc[idx,'score'] = p

    # results = pd.merge(results,data_dict['unique_model_id'],how='left',on='mappedID')
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
    

    