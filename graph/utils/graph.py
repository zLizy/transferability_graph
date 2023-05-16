import torch
import numpy as np
import pandas as pd
from torch import Tensor
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.data import Data
# from utils.CustomRandomLinkSplit import RandomLinkSplit

import os
os.system('unset LD_LIBRARY_PATH')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class HGraph:
    def __init__(   self,
                    unique_model_id,
                    model_features,
                    unique_dataset_id,
                    dataset_features,
                    edge_index_model_to_dataset,
                    edge_index_dataset_to_dataset,
                    edge_attr_model_to_dataset,
                    contain_data_similarity=True,        
                    contain_dataset_feature=False,
                    contain_model_feature=False,
                    add_attr=False
                ):
        # self.prep()
        self.data = HeteroData()
        # Save node indices:
        self.data["model"].node_id = torch.arange(len(unique_model_id))
        if contain_model_feature:
            self.data["model"].x = torch.from_numpy(model_features).to(torch.float)

        self.data["dataset"].node_id = torch.arange(len(unique_dataset_id))
        # self.data["dataset"].x = dataset_features
        # self.data["dataset"].x = torch.from_numpy(dataset_features).to(torch.float)
        # datset_features = np.around(np.random.random_sample((len(dataset_features), 128))+0.00001,3)
        # dataset_features = np.random.randint(0,1,size=(len(dataset_features),20))
        if contain_dataset_feature:
            self.data['dataset'].x = torch.from_numpy(dataset_features).to(torch.float)
        # print()
        # print('self.data["dataset"].x.shape')
        # print('========')
        # print(self.data["dataset"].x.dtype)
        # print(self.data["dataset"].x.shape)
        

        # self.data["model"].x = model_features
        self.data["model", "trained_on", "dataset"].edge_index = edge_index_model_to_dataset  # TODO
        if contain_data_similarity:
            self.data["dataset", "similar_to", "dataset"].edge_index = edge_index_dataset_to_dataset  # TODO
        if add_attr:
            self.data['model', 'trained_on', 'dataset'].edge_attr = edge_attr_model_to_dataset # TODO
        self.transform()
        # self.split()
        self._print()

    def transform(self):
        self.data = T.ToUndirected()(self.data)
        # self.data = T.AddSelfLoops()(self.data)
        # self.data = T.NormalizeFeatures()(self.data)

    def _print(self):
        print()
        print("Data:")
        print("==============")
        print(self.data)
        print(self.data.metadata())
        print(self.data['model'].num_nodes)
        print("self.data['model'].node_id.shape")
        print(self.data['model'].node_id.shape)
        num_edges = self.data["model", "trained_on", "dataset"].num_edges
        print(f'self.data["model", "trained_on", "dataset"].num_edges: {num_edges}')
        # num_edges = self.data["dataset", "similar_to", "dataset"].num_edges
        # print(f'self.data["dataset", "similar_to", "dataset"].num_edges: {num_edges}')
        # print("self.data['dataset'].x")
        # print(self.data['dataset'].x)
        # print(self.data["model", "trained_on", "dataset"].edge_label_index)
        # print(self.data["model", "trained_on", "dataset"].edge_label)
        print()

    def split(self,num_val=0.1,num_test=0.2):
        # transform = RandomLinkSplit(
        transform = T.RandomLinkSplit(
        
            num_val=num_val,  # TODO
            num_test=num_test,  # TODO
            disjoint_train_ratio=0.3,  # TODO
            neg_sampling_ratio=2.0,  # TODO
            add_negative_train_samples=False,  # TODO
            # edge_types=self.data.metadata()[0], #("model", "trained_on", "dataset"),
            edge_types = ("model", "trained_on", "dataset"),
            is_undirected=True,
            # rev_edge_types = self.data.metadata()[1]
            rev_edge_types=("dataset", "rev_trained_on", "model"), 
        )
        # self.train_data, self.val_data, self.test_data = # self.val_data, 
        return transform(self.data) 

class Graph():
    def __init__(   self,
                    edge_index_model_to_dataset,
                    edge_index_dataset_to_dataset,
                ):

        max_model_id = int(torch.max(edge_index_model_to_dataset[0,:]).item()) + 1
        edge_index_model_to_dataset[1,:] += max_model_id
        edge_index_dataset_to_dataset += max_model_id
        edge_index = torch.cat((edge_index_model_to_dataset, edge_index_dataset_to_dataset), 1)
        self.data = Data(edge_index=edge_index)