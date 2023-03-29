import torch
import pandas as pd
from torch import Tensor
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

class Graph:
    def __init__(self,model_features,dataset_features,edge_index_model_to_dataset):
        # self.prep()
        self.data = HeteroData()
        # Save node indices:
        self.data["model"].node_id = torch.arange(len(model_features))
        self.data["dataset"].node_id = torch.arange(len(dataset_features))
        self.data["dataset"].x = dataset_features
        self.data["model"].x = model_features
        self.data["model", "trained_on", "dataset"].edge_index = edge_index_model_to_dataset  # TODO
        # data['paper', 'cites', 'paper'].edge_attr = ... # [num_edges_cites, num_features_cites]
        self.data = T.ToUndirected()(self.data)
        # self.split()
        self._print()

    def _print(self):
        print(self.data)
        # print("Training data:")
        # print("==============")
        # print(self.train_data)
        # print(self.train_data["model", "trained_on", "dataset"].num_edges)
        # print(self.train_data["model", "trained_on", "dataset"].edge_label_index)
        # print(self.train_data["model", "trained_on", "dataset"].edge_label)
        # print()
        # print("Validation data:")
        # print("================")
        # print(self.val_data)
    
    def split(self,num_val=0.1,num_test=0.1):
        transform = T.RandomLinkSplit(
            num_val=num_val,  # TODO
            num_test=num_test,  # TODO
            disjoint_train_ratio=0.3,  # TODO
            neg_sampling_ratio=2,  # TODO
            add_negative_train_samples=False,  # TODO
            edge_types=("model", "trained_on", "dataset"),
            rev_edge_types=("dataset", "rev_trained_on", "model"), 
        )
        self.train_data, self.val_data, self.test_data = transform(self.data)

    def get_unique_node(self,col,name):
        unique_id = col.unique()
        unique_id = pd.DataFrame(data={
            name: unique_id,
            'mappedID': pd.RangeIndex(len(unique_id)),
        })
        return unique_id

    def merge(self,df1,df2,col_name):
        mapped_id = pd.merge(df1, df2,left_on=col_name, right_on=col_name, how='left')
        mapped_id = torch.from_numpy(mapped_id['mappedID'].values)
        return mapped_id

    def prep(self):
        file = '../models/models.csv'
        df = pd.read_csv(file)
        print(df.head())
        self.unique_model_id = self.get_unique_node(df['model'],'model')
        self.unique_dataset_id = self.get_unique_node(df['dataset'],'dataset')
        print(self.unique_model_id.head())
        
        # Perform merge to obtain the edges from models and datasets:
        mapped_model_id = self.merge(df['model'],self.unique_model_id,'model')
        mapped_dataset_id = self.merge(df['dataset'],self.unique_dataset_id,'dataset')
        self.edge_index_model_to_dataset = torch.stack([mapped_model_id, mapped_dataset_id], dim=0)
        print(self.edge_index_model_to_dataset)

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor,predict_type: str) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        # if predict_type == 'dataset':
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.movie_lin = torch.nn.Linear(20, hidden_channels)
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.movie_emb = torch.nn.Embedding(data["movie"].num_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()
    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.user_emb(data["user"].node_id),
          "movie": self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id),
        } 
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )
        return pred