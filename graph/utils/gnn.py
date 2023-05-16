from torch_geometric.nn import SAGEConv,GATConv, HGTConv,to_hetero,Linear
import torch.nn.functional as F
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
# cuda issue
import os
# os.system('unset LD_LIBRARY_PATH')
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class EebedGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        # self.conv1 = SAGEConv(-1, hidden_channels)
        # self.conv1 = SAGEConv((-1, -1), hidden_channels)
        # self.conv2 = SAGEConv(-1, hidden_channels)
        # self.conv1 = GCNConv(-1, hidden_channels) 
        # self.conv1 = GATConv((-1, -1), hidden_channels,add_self_loops=False)
        # self.lin1 = Linear(-1, hidden_channels)
        # self.conv2 = GATConv((-1, -1), hidden_channels,add_self_loops=False)
        # self.lin2 = Linear(-1, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor: #: Tensor
        # print(f'x.shape: {x.shape}')
        # Define a 2-layer GNN computation graph.
        # Use a *single* `ReLU` non-linearity in-between.
        x = F.relu(self.conv1(x, edge_index)) #+ self.lin1(x)
        x = self.conv2(x, edge_index) #+ self.lin2(x)
        return x
        # raise NotImplementedError

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1,-1), hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor: #: Tensor
        x = F.relu(self.conv1(x, edge_index)) #+ self.lin1(x)
        x = self.conv2(x, edge_index) #+ self.lin2(x)
        return x

class Model(torch.nn.Module):
    def __init__(
        self,
        metadata,
        num_dataset_nodes,
        num_model_nodes,
        dim_model_feature,
        dim_dataset_feature,
        contain_model_feature,
        contain_dataset_feature,
        embed_model_feature,
        embed_dataset_feature,
        gnn_method,
        hidden_channels,
        node_types=None
    ):
        super().__init__()
        self.contain_model_feature = contain_model_feature
        self.contain_dataset_feature = contain_dataset_feature
        self.embed_model_feature = embed_model_feature
        self.embed_dataset_feature = embed_dataset_feature
        self.gnn_method = gnn_method

        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        # print()
        # print('=========')
        # print(f'data["dataset"].x.shape: {data["dataset"].x.shape[0]},{data["dataset"].x.shape[1]}')
        if self.contain_dataset_feature:
            self.dataset_lin = torch.nn.Linear(dim_dataset_feature, hidden_channels)# data["dataset"].x.shape[1]
        if self.contain_model_feature:
            self.model_lin = torch.nn.Linear(dim_model_feature, hidden_channels)# data["dataset"].x.shape[1]
        # print()
        # print("== self.dataset_lin.state_dict()['weight'].shape")
        # print(self.dataset_lin.state_dict()['weight'].shape)

        self.model_emb = torch.nn.Embedding(num_model_nodes, hidden_channels)# data["model"].num_nodes
        self.dataset_emb = torch.nn.Embedding(num_dataset_nodes, hidden_channels)# data["dataset"].num_nodes
        
        ## Instantiate homogeneous GNN:
        if self.gnn_method == 'SAGEConv':
            self.gnn = GNN(hidden_channels)
        elif self.gnn_method == 'GATConv':
            self.gnn = GAT(hidden_channels)
        elif self.gnn_method == 'HGTConv':
            self.gnn = HGT(hidden_channels,node_types,metadata)

        # Convert GNN model into a heterogeneous variant:
        if self.gnn_method != 'HGTConv':
            self.gnn = to_hetero(self.gnn, metadata=metadata)#,aggr='sum')

        self.classifier = Classifier()
        self.flag = True

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {}
        if self.contain_model_feature:
            if self.embed_model_feature:
                # any meaning?
                # print('===============')
                # print(f'data["model"].node_id.shape: {data["model"].node_id.shape}')
                # print(f'data["model"].x.shape: {data["model"].x.shape}')
                try:
                    model_emb = self.model_lin(data['model'].x) #+ self.model_emb(data["model"].node_id)
                except:
                    x = data["model"].x
                    x = torch.index_select(x,0,data['model'].node_id)
                    model_emb = self.model_lin(x) + self.model_emb(data["model"].node_id)
                    del x
            else:
                model_emb = data['model'].x
            # x_dict['model'] = model_emb
        else:
            if self.embed_model_feature:
                model_emb = self.model_emb(data["model"].node_id)
 
        if self.contain_dataset_feature:
            if self.embed_dataset_feature:
                dataset_emb = self.dataset_lin(data["dataset"].x) #+ self.dataset_emb(data["dataset"].node_id)
            else:
                dataset_emb = data['dataset'].x
            
        else:
            if self.embed_dataset_feature:
                dataset_emb = self.dataset_emb(data["dataset"].node_id)
        x_dict = {
            'model': model_emb,
            'dataset': dataset_emb
        } 
        if self.flag:
            print(f'data["dataset"].x:{data["dataset"].x.shape}')
            print(f'data["dataset"].node_id:{data["dataset"].node_id.shape}')
            self.flag=False
        
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        self.x_dict = self.gnn(x_dict,data.edge_index_dict)#(x_dict, data.edge_index_dict)

        pred = self.classifier(
            self.x_dict["model"],
            self.x_dict["dataset"],
            data["model", "trained_on", "dataset"].edge_label_index,
        )
        return pred

        


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, node_types, metadata, num_heads=2, num_layers=2):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata,
                           num_heads, group='sum')
            self.convs.append(conv)

        # self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict
        # return self.lin(x_dict['author'])

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin2 = Linear(-1, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_model: Tensor, x_dataset: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_model = x_model[edge_label_index[0]]
        edge_feat_dataset= x_dataset[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_model * edge_feat_dataset).sum(dim=-1)


