import numpy as np
import pandas as pd
import torch
from torch import Tensor
print(torch.__version__)
import os
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.data import download_url, extract_zip
# We can make use of the `loader.LinkNeighborLoader` from PyG:
from torch_geometric.loader import LinkNeighborLoader
from graph import Graph
import tqdm
import torch.nn.functional as F

def get_unique_node(col,name):
    unique_id = col.unique()
    unique_id = pd.DataFrame(data={
            name: unique_id,
            'mappedID': pd.RangeIndex(len(unique_id)),
        })
    return unique_id

def merge(df1,df2,col_name):
    mapped_id = pd.merge(df1, df2,left_on=col_name, right_on=col_name, how='left')
    mapped_id = torch.from_numpy(mapped_id['mappedID'].values)
    return mapped_id

from torch_geometric.data import download_url, extract_zip

url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
extract_zip(download_url(url, '.'), '.')

def get_df():
    movies_path = './ml-latest-small/movies.csv'
    ratings_path = './ml-latest-small/ratings.csv'

    # Load the entire movie data frame into memory:
    movies_df = pd.read_csv(movies_path, index_col='movieId')

    # Split genres and convert into indicator variables:
    genres = movies_df['genres'].str.get_dummies('|')
    print(genres[["Action", "Adventure", "Drama", "Horror"]].head())
    print(genres.values)
    # Use genres as movie input features:
    movie_feat = torch.from_numpy(genres.values).to(torch.float)
    assert movie_feat.size() == (9742, 20)  # 20 genres in total.

    # Load the entire ratings data frame into memory:
    ratings_df = pd.read_csv(ratings_path)

    return movies_df, ratings_df, movie_feat

def preprocess():
    movies_df, ratings_df, movie_feat = get_df()

    # Create a mapping from unique user indices to range [0, num_user_nodes):
    unique_user_id = ratings_df['userId'].unique()
    unique_user_id = pd.DataFrame(data={
        'userId': unique_user_id,
        'mappedID': pd.RangeIndex(len(unique_user_id)),
    })
    print("Mapping of user IDs to consecutive values:")
    print("==========================================")
    print(unique_user_id.head())
    print()
    # Create a mapping from unique movie indices to range [0, num_movie_nodes):
    unique_movie_id = ratings_df['movieId'].unique()
    unique_movie_id = pd.DataFrame(data={
        'movieId': unique_movie_id,
        'mappedID': pd.RangeIndex(len(unique_movie_id)),
    })
    print("Mapping of movie IDs to consecutive values:")
    print("===========================================")
    print(unique_movie_id.head())
    # Perform merge to obtain the edges from users and movies:
    ratings_user_id = pd.merge(ratings_df['userId'], unique_user_id,
                                left_on='userId', right_on='userId', how='left')
    ratings_user_id = torch.from_numpy(ratings_user_id['mappedID'].values)
    ratings_movie_id = pd.merge(ratings_df['movieId'], unique_movie_id,
                                left_on='movieId', right_on='movieId', how='left')
    ratings_movie_id = torch.from_numpy(ratings_movie_id['mappedID'].values)
    # With this, we are ready to construct our `edge_index` in COO format
    # following PyG semantics:
    edge_index_user_to_movie = torch.stack([ratings_user_id, ratings_movie_id], dim=0)
    assert edge_index_user_to_movie.size() == (2, 100836)
    print()
    print("Final edge indices pointing from users to movies:")
    print("=================================================")
    print(f'edge_index_user_to_movie.dtype: {edge_index_user_to_movie.dtype}')
    print(edge_index_user_to_movie)

    return  {   
                'unique_user_id':unique_user_id,
                'unique_movie_id':unique_movie_id,
                'movie_feat':movie_feat,
                'edge_index_user_to_movie':edge_index_user_to_movie,
                'edge_attr_model_to_dataset': np.array(0),
                'edge_index_dataset_to_dataset':np.array(0)

            }

data_dict = preprocess()
unique_user_id = data_dict['unique_user_id']
movie_feat = data_dict['movie_feat']
unique_movie_id = data_dict['unique_movie_id']
edge_index_user_to_movie = data_dict['edge_index_user_to_movie']
# edge_attr_model_to_dataset = data_dict['edge_attr_model_to_dataset']
# edge_index_dataset_to_dataset = data_dict['edge_index_dataset_to_dataset']

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

data = HeteroData()
# Save node indices:
data["user"].node_id = torch.arange(len(unique_user_id))
data["movie"].node_id = torch.arange(movie_feat.shape[0]) #torch.arange(len(movies_df))
# Add the node features and edge indices:
data["movie"].x = movie_feat
print()
print('==========')
print(f'data["movie"].x.shape: {data["movie"].x.shape}')
print(f'dtype: {data["movie"].x.dtype}')
print(data["movie"].x)
data["user", "rates", "movie"].edge_index = edge_index_user_to_movie

print()
print('======= data ========')
print(data)
# We also need to make sure to add the reverse edges from movies to users
# in order to let a GNN be able to pass messages in both directions.
# We can leverage the `T.ToUndirected()` transform for this from PyG:
data = T.ToUndirected()(data)

print()
print('======= data - ToUndirected ========')
print(data)

transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=("user", "rates", "movie"),
    rev_edge_types=("movie", "rev_rates", "user"), 
)
train_data, val_data, test_data = transform(data)


# In the first hop, we sample at most 20 neighbors.
# In the second hop, we sample at most 10 neighbors.
# In addition, during training, we want to sample negative edges on-the-fly with
# a ratio of 2:1.
# We can make use of the `loader.LinkNeighborLoader` from PyG:
from torch_geometric.loader import LinkNeighborLoader

# Define seed edges:
edge_label_index = train_data["user", "rates", "movie"].edge_label_index
edge_label = train_data["user", "rates", "movie"].edge_label
print()
print('edge_label_index')
print('=======')
print(edge_label_index)
print()
print('edge_label')
print('=======')
print(edge_label)      

train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(("user", "rates", "movie"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=True,
)

from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        print(f'x.shape: {x.size()}')
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.movie_lin = torch.nn.Linear(20, hidden_channels)
        print()
        print("== self.movie_lin.state_dict()['weight'].shape")
        print(self.movie_lin.state_dict()['weight'].shape)

        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.movie_emb = torch.nn.Embedding(data["movie"].num_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()
    def forward(self, data: HeteroData) -> Tensor:
        # print('--------')
        # print(f'data["movie"].x.shape:{data["movie"].x.shape}')
        # print(data['movie'].x)
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

model = Model(hidden_channels=64)

import tqdm
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1, 6):
    total_loss = total_examples = 0
    for sampled_data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        sampled_data.to(device)
        # print(sampled_data)
        pred = model(sampled_data)
        ground_truth = sampled_data["user", "rates", "movie"].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

        
# Evaluate
# Define the validation seed edges:
edge_label_index = val_data["user", "rates", "movie"].edge_label_index
edge_label = val_data["user", "rates", "movie"].edge_label
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20, 10],
    edge_label_index=(("user", "rates", "movie"), edge_label_index),
    edge_label=edge_label,
    batch_size=3 * 128,
    shuffle=False,
)
sampled_data = next(iter(val_loader))

from sklearn.metrics import roc_auc_score
preds = []
ground_truths = []
for sampled_data in tqdm.tqdm(val_loader):
    with torch.no_grad():
        sampled_data.to(device)
        preds.append(model(sampled_data))
        ground_truths.append(sampled_data["user", "rates", "movie"].edge_label)
pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
auc = roc_auc_score(ground_truth, pred)
print()
print(f"Validation AUC: {auc:.4f}")