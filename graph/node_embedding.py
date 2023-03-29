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

MODEL_FEATURE_DIM = 1024
file = '../model_embed/models.csv'
df = pd.read_csv(file)
df = df.loc[~df['dataset'].str.contains('imagenet')]

unique_model_id = get_unique_node(df['model'],'model')
unique_dataset_id = get_unique_node(df['dataset'],'dataset')
print(unique_model_id)
print(unique_dataset_id)

# retrieve the embeddings of the dataset
data_feat = []
for i, row in unique_dataset_id.iterrows():
    path = os.path.join('../dataset_embed/feature','resnet50',row['dataset']+'.npy')
    features = np.load(path)
    features = np.mean(features,axis=0)
    print(f"shape of {row['dataset']} is {features.shape}")
    data_feat.append(features)
data_feat = np.stack(data_feat)
print(f'data_feat.shape:{data_feat.shape}')

# retrieve the embeddings of the model
model_feat = []
for i, row in unique_model_id.iterrows():
    dataset_name = df.loc[df['model']==row['model']]['dataset'].values[0]
    model_name = row['model'].replace('/','_')
    print(dataset_name,model_name)
    path = os.path.join('../model_embed/feature',dataset_name,model_name+'.npy')
    try:
        features = np.load(path)
        features = np.mean(features,axis=0)
    except Exception as e:
        print(e)
        features = np.zeros((MODEL_FEATURE_DIM,))
    model_feat.append(features)
model_feat = np.stack(model_feat)

# Perform merge to obtain the edges from models and datasets:
mapped_model_id = merge(df['model'],unique_model_id,'model')
mapped_dataset_id = merge(df['dataset'],unique_dataset_id,'dataset')
edge_index_model_to_dataset = torch.stack([mapped_model_id, mapped_dataset_id], dim=0)
print(edge_index_model_to_dataset)

graph = Graph(model_feat,data_feat,edge_index_model_to_dataset)



        
