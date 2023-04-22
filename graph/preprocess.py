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

import sys
sys.path.append('../')

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
file = '../doc/model_config_dataset.csv'
df = pd.read_csv(file)
df['dataset'] = df['labels']
# df = df.loc[~df['dataset'].str.contains('imagenet')]

## Node idx
unique_model_id = get_unique_node(df['model'],'model')
unique_dataset_id = get_unique_node(df['dataset'],'dataset')
print(unique_model_id)
print(unique_dataset_id)

## Retrieve the embeddings of the dataset
data_feat = []
DATA_EMB_METHOD = 'domain_similarity'
FEATURE_DIM = 2048
for i, row in unique_dataset_id.iterrows():
    path = os.path.join(f'../dataset_embed/{DATA_EMB_METHOD}/feature','resnet50',row['dataset']+'_feature.npy')
    try:
        features = np.load(path)
    except Exception as e:
        print('----------')
        print(e)
        features = np.zeros((1,FEATURE_DIM))
    if (features == np.zeros((1,FEATURE_DIM))).all():
        print('Try to obtain missing features')
        try:
            from dataset_embed.domain_similarity.embed import embed
            # dataset_name = 'hfpics' if '[' in row['dataset'] else row['dataset']
            features = embed('../',row['dataset'])
            print('----------')
        except Exception as e:
            print(e)
            print('----------')
            features = np.zeros((1,FEATURE_DIM))
    features = np.mean(features,axis=0)
    print(f"shape of {row['dataset']} is {features.shape}")
    data_feat.append(features)
data_feat = np.stack(data_feat)
print(f'== data_feat.shape:{data_feat.shape}')

## Retrieve the embeddings of the model
model_feat = []
DATA_EMB_METHOD = 'attribution_map'
ATTRIBUTION_METHOD = 'saliency'
INPUT_SHAPE = 224
for i, row in unique_model_id.iterrows():
    model_match_rows = df.loc[df['model']==row['model']]
    dataset_name = model_match_rows['dataset'].values[0]
    IMAGE_SHAPE = int(model_match_rows['input_shape'].values[0])
    model_name = row['model'].replace('/','_')
    print(dataset_name,model_name)
    path = os.path.join(f'../model_embed/{DATA_EMB_METHOD}/feature',dataset_name,model_name+f'_{ATTRIBUTION_METHOD}.npy')
    try:
        features = np.load(path)
        features = np.mean(features,axis=0)
        if features.shape[0] != INPUT_SHAPE:
            print(f'== features.shape:{features.shape}')
            features = np.resize(features,(INPUT_SHAPE,INPUT_SHAPE))
    except Exception as e:
        print('----------')
        print(e)
        print('----------')
        features = np.zeros((INPUT_SHAPE,INPUT_SHAPE))
    model_feat.append(features)
model_feat = np.stack(model_feat)
print(f'== model_feat.shape:{model_feat.shape}')

## Perform merge to obtain the edges from models and datasets:
mapped_model_id = merge(df['model'],unique_model_id,'model')
mapped_dataset_id = merge(df['dataset'],unique_dataset_id,'dataset')
edge_index_model_to_dataset = torch.stack([mapped_model_id, mapped_dataset_id], dim=0)
print(f'== edge_index_model_to_dataset')
print(edge_index_model_to_dataset)


graph = Graph(model_feat,data_feat,edge_index_model_to_dataset)
