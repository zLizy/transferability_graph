import time
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
import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")

import itertools
import scipy.spatial.distance as distance
# distance.cosine(h1, h2)

import sys
sys.path.append('../')

def get_dataset_edge_index(features,method='cosine'):
    n = features.shape[0]
    print(f'len(dataset_features):{n}')
    thres = 0.6
    distance_matrix = np.zeros([n,n])
    data_source = []
    data_target = []
    for (i, e1), (j, e2) in itertools.combinations(enumerate(features), 2):
        similarity = distance.cosine(e1, e2)
        distance_matrix[i, j] = similarity
        if similarity > thres:
            data_source.append(i)
            data_target.append(j)
        # distance_matrix[j, i] = distance_matrix[i, j]
    # data_source = np.asarray(data_source)
    # data_target = np.asarray(data_target)
    print(f'len(data_source):{len(data_source)}')
    return torch.stack([torch.tensor(data_source), torch.tensor(data_target)]) #dim=0
    

def get_unique_node(col,name):
    unique_id = col.unique()
    unique_id = pd.DataFrame(data={
            name: unique_id,
            'mappedID': pd.RangeIndex(len(unique_id)),
        })
    return unique_id

def merge(df1,df2,col_name):
    mapped_id = pd.merge(df1, df2,left_on=col_name, right_on=col_name, how='left')# how='left
    mapped_id = torch.from_numpy(mapped_id['mappedID'].values)
    return mapped_id


## Train
def train(model,train_data,batch_size=4,epochs=5):
    start = time.time()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,capturable=True)
    train_loader = get_dataloader(train_data,batch_size=batch_size,is_train=True)
    # print("Capturing:", torch.cuda.is_current_stream_capturing())
    # torch.cuda.empty_cache()
    for epoch in range(1, epochs+1):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            # print(sampled_data['dataset'].x)
            sampled_data.to(device)
            # print(sampled_data)
            # print(sampled_data['dataset'].x)
            pred = model(sampled_data)
            ground_truth = sampled_data["model", "trained_on", "dataset"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
        if (total_loss / total_examples) < 1: break
    train_time = time.time()-start
    return model,round(total_loss/total_examples,4), train_time


# Dataloader
def get_dataloader(data,batch_size=8,is_train=False):
    print('== get dataloader ==')
    # Define the validation seed edges:
    edge_label_index = data["model", "trained_on", "dataset"].edge_label_index
    # print(f'== edge_label_indx.dtype: {edge_label_index.dtype}')
    edge_label = data["model", "trained_on", "dataset"].edge_label
    if is_train:
        dataloader = LinkNeighborLoader(
            data=data,
            num_neighbors=[8,4],#[20, 10],
            neg_sampling_ratio=2.0,
            edge_label_index=(("model", "trained_on", "dataset"), edge_label_index),
            edge_label=edge_label,
            batch_size=batch_size,
            shuffle=True,
        )
    else:
        dataloader = LinkNeighborLoader(
            data=data,
            num_neighbors=[8,4], #[20, 10],
            edge_label_index=(("model", "trained_on", "dataset"), edge_label_index),
            edge_label=edge_label,
            batch_size=batch_size,
            shuffle=False,
        )
    ## print a batch
    # batch = next(iter(dataloader))
    # print("Batch:", batch)
    # print("batch edge_index", batch["model", "trained_on", "dataset"].edge_index)
    # print("Labels:", batch["model", "trained_on", "dataset"].edge_label)
    # print("Batch indices:", batch["model", "trained_on", "dataset"].edge_label_index)
    
    # sampled_data = next(iter(val_loader))
    return dataloader
    

def validate(model,val_data,batch_size=8):
    preds = []
    ground_truths = []
    val_loader = get_dataloader(val_data,batch_size=batch_size)
    for sampled_data in tqdm.tqdm(val_loader):
        with torch.no_grad():
            sampled_data.to(device)
            preds.append(model(sampled_data))
            ground_truths.append(sampled_data["model", "trained_on", "dataset"].edge_label)
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    print()
    print(f"Validation AUC: {auc:.4f}")
    return auc

def predict_model_for_dataset(model,data,gnn_method='SageConv'):
    preds = []
    # ground_truths = []
    # map all the model nodes to the dataset node
    # model - dataset
    # print()
    # print('============')
    # print(f'dataset_index: {dataset_index}')
    # print(f"data['model'].num_nodes: {data['model'].num_nodes}")

    with torch.no_grad():
        data.to(device)
        preds.append(model(data))
        # ground_truths.append(edge_label)
    # pred = torch.cat(preds, dim=0).cpu().numpy()
    pred = preds[0].cpu().numpy()
    return pred
    # ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    # auc = roc_auc_score(ground_truth, pred)
    # print()
    # print(f"Validation AUC: {auc:.4f}")
    # return auc