from transformers import AutoImageProcessor, ResNetModel, ConvNextModel, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from torch.utils import data
from torchvision import transforms
import torch
import torchvision
import tensorflow_datasets as tfds
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import random
random.seed(10)

import os
import sys
sys.path.append('../')
sys.path.append('../..')
import pandas as pd
from util import dataset
from util.tfds import VTABIterableDataset

MAX_NUM_SAMPLES = 5000
GET_FEATURE = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## model_dataset mapping
## visual datasets
# df = pd.read_csv('../../doc/model_config_dataset.csv')
## textual datasets
# df = pd.read_csv('../../doc/text_model_config_dataset.csv')

## Load datasets
datasets_list = [   "trec", "chest_xray_classification"]

for dataset_name in datasets_list[:]:
    dataset_name = dataset_name.replace('/','_').replace('-','_')
    print(f'=========== dataset_name: {dataset_name} ===============')
    if '[' in dataset_name: #  'hfpics'
        # labels = df[df['dataset']==dataset_name]
        classes = dataset_name.strip('][').replace("'","").split(', ')
        # print(f'classes: {classes}')
        # classes = "['corgi','']"
        ds_type = 'hfpics'
        ds = dataset.__dict__[ds_type]('../../datasets/', classes)
        train_dataset = ds.train_set
    else:
        dataset_name = dataset_name.lower()
        ds = dataset.__dict__[dataset_name]('../../datasets/')
        train_dataset = ds.train_set
    try:
        length = len(train_dataset)
    except:
        length = train_dataset.get_num_samples('train')
    if length < MAX_NUM_SAMPLES: 
        LEN = length
    else:
        LEN = MAX_NUM_SAMPLES
    print(f'dataset size: {length}')
    print(type(train_dataset))

    ## Load model
    model = ds.load_model(device)

    ## Feature initiation.
    features_tensor = torch.zeros(1, model.feature_dimension).to(device)
    print(f'features.shape: {features_tensor.shape}')
    labels_tensor = torch.zeros(1,).to(device)

    ## Load dataset
    print_flag = False
    # randomly sample MAX_NUM_SAMPLES
    idx = random.sample(range(length), k=LEN)
    train_dataset = torch.utils.data.Subset(train_dataset, idx)

    dataloader = DataLoader(
                    train_dataset,
                    batch_size=1, # may need to reduce this depending on your GPU
                    num_workers=0, # may need to reduce this depending on your num of CPUs and RAM
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True,
                )
    print(f'dataloader size: {len(dataloader)}')
    with torch.no_grad():
        for x,y in tqdm(dataloader):
            if GET_FEATURE:
                output = model.get_features(x, len(y))
                features_tensor = torch.cat((features_tensor, output), 0)
            labels_tensor = torch.cat((labels_tensor, y.to(device)), 0)

    
    if not os.path.exists(os.path.join('./feature', model.identifier)):
        os.makedirs(os.path.join('./feature', model.identifier))
    features_tensor = features_tensor.cpu().detach().numpy()
    features_tensor = features_tensor[1:]
    labels_tensor = labels_tensor.cpu().detach().numpy()
    labels_tensor = labels_tensor[1:]
    print(f'labels_tensor: {labels_tensor.shape}')
    # unique,counts = np.unique(labels_tensor,return_counts=True)
    save_dir = os.path.join('./feature', model.identifier, dataset_name)
    if GET_FEATURE:
        # np.save(os.path.join(save_dir + f'_feature_{MAX_NUM_SAMPLES}.npy'), features)
        sorted_label = sorted(list(set(labels_tensor)))
        feature_per_class = np.zeros((len(sorted_label), model.feature_dimension), dtype=np.float32)
        counter = 0
        for i in sorted_label:
            idx = [(l==i) for l in labels_tensor]
            feature_per_class[counter, :] = np.mean(features_tensor[idx, :], axis=0)
            counter += 1
        np.save(os.path.join(save_dir + f'_feature.npy'), feature_per_class)
    # np.save(os.path.join(save_dir + f'_label_{MAX_NUM_SAMPLES}.npy'), labels)
    # np.save(os.path.join(save_dir + f'_weight_{MAX_NUM_SAMPLES}.npy'), counts)