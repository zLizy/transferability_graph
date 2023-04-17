from transformers import AutoImageProcessor, ResNetModel
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torchvision
from tqdm import tqdm
import numpy as np
import random
import os
import sys
sys.path.append('../')
sys.path.append('../..')
from util import dataset

MAX_NUM_SAMPLES = 5000
FEATURE_DIM = 2048

### Check output dimension -> it should be 2048
model_name = 'resnet50'
# model = torchvision.models.resnet50(weights="DEFAULT")
model = ResNetModel.from_pretrained("microsoft/resnet-50").to('cuda')

## Load datasets
datasets_list = [
                    'food101','cifar10','cifar100','caltech101',
                    'cars','eurosat','clevr_count_all','clevr_closest_object_distance',
                    'dmlab', 'kitti_closest_vehicle_distance',
                    'flowers','pets','pcam','sun397'
                ]

for dataset_name in datasets_list:
    ds = dataset.__dict__[dataset_name]('../../datasets/')[0]
    idx = random.sample(range(len(ds)), k=MAX_NUM_SAMPLES)
    ds = torch.utils.data.Subset(ds, idx)
    dataloader = DataLoader(
                    ds,
                    batch_size=64, # may need to reduce this depending on your GPU 
                    num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True
                )
    # Feature extraction.
    features = torch.zeros(1,FEATURE_DIM).to('cuda')
    print(f'features.shape: {features.shape}')
    labels = torch.zeros(1).to('cuda')

    with torch.no_grad():
        for x, y in tqdm(dataloader):
            # print(f'len of a batch: {len(y)}')
            output = model(x.cuda())
            feature = torch.reshape(output.pooler_output,(len(y),FEATURE_DIM))
            features = torch.cat((features,feature),0)
            labels = torch.cat((labels,y.cuda()),0)
    if not os.path.exists(os.path.join('./feature', model_name)):
        os.makedirs(os.path.join('./feature', model_name))
    features = features.cpu().detach().numpy()
    features = features[1:]
    labels = labels.cpu().detach().numpy()
    labels = labels[1:]
    unique,counts = np.unique(labels,return_counts=True)
    save_dir = os.path.join('./feature', model_name,dataset_name)
    np.save(os.path.join(save_dir + f'_feature_{MAX_NUM_SAMPLES}.npy'), features)
    np.save(os.path.join(save_dir + f'_label_{MAX_NUM_SAMPLES}.npy'), labels)
    np.save(os.path.join(save_dir + f'_weight_{MAX_NUM_SAMPLES}.npy'), counts)