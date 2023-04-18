from transformers import AutoImageProcessor, ResNetModel,ConvNextModel
from torch.utils.data import DataLoader
from torch.utils import data
from torchvision import transforms
import torch
import torchvision
from tqdm import tqdm
import numpy as np
import random
random.seed(10)

import os
import sys
sys.path.append('../')
sys.path.append('../..')
from util import dataset
from util.tfds import VTABIterableDataset

MAX_NUM_SAMPLES = 5000
FEATURE_DIM = 2048
GET_FEATURE = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Check output dimension -> it should be 2048
model_name = 'resnet50'
# model = torchvision.models.resnet50(weights="DEFAULT")
model = ResNetModel.from_pretrained("microsoft/resnet-50").to(device)
# model = ConvNextModel.from_pretrained("facebook/convnext-base-224-22k").to('cuda')


## Load datasets
datasets_list = [
                    'food101','cifar10','cifar100','caltech101',
                    'stanfordcars','eurosat','clevr_count_all','clevr_closest_object_distance',
                    'dmlab', 'kitti_closest_vehicle_distance','flowers','pets',
                    'pcam','sun397','smallnorb_label_azimuth','smallnorb_label_elevation',
                    'svhn','resisc45','diabetic_retinopathy',
                    'cats_vs_dogs','keremberke/pokemon-classification','beans','poolrf2001/mask',
                    'Matthijs/snacks','keremberke/chest-xray-classification'
                ]

for dataset_name in datasets_list[3:]:
    dataset_name = dataset_name.replace('/','_').replace('-','_')
    print(f'=========== dataset_name: {dataset_name} ===============')
    ds = dataset.__dict__[dataset_name]('../../datasets/')[0]
    if len(ds)<MAX_NUM_SAMPLES: 
        LEN = len(ds)
    else:
        LEN = MAX_NUM_SAMPLES
    print(f'dataset size: {len(ds)}')
    print(type(ds))

    # dataloader = DataLoader(
    #                 ds,
    #                 batch_size=64, # may need to reduce this depending on your GPU 
    #                 num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
    #                 shuffle=False,
    #                 drop_last=False,
    #                 pin_memory=True
    #             )
    # Feature extraction.
    features = torch.zeros(1,FEATURE_DIM).to(device)
    print(f'features.shape: {features.shape}')
    labels = torch.zeros(1).to(device)
    print_flag = True
    if not isinstance(ds,VTABIterableDataset):
        print('is tfds type')
        idx = random.sample(range(len(ds)), k=LEN)
        ds = torch.utils.data.Subset(ds, idx)
        dataloader = DataLoader(
                    ds,
                    batch_size=64, # may need to reduce this depending on your GPU 
                    num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True
                )
        print(f'dataloader size: {len(dataloader)}')
        with torch.no_grad():
            for x, y in tqdm(dataloader):
                if GET_FEATURE:
                    output = model(x.to(device))
                    if print_flag:
                        print(f'output.pooler_output: {output.pooler_output.shape}')
                        print_flag = False
                    feature = torch.reshape(output.pooler_output,(len(y),FEATURE_DIM))
                    features = torch.cat((features,feature),0)
                labels = torch.cat((labels,y.to(device)),0)
    else:
        for (x,y) in tqdm(ds):
                # inputs = data[0]#(data['image'].numpy())
                # y = data[1] #data['label'].numpy()
                # for x, label in zip(inputs, y):
                #     # if self.target_transform is not None:
                #     #     label = self.target_transform(label)
                #   print(f'x.shape:{x.shape}')
                    if GET_FEATURE:
                        x = torch.reshape(x,(1,)+x.shape)
                        output = model(x.to(device))
                        if print_flag:
                            print(f'output.pooler_output: {output.pooler_output.shape}')
                            print_flag = False
                        feature = torch.reshape(output.pooler_output,(1,FEATURE_DIM))
                        features = torch.cat((features,feature),0)
                    # labels = torch.cat((labels,y),0)

    
    if not os.path.exists(os.path.join('./feature', model_name)):
        os.makedirs(os.path.join('./feature', model_name))
    features = features.cpu().detach().numpy()
    features = features[1:]
    labels = labels.cpu().detach().numpy()
    labels = labels[1:]
    unique,counts = np.unique(labels,return_counts=True)
    save_dir = os.path.join('./feature', model_name,dataset_name)
    if GET_FEATURE:
        # np.save(os.path.join(save_dir + f'_feature_{MAX_NUM_SAMPLES}.npy'), features)
        sorted_label = sorted(list(set(labels)))
        feature_per_class = np.zeros((len(sorted_label), FEATURE_DIM), dtype=np.float32)
        counter = 0
        for i in sorted_label:
            idx = [(l==i) for l in labels]
            feature_per_class[counter, :] = np.mean(features[idx, :], axis=0)
            counter += 1
        np.save(os.path.join(save_dir + f'_feature.npy'), feature_per_class)
    # np.save(os.path.join(save_dir + f'_label_{MAX_NUM_SAMPLES}.npy'), labels)
    # np.save(os.path.join(save_dir + f'_weight_{MAX_NUM_SAMPLES}.npy'), counts)