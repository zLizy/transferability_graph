from transformers import SwinModel, ViTMAEModel, ConvNextModel,MobileViTModel, CvtModel, ResNetModel, MobileNetV1Model
from transformers import ViTModel, ViTForImageClassification, ResNetForImageClassification
from transformers import ConvNextFeatureExtractor, AutoImageProcessor, ViTFeatureExtractor
from torch.utils.data import TensorDataset, DataLoader
import requests
# from optimum.intel.openvino import OVModelForImageClassification
from tqdm import tqdm
import random
import os 
import sys
sys.path.append('../')
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
# from datasets import load_dataset
from task2vec_embed import dataset
from autoencoder import AE

epochs = 5
MAX_NUM_SAMPLES = 5000

_EXTRACTOR = {}
_CLASSIFIER = {}

# mean-squared error loss
criterion = nn.MSELoss()

def _add_extractor(extractor_fn):
    _EXTRACTOR[extractor_fn.__name__] = extractor_fn
    return extractor_fn

def _add_classifier(classifier_fn):
    _CLASSIFIER[classifier_fn.__name__] = classifier_fn
    return classifier_fn

def get_extractor(extractor,model_name):
    return _EXTRACTOR[extractor](model_name)

def get_classifier(classifier,model_name):
    return _CLASSIFIER[classifier](model_name)

@_add_classifier
def vitmodel(model_name):
    return ViTModel.from_pretrained(model_name)

@_add_classifier
def cvtmodel(model_name):
    return CvtModel.from_pretrained(model_name)

@_add_classifier
def convnextmodel(model_name):
    return ConvNextModel.from_pretrained(model_name)


@_add_classifier
def swinmodel(model_name):
    return SwinModel.from_pretrained(model_name)

@_add_classifier
def mobilenetv1model(model_name):
    return MobileNetV1Model.from_pretrained(model_name)

@_add_classifier
def resnetmodel(model_name):
    return ResNetModel.from_pretrained(model_name)

@_add_classifier
def resnetforimageclassification(model_name):
    return ResNetForImageClassification.from_pretrained(model_name)

@_add_extractor
def convnextfeatureextractor(model_name):
    return ConvNextFeatureExtractor.from_pretrained(model_name)

@_add_extractor
def mobilevitfeatureextractor(model_name):
    return MobileViTFeatureExtractor.from_pretrained(model_name)

@_add_extractor
def mobilenetv1featureextractor(model_name):
    return MobileNetV1FeatureExtractor.from_pretrained(model_name)

@_add_extractor
def vitfeatureextractor(model_name):
    return ViTFeatureExtractor.from_pretrained(model_name)

@_add_extractor
def autoimageprocessor(model_name):
    return AutoImageProcessor.from_pretrained(model_name)

def run(model_name,dataset_name,dataloader,extractor,classifier,output_shape=2048):
    # transformer
    # feature_extractor = get_extractor(extractor,model_name)
    model = get_classifier(classifier,model_name).cuda()
    print('==================')
    print(f'model_name: {model_name}, dataset_name: {dataset_name}')
    m_name = model_name.replace('/','_')
    # print(model)

    # Feature extraction.
    features = ''
    labels = torch.zeros(1).to('cuda')
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            # print(x)
            # inputs = feature_extractor(images=x, return_tensors="pt")
            output = model(x.cuda(),output_hidden_states=True,return_dict=True)
            last_hidden_layer = output.pooler_output
            shape = list(last_hidden_layer.shape)
            print(f'last_hidden_layer.shape: {shape}')
            last_hidden_layer = torch.reshape(last_hidden_layer,(len(y),shape[1]))
            if features == '':
                features = torch.zeros(1,shape[1]).to('cuda')
            features = torch.cat((features,last_hidden_layer),0)
            labels = torch.cat((labels,y.cuda()),0)
    if not os.path.exists(os.path.join('./feature', dataset_name)):
        os.makedirs(os.path.join('./feature', dataset_name))
    features = features.cpu().detach().numpy()
    features = features[1:]
    labels = labels.cpu().detach().numpy()
    labels = labels[1:]
    save_dir = os.path.join('./feature', dataset_name, m_name)
    np.save(os.path.join(save_dir + f'_feature.npy'), features)
    np.save(os.path.join(save_dir + f'_label.npy'), labels)

def main():
    file = 'models.csv'
    ds = ''
    ds_name = ''
    df = pd.read_csv(file)
    for i, row in df.loc[23:].iterrows():
        model_name = row['model']
        dataset_name = row['dataset']
        if 'imagenet' in dataset_name: 
            print(f'skip {model_name}')
            continue
        if (ds_name != dataset_name) or (not ds):
            ds = dataset.__dict__[dataset_name]('../datasets/')[0]
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
            ds_name = dataset_name
        extractor = row['extractor'].lower()
        classifier = row['classifier'].lower()

        run(model_name,dataset_name,dataloader,extractor,classifier)

if __name__ == "__main__":
    main()