import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import models
from transformers import AutoModel
from transformers import AutoModelForImageClassification

import os
import sys
sys.path.append('../')
import random
from util import dataset
from model_embed.attribution_map.methods._model import *

from captum.attr import IntegratedGradients
from model_embed.attribution_map.methods.saliency import Saliency
from model_embed.attribution_map.methods.lrp import LRP
from captum.attr import DeepLift
from model_embed.attribution_map.methods.input_x_gradient import InputXGradient
from captum.attr import NoiseTunnel
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# cuda
CUDA_LAUNCH_BLOCKING=1

_METHODS = {}

def _add_method(method_fn):
    _METHODS[method_fn.__name__] = method_fn
    return method_fn


@_add_method
def saliency(model,input,label):
    saliency = Saliency(model)
    # print(saliency)
    # input_new = transforms.Resize(input_shape)(input)
    grads = saliency.attribute(input.requires_grad_(True).to(device), target=torch.tensor(label).to(device))#target=.to(device)), label.clone().detach().requires_grad_(True)
    # print(f'grads.shape: {grads.shape}')
    # grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (0, 2, 3, 1))
    return grads
    # print(grads.shape)

@_add_method
def input_x_gradient(model,input,label):
    # integrated_gradients = IntegratedGradients(model)
    # attributions_ig = integrated_gradients.attribute(input, target=label, n_steps=200)
    input_x_gradient = InputXGradient(model)
    # Computes inputXgradient for class 4.
    attribution = input_x_gradient.attribute(input.to(device), target=torch.tensor(label).to(device))
    return attribution

@_add_method
def elrp(model,input,label):
    lrp = LRP(model)
    # Attribution size matches input size: 3x3x32x32
    attribution = lrp.attribute(input.to(device), target=torch.tensor(label).to(device))
    return attribution

def embed(root,model_name,dataset_name,method,input_shape=224,batch_size=64):
    try:
        from util import dataset
    except:
        from ..util import dataset
    # model = ConvNextModel.from_pretrained("facebook/convnext-base-224-22k").to('cuda')
    # dataloader = dataset.get_dataset(root,dataset_name,input_shape=input_shape,batch_size=batch_size)
    dataloader = dataset.get_dataloader(root,dataset_name,data_sets=[],input_shape=input_shape,splits=[''])[0]
    features_tensor = get_features(root,model_name,dataset_name,dataloader,method,input_shape)
    return features_tensor

def get_features(root,model_name,dataset_name,dataloader,method,input_shape=224):
        # Load model
        # imgclassification_name = row['architecture'].replace("'","").replace('[','').replace(']','').lower()
        try:
            # model = AutoModel.from_pretrained(model_name).cuda()
            model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
            # model = _CLASSIFIER.__dict__[imgclassification_name](model_name)
            print('== Finish loading model')
        except Exception as e:
            print('================')
            print(e)
            print(f'== Fail - model_name: {model_name}')
        # input_shape = model.config.image_size
        IMAGE_SHAPE = (3,input_shape,input_shape)
        ## initiate embeddings
        features = np.zeros((1,)+IMAGE_SHAPE)
        print(features.shape)

        with torch.no_grad():
            for x,y in tqdm(dataloader):
                    # torch.cuda.empty_cache()
                    # try:
                    # print(x.shape,y.shape)
                    attributions =  _METHODS[method](model,x,y) 
                    attributions = attributions.cpu().detach().numpy()
                    # except Exception as e:
                    #     print(e)
                    #     attributions = np.zeros((1,)+IMAGE_SHAPE)
                    features = np.concatenate((features,attributions),0)
        features = features[1:]
        print(f'features.shape: {features.shape}')

        ######### Save features ###########
        model_name = model_name.replace('/','_')
        if not os.path.exists(os.path.join(root,'model_embed/attribution_map/feature', f'{dataset_name}')):
            os.makedirs(os.path.join(root,'model_embed/attribution_map/feature', f'{dataset_name}'))
        attributes = np.mean(features,axis=0)
        print(f'attributes.shape: {attributes.shape}')
        if (attributes != np.zeros((1,)+IMAGE_SHAPE)).all():
            np.save(os.path.join(root,'model_embed/attribution_map/feature', f'{dataset_name}',f'{model_name}_{method}.npy'), attributes)
        return attributes

