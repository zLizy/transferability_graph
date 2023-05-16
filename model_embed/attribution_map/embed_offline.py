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
sys.path.append('../../')
import random
from util import dataset
from methods._model import *

from captum.attr import IntegratedGradients
from methods.saliency import Saliency
from methods.lrp import LRP
from captum.attr import DeepLift
from methods.input_x_gradient import InputXGradient
from captum.attr import NoiseTunnel
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model(model_idx):
    file = '../../doc/model_config_dataset.csv'
    ds = ''
    ds_name = ''
    df = pd.read_csv(file)
    # About 291 models
    model_list = np.unique(df['model'])
    # for i, row in df.loc[10:15].iterrows():
    
    print('==============')
    model_name = model_list[model_idx]
    print(f'model_name: {model_name}')
    try:
        model = AutoModel.from_pretrained(model_name).cuda()
    except Exception as e:
        print('================')
        print(e)
        print(f'Fail - model_name: {model_name}')
        return
    
    return model

_METHODS = {}

def _add_method(method_fn):
    _METHODS[method_fn.__name__] = method_fn
    return method_fn


@_add_method
def saliency(model,input,label,IMAGE_SHAPE):
    saliency = Saliency(model)
    # print(saliency)
    # input_new = transforms.Resize(input_shape)(input)
    grads = saliency.attribute(input.to(torch.float).to(device), target=label.to(device))
    # print(f'grads.shape: {grads.shape}')
    # grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (0, 2, 3, 1))
    return grads
    # print(grads.shape)

@_add_method
def input_x_gradient(model,input,label,IMAGE_SHAPE):
    # integrated_gradients = IntegratedGradients(model)
    # attribution = integrated_gradients.attribute(input.to(device), target=label.clone().detach().to(device), n_steps=200)
    input_x_gradient = InputXGradient(model)
    # # Computes inputXgradient for class 4.
    try:
        attribution = input_x_gradient.attribute(input.to(torch.float).to(device), target=label.to(device))
    except TypeError as e:
        print(e)
        attribution = np.zeros((1,)+IMAGE_SHAPE)
    return attribution

@_add_method
def elrp(model,input,label):
    lrp = LRP(model)
    # Attribution size matches input size: 3x3x32x32
    attribution = lrp.attribute(input.to(device), target=torch.tensor(label).to(device))
    return attribution

def main():
    INPUT_SHAPE = 224
    # imlist_size = 1
    explain_methods = ['input_x_gradient']#'elrp','saliency',
    
    # columns = [model,input_shape,output_shape,architectures,task,dataset,#labels,labels,task_specific_params,problem_type,finetuning_task]
    ### Get model and dataset info
    file = '../../doc/model_config_dataset.csv'
    df = pd.read_csv(file)
    df['dataset'] = df['labels']
    for i, row in df.iloc[:].iterrows():
        dataset_name = row['dataset']
        dataset_name = dataset_name.replace('/','_').replace('-','_')   
        
        # if dataset_name == 'eurosat': continue        
        model_name = row['model']
        print(f'=========== model_name: {model_name}; dataset_name: {dataset_name} ===============')
        if dataset_name in ['FastJobs_Visual_Emotional_Analysis','poolrf2001_facemask','davanstrien_iiif_manuscripts_label_ge_50']: 
            continue
        if dataset_name == 'imagenet_21k': dataset_name = 'imagenet'
        INPUT_SHAPE = int(row['input_shape'])
        print(f'input_shape: {INPUT_SHAPE}')
        path = os.path.join(f'./feature',dataset_name,model_name.replace('/','_')+f'_{explain_methods[0]}.npy')
        if os.path.exists(path): 
            print(f'== exist path: {path}')
            continue
        # Load dataset
        # torch.cuda.empty_cache()
        if '[' in dataset_name:
            classes = row['labels'].strip('][').replace("'","").split(', ')
            print(f'classes: {classes}')
            try:
                ds, _, ds_type = dataset.__dict__['hfpics']('../../datasets/',classes,input_shape=INPUT_SHAPE)
            except FileNotFoundError as e:
                print(e)
                continue
            dataset_name = classes
        else:
            try:
                ds, _, ds_type = dataset.__dict__[dataset_name.lower()]('../../datasets/',input_shape=INPUT_SHAPE)
            except Exception as e:
                print(e)
                continue
        print(f'== finish loading dataset, len(ds): {len(ds)}')
        
        # Load model
        # imgclassification_name = row['architecture'].replace("'","").replace('[','').replace(']','').lower()
        try:
            # model = AutoModel.from_pretrained(model_name).cuda()
            model = AutoModelForImageClassification.from_pretrained(model_name,problem_type="multi_class_classification").to(device)
            for param in model.base_model.parameters():
                param.requires_grad = True
            model.train()
            # model = _CLASSIFIER.__dict__[imgclassification_name](model_name)
            print('== Finish loading model')
        except Exception as e:
            print('================')
            print(e)
            print(f'== Fail - model_name: {model_name}')
            continue
        # get model
        # idx = 0
        # model = get_model(idx)
        # input_shape = model.config.image_size
        IMAGE_SHAPE = (3,INPUT_SHAPE,INPUT_SHAPE)
        ## initiate embeddings
        imlist_size = len(ds)
        # elrp = torch.zeros([1] + list(IMAGE_SHAPE), float).to(device)
        # elrp = torch.zeros((1,)+IMAGE_SHAPE).to(device)
        # salien = torch.zeros((1,)+IMAGE_SHAPE).to(device)
        # gradXinput = torch.zeros((1,)+IMAGE_SHAPE).to(device)
        features = np.zeros((1,)+IMAGE_SHAPE)
        print(features.shape)

        # get attribution map attribute
        LEN = min(5000,imlist_size)
        idx = random.sample(range(imlist_size), k=LEN)
        ds = torch.utils.data.Subset(ds, idx)
        dataloader = DataLoader(
                    ds,
                    batch_size=8, # may need to reduce this depending on your GPU 
                    num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
                    shuffle=False,
                    drop_last=False,
                    # pin_memory=True
                )
        # torch.cuda.empty_cache()
        with torch.no_grad():
            for x,y in tqdm(dataloader):
                attributions = {explain_methods[0]:0}
                for explain_method in explain_methods:
                    try:
                        attribution =  _METHODS[explain_method](model,x,y,IMAGE_SHAPE)
                        if isinstance(attribution,np.ndarray):
                            attributions[explain_method] = attribution
                        else:
                            attributions[explain_method] = attribution.cpu().detach().numpy()
                            continue
                    except RuntimeError as e:
                        print(e)
                        continue
                        # attributions[explain_method] = torch.zeros((1,)+IMAGE_SHAPE).to(device)
                        # attributions[explain_method] = np.zeros((1,)+IMAGE_SHAPE)
                # elrp = torch.cat((elrp,attributions['elrp']),0)
                # salien = torch.cat((elrp,attributions['saliency']),0)
                # gradXinput = torch.cat((gradXinput,attributions['input_x_gradient']),0)
                if (attributions[explain_method] == np.zeros((1,)+IMAGE_SHAPE)).all(): continue
                features = np.concatenate((features,attributions['input_x_gradient']),0)
                # saliency[im_i] = attributions['saliency']
                # if ((im_i+1) % 500) == 0:
                #     print('{} images done.'.format(im_i))
        # elrp = elrp.cpu().detach().numpy()[1:]
        # salien = salien.cpu().detach().numpy()[1:]
        if (features == np.zeros((1,)+IMAGE_SHAPE)).all(): continue
        features = features[1:]#.cpu().detach().numpy()[1:]
        print(f'gradXinput.shape: {features.shape}')

        ######### Save features ###########
        model_name = model_name.replace('/','_')
        if not os.path.exists(os.path.join('./feature', f'{dataset_name}')):
            os.makedirs(os.path.join('./feature', f'{dataset_name}'))
        
        # features = {'elrp':elrp, 'saliency':salien, 'input_x_gradient':gradXinput}
        features = {'input_x_gradient':features}
        for method in explain_methods:
            attributes = np.mean(features[method],axis=0)
            print(f'attributes.shape: {attributes.shape}')
            if (attributes != np.zeros((1,)+IMAGE_SHAPE)).all():
                np.save(os.path.join('./feature', f'{dataset_name}',f'{model_name}_{method}.npy'), attributes)
        ############## Reset graph and paths ##############
        # print('Task {} Done!'.format(task))
    print('All Done.')
    return

    # _ = viz.visualize_image_attr_multiple(grads,
    #                                     np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
    #                                     ["original_image", "heat_map"],
    #                                     ["all", "positive"],
    #                                     cmap=default_cmap,
    #                                     show_colorbar=True)

if __name__ == "__main__":
    main()