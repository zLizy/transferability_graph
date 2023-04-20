import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import models
from transformers import AutoModel
from transformers import AutoImageClassifier

import sys
sys.path.append('../../')
from util import dataset

from captum.attr import IntegratedGradients
from methods.saliency import Saliency
from captum.attr import DeepLift
from captum.attr import InputXGradient
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
def saliency(model,input):
    saliency = Saliency(model)
    # print(saliency)
    # input_new = transforms.Resize(input_shape)(input)
    # print(f'intput_new: {input_new.shape}')
    grads = saliency.attribute(input, target=0)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
    return grads
    # print(grads.shape)

@_add_method
def input_x_gradient(model,input,label):
    # integrated_gradients = IntegratedGradients(model)
    # attributions_ig = integrated_gradients.attribute(input, target=label, n_steps=200)
    input_x_gradient = InputXGradient(model)
    # Computes inputXgradient for class 4.
    attribution = input_x_gradient.attribute(input, target=label)
    return attribution

@_add_method
def lrp(model,input,label):
    from captum.attr import LRP
    lrp = LRP(net)
    # Attribution size matches input size: 3x3x32x32
    attribution = lrp.attribute(input, target=label)
    return attribution

def main():
    INPUT_SHAPE = 224
    # imlist_size = 1
    explain_methods = ['lrp','saliency','input_x_gradient']
    
    # columns = [model,input_shape,output_shape,architectures,task,dataset,#labels,labels,task_specific_params,problem_type,finetuning_task]
    ### Get model and dataset info
    file = '../../doc/model_config_dataset.csv'
    df = pd.read_csv(file)
    for i, row in df.iterrows():
        dataset_name = row['dataset'].values[0]
        dataset_name = dataset_name.replace('/','_').replace('-','_')
        model_name = row['model'].values[0]
        print(f'=========== model_name: {model_name}; dataset_name: {dataset_name} ===============')
        INPUT_SHAPE = int(row['input_shape'].value[0])
        print(f'input_shape: {INPUT_SHAPE}')
        # Load dataset
        ds, _, ds_type = dataset.__dict__[dataset_name]('../../datasets/')
        # input = torch.rand(1,3,INPUT_SHAPE,INPUT_SHAPE).cuda()
        baseline = torch.zeros(1,3,INPUT_SHAPE, INPUT_SHAPE).cuda()
        # Load model
        try:
            model = AutoModel.from_pretrained(model_name).cuda()
        except Exception as e:
            print('================')
            print(e)
            print(f'Fail - model_name: {model_name}')
            continue

        # get model
        # idx = 0
        # model = get_model(idx)
        # input_shape = model.config.image_size
        IMAGE_SHAPE = (3,INPUT_SHAPE,INPUT_SHAPE)
        ## initiate embeddings
        elrp = np.zeros([imlist_size,len(ds.classes)] + list(IMAGE_SHAPE), float)
        salien = np.zeros([imlist_size,len(ds.classes)] + list(IMAGE_SHAPE), float)
        gradXinput = np.zeros([imlist_size,len(ds.classes)] + list(IMAGE_SHAPE), float)
        print(salien.shape)

        # get attribution map attribute
        for im_i in range(imlist_size):
            attributions = {
                    explain_method: _METHODS[explain_method](model,ds[im_i][0].to(device),ds[im_i][1]) for
                    explain_method in explain_methods}
            elrp[im_i] = attributions['lrp']
            saliency[im_i] = attributions['saliency']
            gradXinput[im_i] = attributions['input_x_gradient']
            if ((im_i+1) % 500) == 0:
                print('{} images done.'.format(im_i))

        np.save(os.path.join(explain_result_root, task, 'elrp.npy'), elrp)
        np.save(os.path.join(explain_result_root, task, 'saliency.npy'), saliency)
        np.save(os.path.join(explain_result_root, task, 'gradXinput.npy'), gradXinput)
        ############## Reset graph and paths ##############
        print('Task {} Done!'.format(task))
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