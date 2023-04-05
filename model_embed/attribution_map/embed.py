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

from captum.attr import IntegratedGradients
from saliency import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz


def get_model(model_idx):

    file = '../../doc/ftrecords_img.csv'
    ds = ''
    ds_name = ''
    df = pd.read_csv(file)
    # About 291 models
    model_list = np.unique(df['model_identifier'])
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

def main():
    INPUT_SHAPE = 224
    imlist_size = 1
    # image input
    input = torch.rand(1,3,INPUT_SHAPE,INPUT_SHAPE).cuda()
    baseline = torch.zeros(1,3,INPUT_SHAPE, INPUT_SHAPE).cuda()
    img_shape = (3,INPUT_SHAPE,INPUT_SHAPE)
    print(f'input.shape: {input.shape}')

    # get model
    idx = 0
    model = get_model(idx)
    input_shape = model.config.image_size
    print(f'input_shape: {input_shape}')

    # initiate embeddings
    elrp = np.zeros([imlist_size] + list(img_shape), float)
    salien = np.zeros([imlist_size] + list(img_shape), float)
    gradXinput = np.zeros([imlist_size] + list(img_shape), float)
    print(salien.shape)

    # get attribution map attribute
    saliency = Saliency(model)
    print(saliency)
    # input_new = transforms.Resize(input_shape)(input)
    # print(f'intput_new: {input_new.shape}')
    grads = saliency.attribute(input, target=0)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
    print(grads.shape)

    # _ = viz.visualize_image_attr_multiple(grads,
    #                                     np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
    #                                     ["original_image", "heat_map"],
    #                                     ["all", "positive"],
    #                                     cmap=default_cmap,
    #                                     show_colorbar=True)

if __name__ == "__main__":
    main()