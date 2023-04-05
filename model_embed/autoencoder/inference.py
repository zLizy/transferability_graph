from transformers import AutoModel, SwinModel, ViTMAEModel, ConvNextModel,MobileViTModel, CvtModel, ResNetModel, MobileNetV1Model
from transformers import ViTModel, ViTForImageClassification, ResNetForImageClassification
from transformers import ConvNextFeatureExtractor, AutoFeatureExtractor,AutoImageProcessor, ViTFeatureExtractor
from transformers import AutoConfig
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import requests
# from optimum.intel.openvino import OVModelForImageClassification
from tqdm import tqdm
import random
import os 
import sys
sys.path.append('../../dataset_embed')
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

# @_add_extractor
# def autoimageprocessor(model_name):
#     return AutoImageProcessor.from_pretrained(model_name)

def run(model_name,dataloader,output_shape=2048):
    # transformer
    # feature_extractor = get_extractor(extractor,model_name)
    # model = get_classifier(classifier,model_name).cuda()
    try:
        model = ViTForImageClassification.from_pretrained(model_name).cuda()
    except Exception as e:
        print('================')
        print(e)
        print(f'Fail - model_name: {model_name}')
        return
    print(f'model_name: {model_name}')
    m_name = model_name.replace('/','_')
    # print(model)
    # print(model.config)
    try:
        input_shape = model.config.image_size
    except:
        print('----- no image_size in config -------')
        processor = AutoImageProcessor.from_pretrained(model_name)
        input_shape = list(processor.size.values())[0]
    print(f'input_shape: {input_shape}')
    # Feature extraction.
    features = []
    labels = torch.zeros(1).to('cuda')
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            print(f'!!!!!!! x.shape: {x.shape}')
            # inputs = feature_extractor(images=x, return_tensors="pt")
            if input_shape != 224:
                x = transforms.Resize(input_shape)(x)
            output = model(x.cuda(),output_hidden_states=True,return_dict=True)#output_hidden_states=True,return_dict=True
            try:
                last_hidden_layer = output.pooler_output
            except Exception as e:
                print(e)
                last_hidden_layer = output.last_hidden_state.mean(dim=1)
                # continue
                # print(model.config.keys())
            if last_hidden_layer is None:
                print('================')
                print(f'Fail - model_name: {model_name}')
                continue
            shape = list(last_hidden_layer.shape)
            print(f'last_hidden_layer.shape: {shape}')
            last_hidden_layer = torch.reshape(last_hidden_layer,(len(y),shape[1]))
            features.append(last_hidden_layer)
            # features = torch.cat((features,last_hidden_layer),0)
            # labels = torch.cat((labels,y.cuda()),0)

def check_inference(model_name):
    dataset_name = 'cifar10' #row['dataset']
    if 'imagenet' in dataset_name: 
        print(f'skip {model_name}')
        return
    if True:
        ds = dataset.__dict__[dataset_name]('../datasets/')[0]
        idx = random.sample(range(len(ds)), k=64)
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
        # extractor = row['extractor'].lower()
        # classifier = row['classifier'].lower()

    run(model_name,dataloader)

def check_config(model_name,columns):
    from PIL import Image
    import requests
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    # columns=['model','input_shape','output_shape','architectures','task','dataset','#labels','labels','task_specific_params','problem_type']
    df_tmp = pd.DataFrame(columns=columns)
    df_tmp = pd.DataFrame({'model':model_name},index=[0],columns=columns)
    # print(df_tmp)
    try:
        config = AutoConfig.from_pretrained(model_name)
    except Exception as e:
        print('================')
        print(e)
        print(f'Fail - loading config: {model_name}')
        return df_tmp
    
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
    except:
        processor = AutoFeatureExtractor.from_pretrained(model_name)
    input_shape = list(processor.size.values())[0]
    df_tmp.iloc[0]['input_shape'] = input_shape
    model = AutoModel.from_pretrained(model_name).to('cuda')
    img = processor(image, return_tensors="pt").to('cuda')
    try:
        output = model(**img,output_hidden_states=True,return_dict=True)
    except:
        img = processor(image, return_tensors="pt").pixel_values.to('cuda')
        output = model(img,output_hidden_states=True,return_dict=True)
    try:
        output_shape = list(output.pooler_output.shape)[1]
    except Exception as e:
        print(e)
        output_shape = output.last_hidden_state.mean(dim=1).shape[1]
    df_tmp.iloc[0]['output_shape'] = output_shape
    try:
        # architectures
        architectures = config.architectures
        df_tmp.iloc[0]['architectures'] = architectures
        print(f'architectures: {architectures}')
    except:
        print('no architectures')

    try:
        # task_name
        task = config.task
        df_tmp.iloc[0]['task'] = task
        print(f'task: {task}')
    except:
        print('no task')

    try:
        # task_name
        finetuning_task = config.finetuning_task
        df_tmp.iloc[0]['finetuning_task'] = finetuning_task
        print(f'finetuning_task: {finetuning_task}')
    except:
        print('no finetuning_task')

    try:
        # num_labels
        num_labels = config.num_labels
        df_tmp.iloc[0]['#labels'] = num_labels
        print(f'num_labels: {num_labels}')
    except:
        num_labels = 0
        print('no num_labels')

    if num_labels == 1000:
        df_tmp.iloc[0]['dataset'] = 'imagenet'
    elif num_labels > 20000:
        df_tmp.iloc[0]['dataset'] = 'imagenet-21k'
    else:
        try:
            # id2label
            id2label = list(config.id2label.values())
            if id2label is not None:
                df_tmp.iloc[0]['labels'] = id2label
            print(f'id2label: {id2label}')
        except:
            print('no id2label')
        # try:
        #     # id2label
        #     label2id = list(config.label2id.keys())
        #     if label2id is not None:
        #         df_tmp.iloc[0]['labels'] = label2id
        #     print(f'label2id: {label2id}')
        # except:
        #     print('no label2id')

    try:
        # task_name
        task_specific_params = config.task_specific_params
        if task_specific_params is not None:
            df_tmp.iloc[0]['task_specific_params'] = task_specific_params
        print(f'task_specific_params: {task_specific_params}')
    except:
        print('no task_specific_params')

    try:
        # problem_type
        problem_type = config.problem_type
        if problem_type is not None:
            df_tmp.iloc[0]['problem_type'] = problem_type
        print(f'problem_type: {problem_type}')
    except:
        print('no problem_type')

    return df_tmp


def main():
    CHECK_CONFIG = False
    CHECK_INFERENCE = True
    file = '../../doc/ftrecords_img.csv'
    miss_file = '../../doc/missing_models.csv'
    miss_models = list(pd.read_csv(miss_file)['model'])
    ds = ''
    ds_name = ''
    df = pd.read_csv(file)

    if CHECK_CONFIG:
        columns=['model','input_shape','output_shape','architectures','task','dataset','#labels','labels','task_specific_params','problem_type']
        if not os.path.exists('../../doc/model_config.csv'):
            df_config = pd.DataFrame(columns=columns)
        else:
            df_config = pd.read_csv('../../doc/model_config.csv')
    # About 291 models
    model_list = np.unique(df['model_identifier'])
    # for i, row in df.loc[10:15].iterrows():
    for i, model_name in enumerate(model_list[:1]):
        print('==============')
        print(i,model_name)
        if model_name in miss_models: 
            print('skip model: {model_name}')
            continue
        # model_name = row['model_identifier']
        if CHECK_INFERENCE:
            #### check if the model has pooler layer
            check_inference(model_name)
        if CHECK_CONFIG:
            row = check_config(model_name,columns)
            df_config = pd.concat([df_config, row])
    if CHECK_CONFIG:
        print(df_config.head())
        df_config.to_csv('../doc/model_config.csv',index=False)
        

if __name__ == "__main__":
    main()