import argparse, os
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm

from torchvision import datasets
from LogME import LogME
import pprint

import time
import json

import torchvision
import torchvision.transforms.functional as TF
from transformers import AutoModel, AutoModelForImageClassification

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
sys.path.append('../')
import random
try:
    from util import dataset
except:
    from ..util import dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# cuda cpu
# CUDA_LAUNCH_BLOCKING=1
CHECK=False

# models_hub = ['mobilenet_v2', 'mnasnet1_0', 'densenet121', 'densenet169', 'densenet201',
#                'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet', 'inception_v3']
models_hub = ['mobilenet_v2']

dataset_map = {
        'oxford_flowers102': 'flowers',
        'svhn_cropped': 'svhn',
        'dsprites':['dsprites_label_orientation','dsprites_label_x_position'],
        'smallnorb':['smallnorb_label_azimuth','smallnorb_label_elevation'],
        'oxford_iiit_pet': 'pets',
        'patch_camelyon': 'pcam',
        'clevr':["clevr_count_all", "clevr_count_left", "clevr_count_far", "clevr_count_near", 
            "clevr_closest_object_distance", "clevr_closest_object_x_location", 
            "clevr_count_vehicles", "clevr_closest_vehicle_distance"],
        # 'kitti': ['label_orientation']
    }

def get_configs():
    parser = argparse.ArgumentParser(
        description='Ranking pre-trained models')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU num for training')
    parser.add_argument('--batch_size', default=48, type=int)
    
    # dataset
    parser.add_argument('--dataset', default="aircraft",
                        type=str, help='Name of dataset')
    parser.add_argument('--data_path', default="/data/FGVCAircraft/train",
                        type=str, help='Path of dataset')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='Num of workers used in dataloading')
    # model
    configs = parser.parse_args()

    return configs


def forward_pass(score_loader, model, fc_layer):
    """
    a forward pass on target dataset
    :params score_loader: the dataloader for scoring transferability
    :params model: the model for scoring transferability
    :params fc_layer: the fc layer of the model, for registering hooks
    returns
        features: extracted features of model
        outputs: outputs of model
        targets: ground-truth labels of dataset
    """
    features = []
    outputs = []
    targets = []
    
    def hook_fn_forward(module, input, output):
        features.append(input[0].detach().cpu())
        outputs.append(output.detach().cpu())
    
    forward_hook = fc_layer.register_forward_hook(hook_fn_forward)
    
    model.eval()
    with torch.no_grad():
        for _, (data, target) in enumerate(score_loader):
            targets.append(target)
            data = data.to(device)
            _ = model(data)
    
    forward_hook.remove()
    features = torch.cat([x for x in features])
    outputs = torch.cat([x for x in outputs])
    targets = torch.cat([x for x in targets])
    
    return features, outputs, targets

def check_data(score_dataset,score_loader):
    print(type(score_dataset),len(score_dataset))
    sample_idx = torch.randint(len(score_loader), size=(1,)).item()
    print(f'sample_idx: {sample_idx},type: {type(sample_idx)}')
    img, label = score_dataset[sample_idx]
    print(label)
    print(img)


def main(method):
    configs = get_configs()
    # torch.cuda.set_device(configs.gpu)

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])
    root = '../'
    file = '../doc/model_config_dataset.csv'
    df_config = pd.read_csv(file)

    file = '../doc/ftrecords_img.csv'
    df = pd.read_csv(file,index_col=0)
    df['dataset'] = df['train_dataset_name']
    datasets = list(df['dataset'].value_counts().index)
    datasets += ['poolrf2001/FaceMask','FastJobs/Visual_Emotional_Analysis',
                    'food101','cifar10','cifar100','caltech101',
                    'stanfordcars','eurosat','clevr_count_all','clevr_closest_object_distance',
                    'dmlab', 'kitti_closest_vehicle_distance','oxford_flowers102','oxford_iiit_pet',
                    'pcam','sun397','smallnorb_label_azimuth','smallnorb_label_elevation',
                    'svhn','resisc45','diabetic_retinopathy_detection',
                    'cats_vs_dogs','beans','keremberke/pokemon-classification',
                    'Matthijs/snacks','chest_xray_classification']
    
    time_count = 0
    count = 0
    start = time.time()
    for dataset_name in datasets[17:]:
        print(f'==== dataset: {dataset_name}')
        df_sub = df[df['dataset']==dataset_name]
        df_sub.index = range(len(df_sub))
        df_sub = df_sub[df_sub['model_identifier'].isin(df_config['model'].values)]
        df_sub = df_sub.groupby('model_identifier').max().reset_index()
        # print(df_sub.head())
        # dataset_name = row['dataset']
        dataset_name = dataset_name.replace('/','_').replace('-','_')   
    
        if dataset_name in ['FastJobs_Visual_Emotional_Analysis','poolrf2001_facemask','davanstrien_iiif_manuscripts_label_ge_50']: 
            continue
        if dataset_name == 'imagenet_21k': dataset_name = 'imagenet'

        if dataset_name in dataset_map.keys() and isinstance(dataset_map[dataset_name],list):
            config = df_sub['configs'].values[0]
            print(config)
            config = config.replace("'",'"')
            if 'dsprites' in dataset_name:
                dataset_name = dataset_name + '_'+ json.loads(config)['label_name']
            else:
                try:
                    dataset_name = json.loads(config)['preprocess']
                except:
                    dataset_name = dataset_name + '_'+ json.loads(config)['label_name']


        dataset_name = dataset_map[dataset_name] if dataset_name in dataset_map.keys() else dataset_name
        print(dataset_name)
        configs.dataset = dataset_name

        dir_path = f'./{method}_scores'
        print(f'=== dir_path: {dir_path} ===')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if os.path.exists(f'{dir_path}/{dataset_name}.csv'):
            df_rank = pd.read_csv(f'{dir_path}/{dataset_name}.csv')
            df_rank.index = df_rank['model'].copy()
            score_dict = df_rank['score'].to_dict()
        else:
            df_rank = pd.DataFrame(columns = ['score','model'])
            score_dict = {}
        
        # Load the dataset
        dataloader = dataset.get_dataloader(root,dataset_name,data_sets=[],input_shape=384,batch_size=16,splits=[''],return_classes=False)[0]
            
        # score_dict = {}
        # if dataset_name == 'eurosat': continue        
        for i, row in df_sub.iloc[:].iterrows():           
            model_name = row['model_identifier']
            configs.model = model_name
            print(f'=========== {i}/{len(df_sub)}; model_name: {model_name}; dataset_name: {dataset_name} ===============')
            if model_name in score_dict.keys(): 
                print(f'-- model {model_name} exists')
                continue
            try:
                input_shape = int(df_config[df_config['model']==model_name]['input_shape'].values[0])
                print(f'input_shape: {input_shape}')
            except Exception as e:
                print(e)
                print('---- There is no such model in model_config_dataset.csv')
                continue
           
            try:
                score_dict[configs.model] = score_model(method,model_name,configs, dataloader,input_shape)
            except Exception as e:
                print(e)
                continue 
            # results = sorted(score_dict.items(), key=lambda i: i[1], reverse=True)
            # torch.save(score_dict, f'logme_{dataset}/results.pth')
            count += 1
            
            time_count = round((time.time()-start)/count,3)
            score_dict['time'] = time_count
            df_results = pd.DataFrame.from_dict(score_dict,orient='index')
            df_results.columns = ['score']
            df_results['model'] = list(df_results.index)
            df_results.index = range(len(df_results))
            df_results.to_csv(os.path.join(dir_path,dataset_name+'.csv'))

            print(f'Models ranking on {configs.dataset}: ')
            pprint.pprint(score_dict)


def get_features(method,model,dataloader,input_shape,GET_FEATURE=True):
    ## Feature initiation.
    # features_tensor = torch.zeros(1,FEATURE_DIM).to(device)
    # print(f'features.shape: {features_tensor.shape}')
    transform = transforms.Compose([
        transforms.Resize((input_shape,input_shape))
    ])
    labels_tensor = torch.zeros(1,).to(device)
    print_flag = True
    torch.cuda.empty_cache()
    with torch.no_grad():
        for x,y in tqdm(dataloader):
            if GET_FEATURE:
                if x.shape[1] != 384:
                    x = transform(x)
                output = model(x.to(device))
                if method == 'LogME':
                    if output.pooler_output == None:
                        output = output.last_hidden_state.mean(dim=1)
                        # https://discuss.huggingface.co/t/last-hidden-state-vs-pooler-output-in-clipvisionmodel/26281
                    else:
                        output = output.pooler_output
                elif method == 'LEEP':
                    output = output.logits
                    # print(f'output.shape: {output.shape},y.shape:{y.shape}')
                if print_flag:
                    # print(batch)
                    print('-----------')
                    print(f'x.shape: {x.shape},y.shape:{y.shape}')
                    print(f'output.shape: {output.shape}')
                    print('-----------')
                    print_flag = False
                    FEATURE_DIM = output.shape[1]
                    features_tensor = torch.zeros(1,FEATURE_DIM).to(device)
                # feature = torch.reshape(output.pooler_output,(len(y),FEATURE_DIM))
                feature = torch.flatten(output, start_dim=1)
                features_tensor = torch.cat((features_tensor,feature),0)
            labels_tensor = torch.cat((labels_tensor,y.to(device)),0)
    features_tensor = features_tensor.cpu().detach().numpy()[1:]
    labels_tensor = labels_tensor.cpu().detach().numpy()[1:]
    return features_tensor, labels_tensor

def score_model(method,model_name,configs, dataloader,input_shape):
    # print(f'Calc Transferabilities of {configs.model} on {configs.dataset}')
    print(f'Conducting transferability calculation with {method} ...')
    if method == 'LogME':
        # Load model
        try:
            # model = AutoModel.from_pretrained(model_name).cuda()
            model = AutoModel.from_pretrained(model_name).to(device)
            # model = _CLASSIFIER.__dict__[imgclassification_name](model_name)
            print('== Finish loading model')
        except Exception as e:
            print('================')
            print(e)
            print(f'== Fail - model_name: {model_name}')

        print('Conducting features extraction...')
        # features, outputs, targets = forward_pass(score_loader, model, fc_layer)
        # predictions = F.softmax(outputs)
        
        features, targets = get_features(method,model,dataloader,input_shape)
        logme = LogME(regression=False)
        # score = logme.fit(features.numpy(), targets.numpy())
        score = logme.fit(features,targets)
        print(f'LogME of {configs.model}: {score}\n')

    elif method == 'LEEP':
        # Load model
        try:
            # model = AutoModel.from_pretrained(model_name).cuda()
            model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
            # model = _CLASSIFIER.__dict__[imgclassification_name](model_name)
            print('== Finish loading model')
        except Exception as e:
            print('================')
            print(e)
            print(f'== Fail - model_name: {model_name}')
        print('Conducting features extraction...')
        features, targets = get_features(method,model,dataloader,input_shape)
        from LEEP import LEEP
        print('Conducting transferability extraction...')
        # try:
        score = LEEP(features,targets)
        # except Exception as e:
            # print(e)
        print(f'LEEP of {configs.model}: {score}\n')
    # save calculated bayesian weight
    # torch.save(logme.ms, f'logme_{configs.dataset}/weight_{configs.model}.pth')
    
    return score
    

if __name__ == '__main__':
    main(method='LogME') # ' LEEP
