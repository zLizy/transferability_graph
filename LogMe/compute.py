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

# import torchvision
import torchvision.transforms.functional as TF
from transformers import AutoModel, AutoModelForImageClassification

import os
import gc
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import traceback

sys.path.append('../')
import random

try:
    # from util import dataset_lists
    from util import dataset
except:
    from ..util import dataset_lists

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cuda cpu
# CUDA_LAUNCH_BLOCKING=1
CHECK = False

# models_hub = ['mobilenet_v2', 'mnasnet1_0', 'densenet121', 'densenet169', 'densenet201',
#                'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet', 'inception_v3']
models_hub = ['mobilenet_v2']

dataset_map = {
    'oxford_flowers102': 'flowers',
    'svhn_cropped': 'svhn',
    'dsprites': ['dsprites_label_orientation', 'dsprites_label_x_position'],
    'smallnorb': ['smallnorb_label_azimuth', 'smallnorb_label_elevation'],
    'oxford_iiit_pet': 'pets',
    'patch_camelyon': 'pcam',
    'clevr': ["clevr_count_all", "clevr_count_left", "clevr_count_far", "clevr_count_near",
              "clevr_closest_object_distance", "clevr_closest_object_x_location",
              "clevr_count_vehicles", "clevr_closest_vehicle_distance"],
    # 'kitti': ['label_orientation']
}


def get_configs():
    parser = argparse.ArgumentParser(description='Ranking pre-trained models')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU num for training')
    parser.add_argument('--batch_size', default=48, type=int)

    # dataset
    parser.add_argument('--dataset_name', default="cifar100",
                        type=str, help='Name of dataset')
    parser.add_argument('--data_path', default="/data/FGVCAircraft/train",
                        type=str, help='Path of dataset')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='Num of workers used in dataloading')

    parser.add_argument('--method', default='LogMe', type=str)
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


def check_data(score_dataset, score_loader):
    print(type(score_dataset), len(score_dataset))
    sample_idx = torch.randint(len(score_loader), size=(1,)).item()
    print(f'sample_idx: {sample_idx},type: {type(sample_idx)}')
    img, label = score_dataset[sample_idx]
    print(label)
    print(img)


def main():
    configs = get_configs()
    method = configs.method
    print()
    print('===== Succeed loading arguments')
    # torch.cuda.set_device(configs.gpu)

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])
    root = '/tudelft.net/staff-umbrella/zlitransfer'
    file = f'{root}/doc/model_config_dataset.csv'
    df_config = pd.read_csv(file)

    all_model = df_config['model'].unique()
    dataset_name = configs.dataset_name

    dataset_name = dataset_map[dataset_name] if dataset_name in dataset_map.keys() else dataset_name
    print(dataset_name)
    configs.dataset = dataset_name

    dir_path = f'./{method}_scores'
    print(f'=== dir_path: {dir_path} ===')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if os.path.exists(f'{dir_path}/{dataset_name}.csv'):
        df_results = pd.read_csv(f'{dir_path}/{dataset_name}.csv', index_col=0)
    else:
        df_results = pd.DataFrame(columns=['model', 'score', 'runtime'])

    # Load the dataset
    # dataloader = dataset.get_dataloader(root,dataset_name,data_sets=[],input_shape=384,batch_size=16,splits=[''],return_classes=False)[0]
    print()
    print(f'root: {root}')
    root = '/tudelft.net/staff-umbrella/zlitransfer'
    data_sets, classes = dataset.get_dataset(root, dataset_name, input_shape=384, splits=[''], return_classes=False)
    loader_list = []
    os.system('/usr/bin/nvidia-smi')
    print(f'len(data_sets): {len(data_sets)}')
    for data in data_sets:
        dataloader = DataLoader(
            data,
            batch_size=32,  # may need to reduce this depending on your GPU
            num_workers=1,  # may need to reduce this depending on your num of CPUs and RAM
            shuffle=False,
            drop_last=False,
            pin_memory=False  # True
        )
        print(f'dataloader size: {len(dataloader)}')
        loader_list.append(dataloader)

    if dataset_name == 'eurosat': return 0

    os.system('/usr/bin/nvidia-smi')
    for i, model_name in enumerate(all_model):
        score_dict = {}
        configs.model = model_name
        print(f'=========== {i}/{len(all_model)}; model_name: {model_name}; dataset_name: {dataset_name} ===============')
        if model_name in list(df_results['model']):
            print(f'-- model {model_name} exists')
            continue
        try:
            input_shape = int(df_config[df_config['model'] == model_name]['input_shape'].values[0])
            print(f'input_shape: {input_shape}')
        except Exception as e:
            print(e)
            print('---- There is no such model in model_config_dataset.csv')
            continue

        try:
            gc.collect()
            torch.cuda.empty_cache()
            os.system('/usr/bin/nvidia-smi')
            time_start_model = time.time()
            score_dict['model'] = model_name

            score = score_model(method, model_name, configs, dataloader, input_shape)

            score_dict['score'] = score
            runtime_model = round((time.time() - time_start_model), 3)
            score_dict['runtime'] = runtime_model

            df_results = pd.concat([df_results, pd.DataFrame(score_dict, index=[0])], ignore_index=True)
            df_results.to_csv(os.path.join(dir_path, dataset_name + '.csv'))

            print(f'Models ranking on {configs.dataset}: ')
            pprint.pprint(df_results)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            continue


def get_features(method, model, dataloader, input_shape, GET_FEATURE=True):
    ## Feature initiation.
    # features_tensor = torch.zeros(1,FEATURE_DIM).to(device)
    # print(f'features.shape: {features_tensor.shape}')
    transform = transforms.Compose([
        transforms.Resize((input_shape, input_shape))
    ])
    labels_tensor = torch.zeros(1, ).to(device)
    predictions_tensor = torch.zeros(1, ).to(device)
    print_flag = True
    
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            if GET_FEATURE:
                if x.shape[1] != 384:
                    x = transform(x)
                outputs = model(x.to(device))
                if method == 'LogME':
                    if outputs.pooler_output == None:
                        output = outputs.last_hidden_state.mean(dim=1)
                        # https://discuss.huggingface.co/t/last-hidden-state-vs-pooler-output-in-clipvisionmodel/26281
                    else:
                        output = outputs.pooler_output
                elif method == 'LEEP' or method == 'NCE':
                    output = outputs.logits
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    # print(f'output.shape: {output.shape},y.shape:{y.shape}')
                if print_flag:
                    # print(batch)
                    print('-----------')
                    print(f'x.shape: {x.shape},y.shape:{y.shape}')
                    print(f'output.shape: {output.shape}')
                    print(f'Predictions: {predictions}')
                    print('-----------')
                    
                    print_flag = False
                    FEATURE_DIM = output.shape[1]
                    features_tensor = torch.zeros(1, FEATURE_DIM).to(device)
                # feature = torch.reshape(output.pooler_output,(len(y),FEATURE_DIM))
                feature = torch.flatten(output, start_dim=1)
                features_tensor = torch.cat((features_tensor, feature), 0)
            labels_tensor = torch.cat((labels_tensor, y.to(device)), 0)
            predictions_tensor = torch.cat((predictions_tensor, predictions), 0)
            
    features_tensor = features_tensor.cpu().detach().numpy()[1:]
    labels_tensor = labels_tensor.cpu().detach().numpy()[1:]
    predictions_tensor = predictions_tensor.cpu().detach().numpy()[1:]
    return features_tensor, labels_tensor, predictions_tensor


def score_model(method, model_name, configs, dataloader, input_shape):
    # print(f'Calc Transferabilities of {configs.model} on {configs.dataset}')
    print(f'Conducting transferability calculation with {method} ...')
    # Load model
    # model = AutoModel.from_pretrained(model_name).cuda()
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
    # model = _CLASSIFIER.__dict__[imgclassification_name](model_name)
    print('== Finish loading model')

    if method == 'LogME':
        print('Conducting features extraction...')
        # features, outputs, targets = forward_pass(score_loader, model, fc_layer)
        # predictions = F.softmax(outputs)

        features, targets, outputs = get_features(method, model, dataloader, input_shape)
        logme = LogME(regression=False)
        # score = logme.fit(features.numpy(), targets.numpy())
        score = logme.fit(features, targets)
        print(f'LogME of {configs.model}: {score}\n')
    elif method == 'LEEP':
        print('Conducting features extraction...')
        features, targets, outputs = get_features(method, model, dataloader, input_shape)
        from LEEP import LEEP
        print('Conducting transferability extraction...')
        # try:
        score = LEEP(features, targets)
        # except Exception as e:
        # print(e)
        print(f'LEEP of {configs.model}: {score}\n')
    elif method == 'NCE':
        features, targets, predictions = get_features(method, model, dataloader, input_shape)
        from NCE import NCE
        score = NCE(source_label=predictions, target_label=targets)
        print(f'NCE of {configs.model}: {score}\n')
    else:
        raise Exception(f'Unknwown method {method}')
    # save calculated bayesian weight
    # torch.save(logme.ms, f'logme_{configs.dataset}/weight_{configs.model}.pth')

    return score


if __name__ == '__main__':
    main()
