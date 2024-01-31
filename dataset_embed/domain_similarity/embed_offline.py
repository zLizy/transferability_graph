from transformers import AutoModel, AutoImageProcessor, ResNetModel,ConvNextModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from torch.utils import data
from torchvision import transforms
import torch
import torchvision
import tensorflow_datasets as tfds
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import random
random.seed(10)

import argparse
import os
import sys
# sys.path.append('../')
# sys.path.append('../..')
import pandas as pd
import dataset
# from util.tfds import VTABIterableDataset


def main(args):

    ## Load datasets
    datasets_list = [   
                    #'toxicity'
                    # 'glue/ax',
                    'glue/cola','glue/sst2',
                    'glue/wnli','glue/rte',
                    'glue/mnli','glue/mrpc','glue/qnli',
                    'glue/qqp','glue/stsb',
                    # 'multi_nli', 'sundanese-twitter', 'trec', 'emotion',
                    # 'tweet_eval/emoji', 'tweet_eval/offensive', 'tweet_eval/emotion',
                    # 'pasinit/scotus','crcb/autotrain-data-isear_bert', 'toxicity',
                    # 'hate_speech_offensive','rotten_tomatoes', 'tweet_eval/hate',
                    # 'tweet_eval/sentiment','ag_news','dbpedia_14','amazon_polarity',
                    # 'tweet_eval/irony','glue/qnli','covid-19_tweets','glue/qqp',
                    # 'imdb','wikipedia','bookcorpus'
                    # 'poolrf2001/FaceMask','FastJobs/Visual_Emotional_Analysis',
                    # 'food101','cifar10','cifar100','caltech101',
                    # 'stanfordcars','eurosat','clevr_count_all','clevr_closest_object_distance',
                    # 'dmlab', 'kitti_closest_vehicle_distance','oxford_flowers102','oxford_iiit_pet',
                    # 'pcam','sun397','smallnorb_label_azimuth','smallnorb_label_elevation',
                    # 'svhn','resisc45','diabetic_retinopathy_detection',
                    # 'cats_vs_dogs','beans','keremberke/pokemon-classification',
                    # 'Matthijs/snacks','chest_xray_classification',
                ]

    for dataset_name in datasets_list[int(args.index):int(args.index)+1]:
        if MODALITY == 'image':
            dataset_name = dataset_name.replace('/','_').replace('-','_')
        print(f'=========== dataset_name: {dataset_name} ===============')
        save_dir = os.path.join('./feature', model_name.replace('/','_'),dataset_name.replace(' ','-').replace('/','_').lower())
        print(f'save_dir: {save_dir}')
        # skip if the feature already exists
        if os.path.exists(save_dir + f'_feature.npy'): continue

        if '[' in dataset_name: #  'hfpics'
            # labels = df[df['dataset']==dataset_name]
            classes = dataset_name.strip('][').replace("'","").split(', ')
            # print(f'classes: {classes}')
            # classes = "['corgi','']"
            ds_type = 'hfpics'
            ds = dataset.__dict__[ds_type]('../../datasets/',classes)
        elif 'glue' in dataset_name:
            classes = dataset_name.split('/')[1]
            ds = dataset.__dict__['glue']('./glue_data',classes,tokenizer)
        else:
            dataset_name = dataset_name.lower()
            ds, _, ds_type = dataset.__dict__[dataset_name]('../../datasets/')
        try:
            length = len(ds)
        except:
            length = ds.get_num_samples('train')
        if length < MAX_NUM_SAMPLES: 
            LEN = length
        else:
            LEN = MAX_NUM_SAMPLES
        print(f'dataset size: {length}')
        print(type(ds))
        
        ## Feature initiation.
        features_tensor = [] #torch.zeros(1,FEATURE_DIM).to(device)
        # print(f'features.shape: {features_tensor.shape}')
        labels_tensor = [] #torch.zeros(1,).to(device)

        ## Load dataset
        print_flag = True
        # randomly sample MAX_NUM_SAMPLES
        # idx = random.sample(range(length), k=LEN)
        # ds = torch.utils.data.Subset(ds, idx)
        
        dataloader = DataLoader(
                        ds,
                        batch_size=int(args.batch_size), # may need to reduce this depending on your GPU 
                        # num_workers=1, # may need to reduce this depending on your num of CPUs and RAM
                        # shuffle=False,
                        # drop_last=False,
                        # pin_memory=True
                    )
        print(f'dataloader size: {len(dataloader)}')
        if MODALITY == 'image':
            with torch.no_grad():
                for x,y in tqdm(dataloader):
                    if GET_FEATURE:
                        output = model(x.to(device))
                        output = output.pooler_output
                        if print_flag:
                            # print(batch)
                            print('-----------')
                            print(f'x.shape: {x.shape},y.shape:{y.shape}')
                            print(f'output.shape: {output.shape}')
                            print('-----------')
                            print_flag = False
                        
                        feature = torch.reshape(output,(len(y),FEATURE_DIM))
                        features_tensor.append(feature)
                    labels_tensor.append(y.to(device))
                labels_tensor = torch.stack(labels_tensor)
        else:
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    if GET_FEATURE:
                        print(f'batch.keys: {batch.keys()}')
                        # for k in batch.keys():
                            # print(k,type(batch[k]),len(batch[k]),torch.stack(batch[k]).shape)
                        batch = {k: torch.stack(v).to(device) for k, v in batch.items()}
                        output = model(**batch)
                        output = output.last_hidden_state.mean(dim=1)
                        if print_flag:
                            # print(batch)
                            print('-----------')
                            print(f'output.shape: {output.shape}')
                            print('-----------')
                            print_flag = False
                        
                        # feature = torch.reshape(output,(len(y),FEATURE_DIM))
                        features_tensor.append(output)
                    # labels_tensor = torch.cat((labels_tensor,y.to(device)),0)

        
        if not os.path.exists(os.path.join('./feature', model_name)):
            os.makedirs(os.path.join('./feature', model_name))
        features_tensor = torch.stack(features_tensor)
        features_tensor = features_tensor.cpu().detach().numpy()
        # labels_tensor = labels_tensor.cpu().detach().numpy()
        # labels_tensor = labels_tensor[1:]
        # print(f'labels_tensor: {labels_tensor.shape}')
        # unique,counts = np.unique(labels_tensor,return_counts=True)
        save_dir = os.path.join('./feature', model_name.replace('/','_'))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if GET_FEATURE:
            # np.save(os.path.join(save_dir + f'_feature_{MAX_NUM_SAMPLES}.npy'), features)
            # sorted_label = sorted(list(set(labels_tensor)))
            # feature_per_class = np.zeros((len(sorted_label), FEATURE_DIM), dtype=np.float32)
            # counter = 0
            # for i in sorted_label:
                # idx = [(l==i) for l in labels_tensor]
                # feature_per_class[counter, :] = np.mean(features_tensor[idx, :], axis=0)
                # counter += 1
            features = np.mean(features_tensor, axis=0)
            np.save(os.path.join(save_dir,dataset_name.replace(' ','_').replace('/','_')+ f'_feature.npy'),features )
        # np.save(os.path.join(save_dir + f'_label_{MAX_NUM_SAMPLES}.npy'), labels)
        # np.save(os.path.join(save_dir + f'_weight_{MAX_NUM_SAMPLES}.npy'), counts)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="A script to finetune a model using a dataset.")
    parser.add_argument("--index", help="The model.")
    # parser.add_argument("--dataset", help="The dataset.")
    parser.add_argument("--batch_size",default=64, help="batch_size.")
    args = parser.parse_args()
    
    MODALITY = 'text'
    MAX_NUM_SAMPLES = 5000
    FEATURE_DIM = 2048 #768 #
    GET_FEATURE = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    if MODALITY == 'text':
        model_name = 'microsoft/deberta-base'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        # df = pd.read_csv('../doc/text_model_config_dataset.csv')
    elif MODALITY == 'visual':
        ## model_dataset mapping
        ## visual datasets
        ### Check output dimension -> it should be 2048
        model_name = 'resnet50'
        # model = torchvision.models.resnet50(weights="DEFAULT")
        model = ResNetModel.from_pretrained("microsoft/resnet-50").to(device)
        # model_name = 'Ahmed9275_Vit-Cifar100'
        # model = AutoModel.from_pretrained('Ahmed9275/Vit-Cifar100').to(device)
        # model_name = 'aricibo_swin-tiny-patch4-window7-224-finetuned-eurosat'
        # model = AutoModel.from_pretrained('aricibo/swin-tiny-patch4-window7-224-finetuned-eurosat').to(device)
        # model_name = 'johnnydevriese_vit_beans'
        # model = AutoModel.from_pretrained('johnnydevriese/vit_beans').to(device)
        # model = ConvNextModel.from_pretrained("facebook/convnext-base-224-22k").to('cuda')
        # df = pd.read_csv('../doc/model_config_dataset.csv')


    main(args)