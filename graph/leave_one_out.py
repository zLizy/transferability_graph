import numpy as np
import pandas as pd
import torch
from torch import Tensor
print(torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")

import scipy.spatial.distance as distance
# distance.cosine(h1, h2)
import tqdm
import os
import json
import pickle
import sys
sys.path.append('../')

from dataset_embed.task2vec_embed import task2vec
sys.modules['task2vec'] = task2vec
from utils._util import *
import argparse

PRINT=True
MODEL_FEATURE_DIM = 1024
dataset_map = {
        # 'oxford_flowers102': 'flowers',
        'svhn_cropped': 'svhn',
        'dsprites':['dsprites_label_orientation','dsprites_label_x_position'],
        'smallnorb':['smallnorb_label_azimuth','smallnorb_label_elevation'],
        # 'oxford_iiit_pet': 'pets',
        'patch_camelyon': 'pcam',
        'clevr':["count_all", "count_left", "count_far", "count_near", 
            "closest_object_distance", "closest_object_x_location", 
            "count_vehicles", "closest_vehicle_distance"],
        # 'kitti': ['label_orientation']
    }

def _print(name,value,level=2):
    if PRINT:
        print()
        if level == 1:
            print('=====================')
        elif level > 1:
            print('---------------')
        print(f'== {name}: {value}')

## Node idx
def get_node_id(df):
    unique_model_id = get_unique_node(df['model'],'model')
    unique_dataset_id = get_unique_node(df['dataset'],'dataset')
    # print(unique_model_id)
    # print(unique_dataset_id)
    return unique_model_id, unique_dataset_id

## Retrieve the embeddings of the dataset
def get_dataset_features(unique_dataset_id,df,approach='domain_similarity'):
    _print('get_dataset_feature','')
    
    data_feat = []
    FEATURE_DIM = 2048
    delete_dataset_row_idx = []
    for i, row in unique_dataset_id.iterrows():
        ds_name = row['dataset']
        dataset_name = ds_name.replace('/','_').replace('-','_')
        _print('dataset_name',dataset_name)
        if dataset_name in ['davanstrien_iiif_manuscripts_label_ge_50',
                            'dsprites',
                            'age_prediction',
                            'FastJobs_Visual_Emotional_Analysis'
                           ]:
            delete_dataset_row_idx.append(i)
            continue
        dataset_name = dataset_map[dataset_name] if dataset_name in dataset_map.keys() else dataset_name
        if isinstance(dataset_name,list):
            # print(dataset_name)
            configs = df[df['dataset']==ds_name]['configs'].values[0].replace("'",'"')
            print(configs)
            if ds_name == 'clevr':
                dataset_name = json.loads(configs)['preprocess']
            else:
                dataset_name = f"{ds_name}_{json.loads(configs)['label_name']}"

        # cannot load imagenet-21k and make them equal
        if dataset_name == 'imagenet_21k': 
            dataset_name = 'imagenet'

        if approach == 'domain_similarity':
            from dataset_embed.domain_similarity.embed import embed
            ds_name = dataset_name.replace(' ','-')
            path = os.path.join(f'../dataset_embed/{approach}/feature','resnet50',f'{ds_name}_feature.npy')
        elif approach == 'task2vec':
            from dataset_embed.task2vec_embed.embed_task import embed
            ds_name = dataset_name.replace(' ','-')
            path = os.path.join(f'../dataset_embed/{approach}_embed/feature',f'{ds_name}_feature.p')
        _print('path',path)

        try:
            if approach == 'domain_similarity':
                features = np.load(path)
            elif approach == 'task2vec':
                with open(path, 'rb') as f:
                    features = pickle.load(f).hessian
                _print('features',features)
                features = features.reshape((1,features.shape[0]))
                _print('features.shape',features.shape)
                FEATURE_DIM = features.shape[1]
        except Exception as e:
            print('----------')
            print(e)
            features = np.zeros((1,FEATURE_DIM))
        if (features == np.zeros((1,FEATURE_DIM))).all():
            print('Try to obtain missing features')
            # features = embed('../',dataset_name)
            try:
                features = embed('../',dataset_name)
            except FileNotFoundError as e:
                print(e)
                print(f'== fail to retrieve features and delete row {i}')
                delete_dataset_row_idx.append(i)
                continue
                print('----------')
                features = np.zeros((1,FEATURE_DIM))
        features = np.mean(features,axis=0)
        # print(f"shape of {dataset_name} is {features.shape}")
        data_feat.append(features)
    data_feat = np.stack(data_feat)
    # size = 25
    # data_feat = data_feat[:,:size]
    # data_feat.astype(np.double)
    print(f'== data_feat.shape:{data_feat.shape}')
    # return torch.from_numpy(data_feat).to(torch.float), delete_dataset_row_idx
    return data_feat,delete_dataset_row_idx

## Retrieve the embeddings of the model
def get_model_features(df,df_config,unique_model_id,complete_model_features=False):
    model_feat = []
    DATA_EMB_METHOD = 'attribution_map'
    ATTRIBUTION_METHOD = 'input_x_gradient' #'input_x_gradient'#'saliency'
    INPUT_SHAPE = 128 #64 # #224
    delete_model_row_idx = []
    for i, row in unique_model_id.iterrows():
        print(f"======== i: {i}, model: {row['model']} ==========")
        model_match_rows = df.loc[df['model']==row['model']]
        # model_match_rows = df_config.loc[df['model']==row['model']]
        if model_match_rows.empty:
            if complete_model_features:
                delete_model_row_idx.append(i)
            else:
                features = np.zeros(INPUT_SHAPE*INPUT_SHAPE)
                model_feat.append(features)
            continue
        if model_match_rows['model'].values[0] == np.nan: 
            delete_model_row_idx.append(i)
            continue
        try:
            dataset_name = model_match_rows['dataset'].values[0].replace('/','_').replace('-','_')
            ds_name = dataset_name
            dataset_name = dataset_map[dataset_name] if dataset_name in dataset_map.keys() else dataset_name
        except:
            print('fail to retrieve model')
            continue
        if isinstance(dataset_name,list):
            # print(dataset_name)
            configs = df[df['dataset']==ds_name]['configs'].values[0].replace("'",'"')
            print(configs)
            if ds_name == 'clevr':
                dataset_name = json.loads(configs)['preprocess']
            else:
                dataset_name = f"{ds_name}_{json.loads(configs)['label_name']}"
        
        # cannot load imagenet-21k and make them equal
        if dataset_name == 'imagenet_21k': 
            dataset_name = 'imagenet'
        
        print(f"== dataset_name: {dataset_name}")
        if dataset_name == 'FastJobs_Visual_Emotional_Analysis': 
            delete_model_row_idx.append(i)
            continue
        IMAGE_SHAPE = int(sorted(model_match_rows['input_shape'].values,reverse=True)[0])
        model_name = row['model']
        # if model_name in ['AkshatSurolia/BEiT-FaceMask-Finetuned','AkshatSurolia/ConvNeXt-FaceMask-Finetuned','AkshatSurolia/DeiT-FaceMask-Finetuned','AkshatSurolia/ViT-FaceMask-Finetuned','Amrrs/indian-foods','Amrrs/south-indian-foods']: 
        #     continue
        path = os.path.join(f'../model_embed/{DATA_EMB_METHOD}/feature',dataset_name,model_name.replace('/','_')+f'_{ATTRIBUTION_METHOD}.npy')
        print(dataset_name,model_name)
        
        try:
            features = np.load(path)
        except Exception as e:
            print('----------')
            print(e)
            if complete_model_features:
                print(f'== Skip this model and delete it')
                delete_model_row_idx.append(i)
                continue 
            else:
                features = np.zeros((INPUT_SHAPE,INPUT_SHAPE))
            # features = np.zeros((INPUT_SHAPE,INPUT_SHAPE))
        print(f'features.shape: {features.shape}')
        if features.shape == (INPUT_SHAPE,INPUT_SHAPE):
            print('Try to obtain missing features')
            from model_embed.attribution_map.embed import embed
            method = ATTRIBUTION_METHOD #'saliency'
            batch_size = 1
            try:
                features = embed('../',model_name,dataset_name,method,input_shape=IMAGE_SHAPE,batch_size=batch_size)
                print('----------')
            except Exception as e:
                print(e)
                print('----------')
                # features = np.zeros((3,INPUT_SHAPE,INPUT_SHAPE))
                delete_model_row_idx.append(i)
                print(f'--- fail - skip row {i}')
                continue
        else:
            if np.isnan(features).all(): 
                features = np.zeros((3,INPUT_SHAPE,INPUT_SHAPE))
        features = np.mean(features,axis=0)
        # print(f'features.shape: {features.shape}')
        if features.shape[1] != INPUT_SHAPE:
            # print(f'== features.shape:{features.shape}')
            features = np.resize(features,(INPUT_SHAPE,INPUT_SHAPE))
        features = features.flatten()
        model_feat.append(features)
    print(f'== model_feat.shape:{len(model_feat)}')
    model_feat = np.stack(model_feat)
    # model_feat.astype(np.double)
    print(f'== model_feat.shape:{model_feat.shape}')
    # return torch.from_numpy(model_feat).to(torch.float), delete_model_row_idx
    return model_feat, delete_model_row_idx

def del_node(unique_id,delete_row_idx):
    ## Drop rows that do not produce dataset features
    print(f'len(unique_id): {len(unique_id)}')
    unique_id = unique_id.drop(labels=delete_row_idx, axis=0)
    return unique_id

def drop_nodes(df,unique_model_id,delete_model_row_idx,unique_dataset_id,delete_dataset_row_idx):
    # reallocate the node id
    unique_dataset_id = get_unique_node(del_node(unique_dataset_id,delete_dataset_row_idx)['dataset'],'dataset')
    unique_model_id = get_unique_node(del_node(unique_model_id,delete_model_row_idx)['model'],'model')

    ## Perform merge to obtain the edges from models and datasets:
    df = df[df['model'].isin(unique_model_id['model'].values)]
    df = df[df['dataset'].isin(unique_dataset_id['dataset'].values)]
    print(f'len(df): {len(df)}')
    return df, unique_model_id, unique_dataset_id

def get_edge_index(df,unique_model_id,unique_dataset_id,accuracy_thres=0.6,ratio=1.0):
    if ratio != 1:
        df = df.sample(frac=ratio, random_state=1)
    print()
    print('==========')
    print(f'len(df): {len(df)}')
    df = df[df['accuracy']>=accuracy_thres]
    print(print(f'len(df) after filtering models with low performance: {len(df)}'))
    mapped_model_id = merge(df['model'],unique_model_id,'model')
    mapped_dataset_id = merge(df['dataset'],unique_dataset_id,'dataset')
    edge_index_model_to_dataset = torch.stack([mapped_model_id, mapped_dataset_id], dim=0)
    print(f'== edge_index_model_to_dataset')
    print(f'mapped_model_id.len: {len(mapped_model_id)}, mapped_dataset_id.len: {len(mapped_dataset_id)}')
    # print(edge_index_model_to_dataset)
    return edge_index_model_to_dataset

## Edge attributes
def get_edge_attr(edge_index_model_to_dataset):
    edge_attr_model_to_dataset = np.zeros((edge_index_model_to_dataset.shape[0],3))
    return edge_attr_model_to_dataset


def preprocess(args):
    file = '../doc/model_config_dataset.csv'
    df_config = pd.read_csv(file)
    df_config['dataset'] = df_config['labels']
    df_config['configs'] = {}
    df_config['accuracy'] = 0.9
    # df = df.loc[~df['dataset'].str.contains('imagenet')]

    file = '../doc/ftrecords_img.csv'
    df_finetuned = pd.read_csv(file)
    df_finetuned['model'] = df_finetuned['model_identifier']
    df_finetuned['dataset'] = df_finetuned['train_dataset_name']
    df_finetuned['accuracy'] = df_finetuned['test_accuracy']
    df_finetuned['input_shape'] = 0

    ######################
    ## Delete the finetune records of the test datset
    ######################
    df_finetuned = df_finetuned[df_finetuned['dataset']!=args.test_dataset]

    df = pd.concat([df_config[['dataset','model','input_shape','configs','accuracy']],df_finetuned[['dataset','model','input_shape','configs','accuracy']]],ignore_index=True)
    df.index = range(len(df))
     ######################
    ## Add an empty row to indicate the dataset
    ######################
    df.loc[len(df)] = {'dataset':args.test_dataset}
    # print()
    # print('=========')
    # print(df.head())
    
    # get node id
    unique_model_id, unique_dataset_id = get_node_id(df)
    # print()
    # print("Mapping of model IDs to consecutive values:")
    # print("==========================================")
    # print(unique_model_id.head())
    # print()
    # print("Mapping of dataset IDs to consecutive values:")
    # print("==========================================")
    # print(unique_dataset_id.head())
    # print()
    
    # get dataset features
    if args.contain_dataset_feature:
        data_feat, delete_dataset_row_idx = get_dataset_features(unique_dataset_id,df,approach=args.dataset_emb_method)
    else:
        data_feat = []
        delete_dataset_row_idx = []
    # assert data_feat.size()[1] == 128  # 20 genres in total.
    
    # get model features
    model_feat, delete_model_row_idx = get_model_features(df,df_config,unique_model_id,args.complete_model_features)#

    # get common nodes
    df, unique_model_id, unique_dataset_id = drop_nodes(df,unique_model_id,delete_model_row_idx,unique_dataset_id,delete_dataset_row_idx)
    
    ##########
    # get specific dataset index
    ##########
    if args.test_dataset != '':
        dataset_idx = unique_dataset_id[unique_dataset_id['dataset']==args.test_dataset]['mappedID'].values[0]
    else:
        dataset_idx = -1

    # get edge index
    edge_index_model_to_dataset = get_edge_index(df,unique_model_id,unique_dataset_id,args.accuracy_thres,args.finetune_ratio)

    # get dataset-dataset edge index
    edge_index_dataset_to_dataset = get_dataset_edge_index(data_feat)

    # get edge attributes
    edge_attr_model_to_dataset = get_edge_attr(edge_index_model_to_dataset)

    return {
                'unique_dataset_id':unique_dataset_id,
                'data_feat':data_feat,
                'unique_model_id':unique_model_id,
                'model_feat':model_feat,
                'edge_index_model_to_dataset':edge_index_model_to_dataset,
                'edge_attr_model_to_dataset':edge_attr_model_to_dataset,
                'edge_index_dataset_to_dataset':edge_index_dataset_to_dataset,
                'dataset_idx': dataset_idx
            }

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def djoin(ldict, req=''):
    return req + ' & '.join([('{0} == "{1}"'.format(k, v)) if isinstance(v,str) else ('{0} == {1}'.format(k, v))  for k,v in ldict.items()])


def main(args):
    print()
    print('========================================')
    root = '../'
    path = os.path.join(root,'doc',f'performance_{args.dataset_emb_method}_{args.gnn_method}.csv')
    args.path = path
    if os.path.exists(path):
        df_perf = pd.read_csv(path,index_col=0)
    else:
        df_perf = pd.DataFrame(
            columns=[
                        'contain_data_similarity',
                        'contain_dataset_feature',
                        'embed_dataset_feature',
                        'contain_model_feature',
                        'embed_model_feature',
                        'complete_model_features',
                        'gnn_method',
                        'accuracy_thres',
                        'finetune_ratio',
                        'hidden_channels',
                        'num_model',
                        'num_dataset',
                        'test_dataset',
                        'train_time',
                        'loss',
                        'val_AUC',
                        'test_AUC'
            ])
    setting_dict = {
        'contain_data_similarity':args.contain_data_similarity,
        'contain_dataset_feature':args.contain_dataset_feature,
        'embed_dataset_feature':args.embed_dataset_feature,
        'contain_model_feature': args.contain_model_feature,
        'embed_model_feature':args.embed_model_feature,
        # 'gnn_method': args.gnn_method,
        'accuracy_thres':args.accuracy_thres,
        'finetune_ratio':args.finetune_ratio,
        # 'complete_model_features':args.complete_model_features,
        'hidden_channels':args.hidden_channels
    }
    print()
    print('======= evaluation_dict ==========')
    evaluation_dict = setting_dict.copy()
    evaluation_dict['test_dataset'] = args.test_dataset
    for k,v in evaluation_dict.items():
        print(f'{k}: {v}')
    # print(evaluation_dict)

    ## Check executed
    query = ' & '.join(list(map(djoin, [evaluation_dict])))
    df_tmp = df_perf.query(query)

    ## skip running because the performance exist
    if not df_tmp.empty: #df_tmp.dropna().empty: 
        return 0
    else:
        print(f'query: {query}')
        # print(df_tmp.dropna())

    # test with MovieLen graph
    test = False
    # if test:
    #     from node_embedding import preprocess
    #     data_dict = preprocess()
    #     unique_dataset_id = data_dict['unique_movie_id']
    #     data_feat = data_dict['movie_feat']
    #     unique_model_id = data_dict['unique_user_id']
    #     model_feat = np.zeros(len(unique_dataset_id))
    #     edge_index_model_to_dataset = data_dict['edge_index_user_to_movie']
    #     edge_attr_model_to_dataset = data_dict['edge_attr_model_to_dataset']
    #     edge_index_dataset_to_dataset = data_dict['edge_index_dataset_to_dataset']
    #     batch_size = 128
    # else:
    data_dict = preprocess(args)
    batch_size = 4

    if args.gnn_method != '""':
        from train_with_GNN import gnn_train
        gnn_train(args,df_perf,data_dict,evaluation_dict,setting_dict,batch_size)
    elif args.gnn_method == '""':
        from utils.basic import get_basic_features
        get_basic_features(args.test_dataset,data_dict,setting_dict)
    elif args.gnn_method == 'node2vec':
        from utils.get_node2vec_features import node2vec_train
        node2vec_train(args,df_perf,data_dict,evaluation_dict,setting_dict,batch_size)

    # model_scripted = torch.jit.script(model) # Export to TorchScript
    # model_scripted.save(os.path.join('./','models','_'.join([ for k,v in evaluation_dict.items()]) +'.pt')) # Save

    ## Load a model
    # model = torch.jit.load('model_scripted.pt')
    # model.eval()
    

if __name__ == '__main__':

    '''
    Configurations
    '''
    parser = argparse.ArgumentParser(description = 'Description')
    parser.add_argument('-contain_dataset_feature',default='True', type=str,help="Whether to apply selectivity on a model level")
    parser.add_argument('-contain_data_similarity',default='True', type=str,help="Whether to apply selectivity on a model level")
    parser.add_argument('-contain_model_feature',default='True', type=str,help="contain_model_feature")
    parser.add_argument('-embed_dataset_feature',default='True', type=str, help='embed_dataset_feature')
    parser.add_argument('-embed_model_feature',default='True', type=str,help="embed_model_feature")
    parser.add_argument('-complete_model_features',default='True',type=str)

    parser.add_argument('-gnn_method',default='', type=str, help='contain_model_feature')
    parser.add_argument('-accuracy_thres',default=0.7, type=float, help='accuracy_thres')
    parser.add_argument('-finetune_ratio',default=1.0, type=float, help='finetune_ratio')
    parser.add_argument('-test_dataset',default='dmlab', type=str, help='remove all the edges from the dataset')
    parser.add_argument('-hidden_channels',default=128, type=int, help='hidden channels')

    parser.add_argument('-dataset_emb_method',default='domain_similarity',type=str) # task2vec
    args = parser.parse_args()
    print(f'args.contain_model_feature: {args.contain_model_feature}')
    print(f'bool - args.contain_model_feature: {str2bool(args.contain_model_feature)}')
    args.contain_data_similarity = str2bool(args.contain_data_similarity)
    args.contain_model_feature = str2bool(args.contain_model_feature)
    args.contain_dataset_feature = str2bool(args.contain_dataset_feature)
    args.embed_model_feature = str2bool(args.embed_model_feature)
    args.embed_dataset_feature = str2bool(args.embed_dataset_feature)
    args.complete_model_features = str2bool(args.complete_model_features)
    # set dataset_emb_method
    args.dataset_emb_method  = 'task2vec'

    main(args)
    
