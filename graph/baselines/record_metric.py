import pandas as pd
import os
from glob import glob
import itertools
from itertools import permutations 

import sys
sys.path.append('../')
from utils.metric import record_metric

# dataset = 'cifar100' 
# dataset = 'dtd'
# dataset = 'caltech101'
# dataset = 'stanfordcars'
# dataset = 'eurosat'
# dataset = 'pets'
dataset = 'oxford_iiit_pet'
# dataset = 'oxford_flowers102'
# dataset = 'svhn'
# dataset = 'smallnorb_label_elevation'
# dataset = 'smallnorb_label_azimuth'
# dataset = 'diabetic_retinopathy_detection'
dataset = 'kitti'

baseline = 'LogME'
configs = ['contain_dataset_feature','not_contain_dataset_feature']
record_path = 'records.csv'

# for config in configs:
#     _dir = f'../rank/{dataset}/{config}/not_contain_model_feature/*/*'
#     files = glob(_dir)
#     print()
#     for file in files:
#         print(f'file - {file}')
#         record_result_metric('../../',dataset,baseline,record_path,df_logme,setting_dict={},filename=file)

root = '..'
_dir = f'{root}/rank/{dataset}'
gnn_methods = [folder for folder in os.listdir(_dir) if folder not in configs + ['.DS_Store']]
if 'LogME' not in gnn_methods:
    gnn_methods.append('LogME')
# gnn_methods = ['GATConv','node2vec+','GATConv_without_transfer']
contain_dataset_feature_config = [True,False]
dataset_embed_method_config  = ['domain_similarity']
contain_model_feature_config = [False]
dataset_reference_model_config = ['google_vit_base_patch16_224']
dataset_distance_method_config = ['correlation','euclidean']
model_dataset_edge_attribute_config = ['LogMe']
hidden_channels_config=[128] #1280
top_pos_K_config = [0.5] #,0.6]
top_neg_K_config = [0.2]
accu_pos_thres_config = [0.5,0.6, 0.7, -1.0]
accu_neg_thres_config = [0.2,0.3,0.4,0.5,0.7]
distance_thres_config=[-1.0]

value_lists = [
        contain_dataset_feature_config,
        dataset_embed_method_config,
        contain_model_feature_config,
        dataset_reference_model_config,
        dataset_distance_method_config,
        model_dataset_edge_attribute_config,
        hidden_channels_config,
        top_pos_K_config,
        top_neg_K_config,
        accu_pos_thres_config,
        accu_neg_thres_config,
        distance_thres_config,
    ]

df_corr_list = pd.DataFrame()
config_list = []

dataset_map = {'oxford_iiit_pet':'pets',
                'oxford_flowers102':'flowers'}
if dataset in dataset_map.keys():
    test_dataset = dataset_map[dataset]
else:
    test_dataset = dataset

for gnn_method in gnn_methods:
    # if gnn_method != 'lr_node2vec': continue
    if gnn_method == 'LogME':
        # # load logme score
        logme = pd.read_csv(f'LogME_scores/{test_dataset}.csv')
        results = logme[logme['model']!='time']
    if 'lr' in gnn_method and 'Conv' not in gnn_method and 'node2vec' not in gnn_method:
        results = pd.read_csv(os.path.join(_dir,gnn_method,'results.csv'))

    count = 0
    combinations = itertools.product(*value_lists)
    for combination in combinations:
        print(f'========= {gnn_method} =============')
        setting_dict = {
            'contain_dataset_feature':combination[0],
            # 'embed_dataset_feature':args.embed_dataset_feature,
            'dataset_embed_method': combination[1], 
            'contain_model_feature': combination[2],
            'dataset_reference_model':combination[3], #model_embed_method
            # 'embed_model_feature':args.embed_model_feature,
            'dataset_edge_distance_method': combination[4],
            'model_dataset_edge_method': combination[5],
            'hidden_channels':combination[6],
            'top_pos_K':combination[7],
            'top_neg_K':combination[8],
            'accu_pos_thres':combination[9],
            'accu_neg_thres':combination[10],
            'distance_thres':combination[11],
            }
        if 'node2vec' in gnn_method:
            connector = '==' # 
        else: 
            connector = '='
        
        pre_key = ['contain_dataset_feature','dataset_embed_method','contain_model_feature']
        pre_config = ['not_contain_model_feature','domain_similarity']
        if setting_dict['contain_dataset_feature'] == True:
            pre_config = ['contain_dataset_feature'] + pre_config
        else:
            pre_config = ['not_contain_dataset_feature'] + pre_config
            
        metric_file_path = 'metric,'+','.join([('{0}{1}{2}'.format(k, '==',str(v))) for k,v in setting_dict.items() if k not in pre_key])
        metric_file_path_full = os.path.join('../','rank',dataset,'/'.join(pre_config),metric_file_path+'.csv')
        # print(f'\nmetric_file_path: {metric_file_path}')

        # metric_file = glob(metric_file_path_full)
        print(f'\nmetric_file: {metric_file_path_full}')

        # if gnn_method == 'LogME':
        #     filename = metric_file_path #[7:]
        # else:
        filename = ','.join([('{0}{1}{2}'.format(k[:14], connector,str(v)[:5])) for k,v in setting_dict.items()])
        if 'without_transfer' in gnn_method:
            filename = 'without_transfer,' + filename
        file = os.path.join('../','rank',dataset,gnn_method,filename+'.csv')
        # print(f'\nfile: {file}')
        obtain_file = glob(file)
        print(f'obtain_file: {obtain_file}')
        
        # if metric_file != []:
        #     if metric_file[0] not in config_list:
        #         config_list.append(metric_file[0])

        if obtain_file != [] or gnn_method in ['LogME'] or 'lr' in gnn_method: 
            # ,mappedID,score,model,rank,test_accuracy
            # config_list.append(obtain_file[0])
            if obtain_file == [] and ('Conv' in gnn_method or 'node2vec' in gnn_method): 
                print(gnn_method)
                continue
            if (gnn_method not in  ['LogME']) and ('Conv' in gnn_method or 'node2vec' in gnn_method):
                result_file = obtain_file[0]
                results = pd.read_csv(result_file,index_col=0)
            else:
                result_file = file
                
            setting_dict['gnn_method'] = gnn_method
            corr = record_metric('correlation',dataset,setting_dict,results,save_path=result_file,root='../../')
            # corr = record_metric('correlation',dataset,gnn_method,results,setting_dict,root='../../')
            df_corr_list.loc[metric_file_path_full,gnn_method] = corr
            pass
        else: 
            pass

df_corr_list.to_csv(f'{dataset}/correlation_index.csv')
df_corr_list.index = range(len(df_corr_list))
df_corr_list.to_csv(f'{dataset}/correlation.csv')