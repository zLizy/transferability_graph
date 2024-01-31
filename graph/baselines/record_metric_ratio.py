import pandas as pd
import os
from glob import glob
import itertools
from itertools import permutations 

import sys
sys.path.append('../')
from utils.metric import record_metric


metric_type = 'rank' 
# metric_type = 'correlation'

datasets = [
            'cifar100' , 
            'dtd', 
            'caltech101',
            'stanfordcars', 
            'oxford_iiit_pet',
            'oxford_flowers102',
            'smallnorb_label_elevation',
            'svhn',
            # 'diabetic_retinopathy_detection',
            # 'kitti',
            # 'pets','flowers',
            # 'smallnorb_label_azimuth',
            # 'eurosat', 
            ]

# datasets = ['cifar100']#'oxford_flowers102', 'eurosat']

baseline = 'LogME'
configs = ['contain_dataset_feature']#,'not_contain_dataset_feature']
record_path = 'records.csv'


for dataset in datasets:

    dataset_map = {'oxford_iiit_pet':'pets',
                    'oxford_flowers102':'flowers'}
    if dataset in dataset_map.keys():
        test_dataset = dataset_map[dataset]
    else:
        test_dataset = dataset

    root = '..'
    _dir = f'{root}/rank_final/{test_dataset}'
    gnn_methods = [folder for folder in os.listdir(_dir) if folder not in configs + ['.DS_Store']]
    
    if 'LogME' not in gnn_methods:
        gnn_methods.append('LogME')
    # if 'NCE' not in gnn_methods:
    #     gnn_methods.append('NCE')
    # gnn_methods = ['GATConv','node2vec+','GATConv_without_transfer']
    contain_dataset_feature_config = [True]#,False]
    dataset_embed_method_config  = ['domain_similarity'] #[' ']#  task2vec
    contain_model_feature_config = [False]
    dataset_reference_model_config = ['google_vit_base_patch16_224']
    dataset_distance_method_config = ['euclidean']
    model_dataset_edge_attribute_config = ['LogMe']
    hidden_channels_config=[128,64,32] #1280
    top_pos_K_config = [0.5] #,0.6]
    top_neg_K_config = [0.5] # 0.2
    accu_pos_thres_config = [0.5] #[0.5,0.6, 0.7, -1.0]
    accu_neg_thres_config = [0.5]# [0.2,0.3,0.4,0.5,0.7]
    distance_thres_config=[-1.0]
    finetune_ratio_config = [1.0]#,0.7,0.5,0.3] # 0.8, 0.6,  
    run_config = range(1)

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
            finetune_ratio_config,
            run_config
        ]

    df_corr_list = pd.DataFrame()
    df_apk_list = pd.DataFrame()

    config_list = []

    for gnn_method in gnn_methods:
        # if gnn_method != 'lr_homo_SAGEConv': continue
        if (not 'normalize' in gnn_method) or (not gnn_method in ['NCE','LogME']) : pass
        
        if gnn_method == 'LogME':
            # # load logme score
            logme = pd.read_csv(f'LogME_scores/{test_dataset}.csv')
            results = logme[logme['model']!='time']
        # if gnn_method == 'NCE':
        #     # # load logme score
        #     try:
        #         nce = pd.read_csv(f'NCE_scores/{test_dataset}.csv')
        #         results = nce[nce['model']!='time']
        #     except:
        #         continue
        
        count = 0
        combinations = itertools.product(*value_lists)
        for combination in combinations:
            print(f'\n========= {dataset} - {gnn_method} =============')
            
            run = combination[13]
            print(f'\n------- run: {run}')

            if combination[1] == 'task2vec':
                dataset_reference_model = 'resnet34'
            else:
                dataset_reference_model = combination[3]

            setting_dict = {
                'contain_dataset_feature':combination[0],
                # 'embed_dataset_feature':args.embed_dataset_feature,
                'dataset_embed_method': combination[1], 
                'contain_model_feature': combination[2],
                'dataset_reference_model':dataset_reference_model, #model_embed_method
                # 'embed_model_feature':args.embed_model_feature,
                'dataset_edge_distance_method': combination[4],
                'model_dataset_edge_method': combination[5],
                'hidden_channels':combination[6],
                'top_pos_K':combination[7],
                'top_neg_K':combination[8],
                'accu_pos_thres':combination[9],
                'accu_neg_thres':combination[10],
                'distance_thres':combination[11],
                'finetune_ratio':combination[12],
                }
            if 'node2vec' in gnn_method:
                connector = '=' # 
            else: 
                connector = '='
            
            addition = ''
            if combination[1] == 'task2vec':
                addition = f'_{combination[1]}'

            if gnn_method != 'LogME' and gnn_method != 'NCE':
                finetune_ratio = combination[12]
                hidden_channels = combination[6]
                try:
                    if  finetune_ratio<= 1:
                        # _dir = f'{root}/rank_final/{test_dataset}'
                        results = pd.read_csv(os.path.join(_dir,gnn_method,f'results{addition}_{finetune_ratio}_{hidden_channels}_{run}.csv'))
                    else:
                        resutls = pd.read_csv(os.path.join(_dir,gnn_method,f'results{addition}_{hidden_channels}_{run}.csv'))
                except:
                    results = pd.DataFrame()
            # if 'all' in gnn_method or 'basic' in gnn_method or 'without_accuracy' in gnn_method or 'e2e' in gnn_method or 'xgb' in gnn_method:
            #     # # results
            #     try:
            #         file = os.path.join(_dir,gnn_method,f'results{addition}_{combination[12]}.csv')
            #         results = pd.read_csv(file,index_col=0)
            #         print(f' --- suceed loading {gnn_method} file: {file}')
            #     except Exception as e:
            #         print(e)
            #         results = pd.DataFrame()
            # elif ('lr' in gnn_method or 'rf' in gnn_method or 'xgb' in gnn_method or 'embedding' in gnn_method or 'svm' in gnn_method) \
            #     and 'Conv' not in gnn_method \
            #     and 'node2vec' not in gnn_method:
            #     print(f'gnn_method: {gnn_method}')
            #     
            

            pre_key = ['contain_dataset_feature','dataset_embed_method','contain_model_feature']
            pre_config = ['not_contain_model_feature',combination[1]]
            if setting_dict['contain_dataset_feature'] == True:
                pre_config = ['contain_dataset_feature'] + pre_config
            else:
                pre_config = ['not_contain_dataset_feature'] + pre_config
                
            metric_file_path = 'metric,'+','.join([('{0}{1}{2}'.format(k, '=',str(v))) for k,v in setting_dict.items() if k not in pre_key])
            metric_file_path_full = f"{run},{os.path.join('../','rank_final',dataset,'/'.join(pre_config),metric_file_path+'.csv')}"
            # print(f'\nmetric_file_path: {metric_file_path}')


            filename = ','.join([('{0}{1}{2}'.format(k[:14], connector,str(v)[:5])) for k,v in setting_dict.items()])
            if 'without_transfer' in gnn_method:
                filename = 'without_transfer,' + filename
            
            result_file = os.path.join('../','rank_final',test_dataset,gnn_method,filename+'.csv')
            # print(f'\nfile: {file}')
            # obtain_file = glob(file)

            # if  obtain_file != [] or \
            if   gnn_method in ['LogME','NCE'] or \
                'lr' in gnn_method or \
                'rf' in gnn_method or \
                'xgb' in gnn_method or \
                'svm' in gnn_method or \
                'embedding' in gnn_method or \
                'e2e' in gnn_method   : 
                # ,mappedID,score,model,rank,test_accuracy
                # config_list.append(obtain_file[0])

                # if 'all' in gnn_method or 'basic' in gnn_method or 'without_accuracy' in gnn_method or 'e2e' in gnn_method:
                #     result_file = file
                # elif obtain_file == [] and ('Conv' in gnn_method or 'node2vec' in gnn_method): 
                #     result_file = file
                #     # print(gnn_method)
                #     # continue
                # elif (gnn_method not in  ['LogME','NCE']) and ('Conv' in gnn_method or 'node2vec' in gnn_method):
                #     result_file = obtain_file[0]
                #     results = pd.read_csv(result_file,index_col=0)
                # else:
                #     result_file = file

                # if obtain_file != []:
                #     result_file = obtain_file[0]
                # else:
                #    result_file = file

                # print(f'\n result_file: {result_file}')
                # print()
                # print(results.head())
                    
                # pass empty results
                if len(results) == 0: continue

                ##  identify common models:
                model_list = list(pd.read_csv('./results/model_list.csv',index_col=0).index)
                results = results[results['model'].isin(model_list)]


                setting_dict['gnn_method'] = gnn_method
                if metric_type == 'correlation':
                    corr, _ = record_metric(metric_type,dataset,setting_dict,results,save_path=result_file,root='../../')
                    # corr = record_metric('correlation',dataset,gnn_method,results,setting_dict,root='../../')
                    df_corr_list.loc[metric_file_path_full,gnn_method] = corr
                else:
                    accu_dict, apk_dict = record_metric(metric_type,dataset,setting_dict,results,save_path=result_file,root='../../')
                    # print(f'\n accu_dict')
                    # print(accu_dict)
                    for key,value in accu_dict.items():
                        df_corr_list.loc[f'top_{key}:{metric_file_path_full}',gnn_method] = value
                    for key,value in apk_dict.items():
                        df_apk_list.loc[f'top_{key}:{metric_file_path_full}',gnn_method] = value
                    
                    # print('succeed')
            else: 
                pass

    df_corr_list.to_csv(f'{dataset}/{metric_type}{addition}_index_ratio.csv')
    df_corr_list.index = range(len(df_corr_list))
    df_corr_list.to_csv(f'{dataset}/{metric_type}{addition}_ratio.csv')

    if not df_apk_list.empty:
        df_apk_list.to_csv(f'{dataset}/apk{addition}_index_ratio.csv')
        df_apk_list.index = range(len(df_apk_list))
        df_apk_list.to_csv(f'{dataset}/apk{addition}_ratio.csv')
