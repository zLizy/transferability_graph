import numpy as np
import pandas as pd
import torch
print(torch.__version__)

from sklearn.metrics import dcg_score
from sklearn.metrics import ndcg_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(f"Device: '{device}'")

import scipy.spatial.distance as distance
# distance.cosine(h1, h2)
import tqdm
import os
import json
import pickle
import sys
# sys.path.append(os.getcwd())
sys.path.append('../')

# from dataset_embed.task2vec_embed import task2vec
# sys.modules['task2vec'] = task2vec
from utils._util import *
from utils.metric import record_metric #record_result_metric
from attributes import *
import argparse

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def djoin(ldict, req=''):
    return req + ' & '.join([('{0} == "{1}"'.format(k, v)) if isinstance(v,str) else ('{0} == {1}'.format(k, v))  for k,v in ldict.items()])


def main(args):
    print()
    print('======================= Begin New Session ==========================')
    root = '../'
    # root = './'
    path = os.path.join(root,'doc',f'performance_{args.dataset_embed_method}_{args.gnn_method}_score.csv')
    args.path = path
    print(f'====== path: {path} ======')
    if os.path.exists(path):
        df_perf = pd.read_csv(path,index_col=0)
    else:
        df_perf = pd.DataFrame(
            columns=[
                        'contain_data_similarity',
                        'dataset_edge_distance_method',
                        
                        'contain_dataset_feature',
                        'embed_dataset_feature',
                        'dataset_embed_method',
                        'dataset_reference_model',

                        'contain_model_feature',
                        'embed_model_feature',
                        'model_embed_method',
                        'complete_model_features',
                        'model_dataset_edge_method',

                        'distance_thres',
                        'top_pos_K',
                        'top_neg_K',
                        'accu_pos_thres',
                        'accu_neg_thres',

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
        # 'contain_data_similarity':args.contain_data_similarity,
        'contain_dataset_feature':args.contain_dataset_feature,
        # 'embed_dataset_feature':args.embed_dataset_feature,
        'dataset_embed_method': args.dataset_embed_method, 
        'contain_model_feature': args.contain_model_feature,
        'dataset_reference_model':args.dataset_reference_model, #model_embed_method
        # 'embed_model_feature':args.embed_model_feature,
        'dataset_edge_distance_method': args.dataset_distance_method,
        'model_dataset_edge_method': args.model_dataset_edge_attribute,
        'gnn_method': args.gnn_method,
        # 'accuracy_thres':args.accuracy_thres,
        # 'complete_model_features':args.complete_model_features,
        'hidden_channels':args.hidden_channels,
        'top_pos_K':args.top_pos_K,
        'top_neg_K':args.top_neg_K,
        'accu_pos_thres':args.accu_pos_thres,
        'accu_neg_thres':args.accu_neg_thres,
        'distance_thres':args.distance_thres,
        'finetune_ratio':args.finetune_ratio,
    }
    if args.dataset_reference_model != 'resnet50':
        setting_dict['dataset_reference_model'] = args.dataset_reference_model
    print()
    print('======= evaluation_dict ==========')
    evaluation_dict = setting_dict.copy()
    evaluation_dict['test_dataset'] = args.test_dataset
    # evaluation_dict['contain_data_similarity'] = args.contain_data_similarity
    # evaluation_dict['embed_dataset_feature']=args.embed_dataset_feature
    # evaluation_dict['model_embed_method'],complete_model_features
    for k,v in evaluation_dict.items():
        print(f'{k}: {v}')
    print(f'gnn_method: {args.gnn_method}\n')
    # print(evaluation_dict)

    ## Check executed
    query = ' & '.join(list(map(djoin, [evaluation_dict])))
    # print(query)
    df_tmp = df_perf.query(query)

    ## skip running because the performance exist
    if not df_tmp.empty: #df_tmp.dropna().empty: 
        print('===== pass ====\n')
        # return 0
        pass
    else:
        print(f'query: {query}')
        # print(df_tmp.dropna())

    if args.gnn_method == 'lr':
        graph_attributes = GraphAttributes(args)
    elif args.dataset_embed_method == 'domain_similarity':
        graph_attributes = GraphAttributesWithDomainSimilarity(args)
    elif args.dataset_embed_method == 'task2vec':
        graph_attributes = GraphAttributesWithTask2Vec(args)
    else:
        graph_attributes = GraphAttributes(args)

    data_dict = {
                    'unique_dataset_id':            graph_attributes.unique_dataset_id,
                    'data_feat':                    graph_attributes.data_features,
                    'unique_model_id':              graph_attributes.unique_model_id,
                    'model_feat':                   graph_attributes.model_features,
                    'edge_index_accu_model_to_dataset':  graph_attributes.edge_index_accu_model_to_dataset,
                    'edge_attr_accu_model_to_dataset':   graph_attributes.edge_attr_accu_model_to_dataset,
                    'edge_index_tran_model_to_dataset':  graph_attributes.edge_index_tran_model_to_dataset,
                    'edge_attr_tran_model_to_dataset':   graph_attributes.edge_attr_tran_model_to_dataset,
                    'edge_index_dataset_to_dataset':graph_attributes.edge_index_dataset_to_dataset,
                    'edge_attr_dataset_to_dataset': graph_attributes.edge_attr_dataset_to_dataset,   
                    'negative_pairs':               graph_attributes.negative_pairs,
                    'test_dataset_idx':                  graph_attributes.test_dataset_idx,
                    'model_idx':                    graph_attributes.model_idx,
                    'node_ID':                      graph_attributes.node_ID,
                    'max_dataset_idx':              graph_attributes.max_dataset_idx,
                    'finetune_records':             graph_attributes.finetune_records,
                    'model_config':                 graph_attributes.model_config
                    # 'max_model_idx':                graph_attributes.max_model_idx
            }
    
    # print()
    # print(f'data_dict["edge_attr_model_to_dataset"]: {data_dict["edge_attr_model_to_dataset"]}')

    batch_size = 16

    if 'node2vec+' in args.gnn_method:
        from graph.methods.train_with_node2vec import node2vec_train
        results, save_path = node2vec_train(args,df_perf,data_dict,evaluation_dict,setting_dict,batch_size,extend=True)
    elif 'node2vec' in args.gnn_method:
        from graph.methods.train_with_node2vec import node2vec_train
        results, save_path = node2vec_train(args,df_perf,data_dict,evaluation_dict,setting_dict,batch_size)
    elif 'Conv' in args.gnn_method:
        from graph.methods.train_with_GNN import gnn_train
        results, save_path = gnn_train(args,df_perf,data_dict,evaluation_dict,setting_dict,batch_size,custom_negative_sampling=True)
    elif args.gnn_method == '""':
        from utils.basic import get_basic_features
        results, save_path = get_basic_features(args.test_dataset,data_dict,setting_dict)
    elif 'lr' in args.gnn_method:
        from graph.methods.train_with_linear_regression import lr_train
        lr_train(args,graph_attributes)
    elif 'dataset' in args.gnn_method:
        from utils.rank_dataset import rank_by_dataset
        # results, save_path = rank_by_dataset()
        pass
    
    if isinstance(results,int): return 0
    
    setting_dict['gnn_method'] = args.gnn_method
    # metric_file_path = 'metric,'+','.join([('{0}{1}{2}'.format(k, '==',str(v))) for k,v in setting_dict.items() if k not in pre_key])
    # metric_file_path_full = os.path.join('../','rank',dataset,'/'.join(pre_config),metric_file_path+'.csv')
    record_metric('correlation',args.test_dataset,setting_dict,results,args.record_path,save_path,root='../')
    # method,test_dataset,setting_dict,gnn_method,results={},record_path='records.csv',metric_file_path='',root='../'
    #('../',args.test_dataset,args.gnn_method,args.record_path,results,setting_dict,save_path)



if __name__ == '__main__':

    '''
    Configurations
    '''
    parser = argparse.ArgumentParser(description = 'Description')
    parser.add_argument('-contain_dataset_feature',default='True', type=str,help="Whether to apply selectivity on a model level")
    parser.add_argument('-contain_data_similarity',default='True', type=str,help="Whether to apply selectivity on a model level")
    parser.add_argument('-contain_model_feature',default='False', type=str,help="contain_model_feature")
    parser.add_argument('-embed_dataset_feature',default='True', type=str, help='embed_dataset_feature')
    parser.add_argument('-embed_model_feature',default='True', type=str,help="embed_model_feature")
    parser.add_argument('-complete_model_features',default='True',type=str)
    parser.add_argument('-dataset_reference_model',default='resnet50',type=str)

    parser.add_argument('-gnn_method',default='SAGEConv', type=str, help='contain_model_feature')
    parser.add_argument('-accuracy_thres',default=0.7, type=float, help='accuracy_thres')
    parser.add_argument('-finetune_ratio',default=1.0, type=float, help='finetune_ratio')
    parser.add_argument('-test_dataset',default='dmlab', type=str, help='remove all the edges from the dataset')
    parser.add_argument('-hidden_channels',default=128, type=int, help='hidden channels') #128
    
    parser.add_argument('-top_pos_K',default=50, type=float, help='hidden channels')
    parser.add_argument('-top_neg_K',default=20, type=float, help='hidden channels')
    parser.add_argument('-accu_pos_thres',default=0.6, type=float, help='hidden channels')
    parser.add_argument('-accu_neg_thres',default=0.2, type=float, help='hidden channels')
    parser.add_argument('-distance_thres',default=-1,type=float)

    parser.add_argument('-dataset_embed_method',default='domain_similarity',type=str) #  task2vec
    parser.add_argument('-dataset_distance_method',default='euclidean',type=str) # correlation
    parser.add_argument('-model_dataset_edge_attribute',default='LogMe',type=str) # correlation

    parser.add_argument('--record_path',default='records.csv',type=str)
    
    
    args = parser.parse_args()
    print(f'args.contain_model_feature: {args.contain_model_feature}')
    print(f'bool - args.contain_model_feature: {str2bool(args.contain_model_feature)}')
    
    # args.dataset_embed_method  =  'task2vec' #'' # 
    
    if args.dataset_embed_method == 'domain_similarity':
        args.dataset_reference_model = 'google_vit_base_patch16_224' #''resnet50'#' #'Ahmed9275_Vit-Cifar100' #''johnnydevriese_vit_beans
    elif args.dataset_embed_method == 'task2vec':
        args.dataset_reference_model = 'resnet34'

    # args.gnn_method = 'node2vec+' #'GATConv' #'HGTConv' #'SAGEConv' 

    args.contain_data_similarity = str2bool(args.contain_data_similarity)
    args.contain_model_feature = str2bool(args.contain_model_feature)
    args.contain_dataset_feature = str2bool(args.contain_dataset_feature)

    args.embed_model_feature = str2bool(args.embed_model_feature)
    args.embed_dataset_feature = str2bool(args.embed_dataset_feature)
    args.complete_model_features = str2bool(args.complete_model_features)

    
    main(args) 
    
