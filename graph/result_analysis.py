import pandas as pd
import numpy as np
import glob
import os
import logging
import sys
import re
from logging.handlers import TimedRotatingFileHandler

os.system('rm -r result.log')

def get_console_handler(formatter=False):
    console_handler = logging.StreamHandler(sys.stdout)
    if formatter:
        formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
        console_handler.setFormatter(formatter)
    return console_handler
def get_file_handler(log_file, formatter=False):
    file_handler = TimedRotatingFileHandler(log_file, when='midnight')
    if formatter:
        formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
        file_handler.setFormatter(formatter)
    return file_handler
def get_logger(logger_name, log_file, use_formatter=False):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG) # better to have too much log than not enough
    logger.addHandler(get_console_handler(use_formatter))
    logger.addHandler(get_file_handler(log_file, use_formatter))
    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False
    return logger

logger = get_logger(__name__, 'result.log', use_formatter=True)
# df_logger = get_logger(str(__name__)+'_dfs', 'logs/debug.log', use_formatter=False)
logger.info('Started')

dataset_name = 'cifar100' #'kitti'

df_config = pd.read_csv('../doc/model_config_dataset.csv')
config_model = df_config['model'].unique()

keys = ['contain_dataset_feature','dataset_embed','contain_model_feature','dataset_reference_model',
        'dataset_edge_distance_method','model_dataset_edge_method','gnn_method',
        'hidden_channels','top_pos_K','top_neg_K','accu_pos_thres','accu_neg_thres']
folder_keys = keys[:3]
folder_keys = [folder_keys[idx] for idx in [0,2,1]]
print(f'folder_keys: {folder_keys}')
file_keys = [key for key in keys if key not in folder_keys+['gnn_method']]
# p = re.compile(r'=[^_]+')
# re_score = p.findall(s[:-4])
# print(re_score)

def extract(filename):
    # lambda filename: pd.read_csv(filename,header=0).drop(index=15)
    df = pd.read_csv(filename,header=0).drop(index=15)
    df['file_path'] = filename
    # df.index = df['Unnamed: 0']
    return df

df_record = pd.read_csv('../doc/ftrecords_img.csv',index_col=0)[['model_identifier','train_dataset_name','test_accuracy']]
df_record_subset = df_record[df_record['train_dataset_name'] == dataset_name].drop_duplicates(subset=['model_identifier', 'train_dataset_name'], keep='last')

def set_output_dir(setting_dict):
    ######### set output folders (tree structure)
    dir = ''
    for key, value in setting_dict.items():
        value = value[1:]
        if 'Fals' in value:
            key = 'not_'+key
        if 'doma' in value:
            key = 'domain_similarity'
        if 'task' in value:
            key = 'task2vec'
        dir = os.path.join(dir,key)
    return dir

def set_file_dict(keys,values):
    _dict = {}
    for key,value in zip(keys,values):
        value = value.replace('=','')
        if 'resn' in value:
            value = 'resnet50'
        if 'corr' in value:
            value = 'correlation'
        if 'LogM' in value:
            value = 'LogMe'
        _dict[key] = value
    return _dict

def compute_and_save(gnn_method,results,setting_dict,file_dict):
    model_result = results['model'].unique()
    model_record = df_record_subset['model_identifier'].unique()
    model_lists = set.intersection(set(model_result),set(model_record))
    # add models in config
    model_lists = set.intersection(set(config_model),model_lists)
    results = results[results['model'].isin(model_lists)].drop_duplicates(subset=['model', 'mappedID'], keep='last')
    
    df_record_unique = df_record_subset[df_record_subset['model_identifier'].isin(model_lists)]

    gt_rank = df_record_unique.sort_values(by=['test_accuracy'],ascending=False)
    gt_rank.index = range(len(gt_rank))
    gt_rank_index = list(gt_rank.index)

    result_rank = gt_rank.reset_index().set_index('model_identifier')
    result_rank_index = result_rank.loc[results['model'].values]['index'].values
    
    # compute metrics
    apks = []
    sums = []
    from utils.ranking_metrics import apk
    for topK in [5,10,15,20,30,40,50]:
        _apk,running_sum = apk(gt_rank_index,result_rank_index,topK)
        apks.append(_apk)
        sums.append(running_sum)
        print(f'--- Top {topK} --- apk: {_apk}')
    apks.extend(sums)
    apks.append(np.mean(apks))
    print(f'\n apks: {apks}')
    index_name = [f'apk_@_{topK}' for topK in [5,10,15,20,30,40,50]]
    index_name += [f'sum_{topK}' for topK in [5,10,15,20,30,40,50]]
    index_name += ['map','num_model']
    print(f'\n index_name: {index_name}')

    # set output dir structure and return config list to delete
    dir = os.path.join('rank',dataset_name,set_output_dir(setting_dict))
    print(f'dir: {dir}')
    filename = 'metric_' + '_'.join(['{0}=={1}'.format(k,v) for k,v in file_dict.items()])+'.csv'
    print(f'\n--- filename: {filename}')
    path = os.path.join(dir,filename)
    metrics = {gnn_method : apks + [len(model_lists)]}

    if not os.path.exists(path):
        df = pd.DataFrame(metrics,index=index_name)
    else:
        df = pd.read_csv(path,index_col=0)
        df[gnn_method] = metrics[gnn_method]
    df.to_csv(path)

methods = ['node2vec','node2vec+','node2vec_without_attr','node2vec+_without_attr']
for method in methods:
    path = f'./rank/{dataset_name}/{method}'
    logger.info(f'\n ---- path: {path}')
    all_files = glob.glob(os.path.join(path,'*.csv'))
    p = re.compile(r'=[^_]+')
    for file in all_files:
        re_score = p.findall(file[:-4])
        # print(f'folder_keys: {folder_keys}')
        re_score = [re_score[i] for i in [0,2,1]+list(range(3,len(re_score)))]
        print(f're_score: {re_score}')
        setting_dict = {key:re_score[i] for i,key in enumerate(folder_keys)}
        file_dict = set_file_dict(file_keys,re_score[3:6]+re_score[7:])
        print(f'file_dict: {file_dict}')
        df = pd.read_csv(file,index_col=0)
        compute_and_save(method,df,setting_dict,file_dict)


# embed_method = 'domain_similarity'
embed_method = 'task2vec'
path = f'./rank/{dataset_name}/contain_dataset_feature/not_contain_model_feature/{embed_method}'
logger.info(f'\n ---- path: {path}')
all_files = glob.glob(os.path.join(path,'*.csv'))
# print(all_files[:4])

li_mapper = map(extract,all_files)
df_list = list(li_mapper)

df = pd.concat(df_list,ignore_index=True)

logger.info('\t'+ df.head().to_string().replace('\n', '\n\t')) 
# columns = ,SAGEConv,GATConv,HGTConv,node2vec,node2vec+

for top in ['sum_10','sum_20','sum_30','sum_50']:
    df_top = df[df['Unnamed: 0']==top]
    logger.info(f'\n----- {top} ------')
    for gnn_method in ['node2vec','node2vec+','SAGEConv','GATConv','HGTConv']:
        # SAGEConv
        df_ = df_top.nlargest(10,gnn_method)
        logger.info(f'\n {gnn_method}')
        df_[['hidden_channels', 'top_pos_K','top_neg_K','accu_pos_thres','accu_neg_thres']] = df_['file_path'].str.extract('.*\d+.*==(\d+).*==(\d+).*==(\d+).*([-10]+[.]\d+).*==(0.[\d+]).csv', expand=True)
        # logger.info('\t'+ df_[['Unnamed: 0',gnn_method,'file_path']].head().to_string().replace('\n', '\n\t')) 
        logger.info('\t'+ df_[['Unnamed: 0',gnn_method,'hidden_channels', 'top_pos_K','top_neg_K','accu_pos_thres','accu_neg_thres']].head().to_string().replace('\n', '\n\t')) 
        # ./rank/cifar100/contain_dataset_feature/not_contain_model_feature/domain_similarity/
        # metric_dataset_reference_model==resnet50_dataset_edge_distance_method==correlation
        # _model_dataset_edge_method==LogMe_hidden_channels==128_top_pos_K==200_top_neg_K==30
        # _accu_pos_thres==0.75_accu_neg_thres==0.2.csv

logger.info('Finished')