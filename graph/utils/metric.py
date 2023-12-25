import pandas as pd
import os
import numpy as np
from scipy import stats

def get_records(root,test_dataset,record_path):
    df_record = pd.read_csv(os.path.join(root,'doc',record_path),index_col=0)[['model','finetuned_dataset','test_accuracy']]
    df_record = df_record[df_record['finetuned_dataset'] == test_dataset].sort_values(by=['test_accuracy'],ascending=False)
    df_record_subset = df_record.drop_duplicates(subset=['model', 'finetuned_dataset'], keep='first')
    return df_record_subset


def map_common_models(results,df_record_subset):
    # print(results.head())

    # get model intersection
    model_result = results['model'].unique()
    model_record = df_record_subset['model'].unique()
    model_lists = set.intersection(set(model_result),set(model_record))

    results = results[results['model'].isin(model_lists)].drop_duplicates(subset=['model'], keep='last')#(subset=['model', 'mappedID'], keep='last')
    ## Rank results by score
    results = results.sort_values(by=['score'],ascending=False)
    results.index = range(len(results))
    # print(f'\n --results: {results}')

    df_record_subset = df_record_subset[df_record_subset['model'].isin(model_lists)]

    return results, df_record_subset


def record_metric(method,test_dataset,gnn_method,results={},record_path='records.csv',metric_file_path='',root='../'):
    df_record_subset = get_records(root,test_dataset,record_path)
    
    df_results, df_record = map_common_models(results,df_record_subset)


    if method == 'correlation':
        record_correlation_metric(gnn_method,df_results,df_record,metric_file_path)
    elif method == 'rank':
        record_rank_metric(gnn_method,df_results,df_record,metric_file_path)


def record_correlation_metric(gnn_method,df_results,df_record,metric_file_path):
    df_results = df_results.drop(columns=['test_accuracy'])

    df_join = df_results.set_index('model').join(df_record.set_index('model'))
    # print('\n df_join:')
    # print(df_join.head())

    df_join.replace([-np.inf], 0, inplace=True)

    x = df_join['score']
    y = df_join['test_accuracy']
    # if (not x.isnull().values.any()) and (not y.isnull().values.any()):
    try:
        corr = correlation(x,y)
    except Exception as e:
        print(e)
        print(f'x: {x}')
        print(f'y: {y}')
        

    if not os.path.exists(metric_file_path):
        metrics = {'correlation':corr}
        df = pd.DataFrame.from_dict(metrics,orient='index',columns=[gnn_method])
    else:
        df = pd.read_csv(metric_file_path,index_col=0)
        df.loc['correlation',gnn_method] = corr
    df.to_csv(metric_file_path)
    # print(df)

    return corr

    

def record_rank_metric(root,test_dataset,gnn_method,record_path='records.csv',results={},setting_dict={},save_path='',filename=''):
    
    print(f'gnn_method: {gnn_method}')

    gt_rank = df_record_subset.copy()
    # print('\n --- gt_rank ---')
    # print(gt_rank.head())
    gt_rank.index = range(len(gt_rank))
    gt_rank_index = list(gt_rank.index)
    # print('\n --- gt_rank ---')
    # print(gt_rank_index)

    result_rank = gt_rank.reset_index().set_index('model')
    result_rank_index = result_rank.loc[results['model'].values]['index'].values

    results['rank'] = result_rank_index
    results['test_accuracy'] = result_rank.loc[results['model'].values]['test_accuracy']
    results = results.sort_values('rank')
    if 'LogME' not in gnn_method:
        try:
            results.to_csv(save_path)
        except:
            dir_path = os.path.join('./rank',f'{test_dataset}', gnn_method)
            results.to_csv(os.path.join(dir_path,'results.csv'))
    elif 'LogME' in gnn_method:
        results.to_csv(os.path.join('../rank',test_dataset,gnn_method,filename.split('/')[-1]))

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
    # path = os.path.join(dir,filename)
    metrics = {gnn_method : apks + [len(model_lists)]}

    # set output dir structure and return config list to delete
    # print(setting_dict)
    if 'LogME' not in gnn_method:
        setting_dict,dir = set_output_dir(os.path.join('rank',test_dataset),setting_dict)
        setting_dict = {k:v for k,v in setting_dict.items() if v != ''}
        print(setting_dict)
        filename = 'metric_' + ','.join(['{0}=={1}'.format(k,v) for k,v in setting_dict.items()])+'.csv'
        filename = os.path.join(dir,filename)
        print(f'\n--- filename: {filename}')

    if not os.path.exists(filename):
        df = pd.DataFrame(metrics,index=index_name)
    else:
        df = pd.read_csv(filename,index_col=0)
        df[gnn_method] = metrics[gnn_method]
    df.to_csv(filename)

def set_output_dir(base_dir,setting_dict):
    ######### set output folders (tree structure)
    delete_config_list = ['gnn_method','contain_dataset_feature','contain_model_feature','dataset_embed_method']
    dir = base_dir
    for i,key in enumerate(delete_config_list[:]):
        print(f'\n--- {key}: {setting_dict[key]}')
        if i != 0:
            if setting_dict[key]:
                if isinstance(setting_dict[key], str):
                    path = setting_dict[key]
                else: 
                    path = key
            else:
                path = f'not_{key}'
            dir = os.path.join(dir,path)
        setting_dict[key] = ''      

    if not os.path.exists(dir):
        os.makedirs(dir)
    return setting_dict, dir

def correlation(x,y): # correlation; rank
    corr = stats.pearsonr(x, y)
    print(f'\ncorr: {corr}')
    return corr[0] #.statistics

if __name__ == '__main__':
    metric()