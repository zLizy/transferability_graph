import pandas as pd

metric_type = 'rank' 
# metric_type = 'apk'
# metric_type = 'correlation'

test_datasets = [
                 'oxford_iiit_pet',
                 'oxford_flowers102',
                 'cifar100', 
                 'caltech101',
                 'dtd',
                 'stanfordcars', 
                #  'eurosat',
                #  'diabetic_retinopathy_detection',
                #  'smallnorb_label_azimuth',
                 'smallnorb_label_elevation',
                 'svhn',
                #  'kitti',
                #  'pets','flowers',
                 ]

data_feature_config = ['contain_dataset_feature'] #, 'not_contain_dataset_feature']


methods = [ 
            'lr_normalize','rf_normalize','svm_normalize','xgb_normalize',
            # 'xgb_data_distance',
            # 'xgb_logme_data_distance',
            # 'xgb_embedding_graph_dataset_scaling',
            # 'xgb_dataset_scaling',
            # 'xgb_embedding_graph',
            # 'NN_embedding_only_dataset_scaling',
            # 'xgb_node2vec_dataset_scaling',
            # 'xgb_node2vec_basic_dataset_scaling',
            # 'xgb_node2vec+_dataset_scaling',
            # 'xgb_node2vec+_basic_dataset_scaling',
            'xgb_data_distance_normalize',
            'xgb_logme_data_distance_normalize',
            'xgb_node2vec_normalize', 'xgb_node2vec+_normalize',
            'xgb_node2vec_basic_normalize', 'xgb_node2vec+_basic_normalize',
            'xgb_node2vec_all_normalize', 'xgb_node2vec+_all_normalize',
            'xgb_node2vec_without_accuracy_basic_normalize', 'xgb_node2vec+_without_accuracy_basic_normalize',
            'xgb_node2vec_without_accuracy_all_normalize', 'xgb_node2vec+_without_accuracy_all_normalize',
            
            'xgb_node2vec+_data_distance_normalize',
            'xgb_node2vec_data_distance_normalize',

            'svm_data_distance_normalize',
            'svm_logme_data_distance_normalize',
            'svm_node2vec_normalize', 'svm_node2vec+_normalize',
            'svm_node2vec_basic_normalize', 'svm_node2vec+_basic_normalize',
            'svm_node2vec_all_normalize', 'svm_node2vec+_all_normalize',
            'svm_node2vec_without_accuracy_basic_normalize', 'svm_node2vec+_without_accuracy_basic_normalize',
            'svm_node2vec_without_accuracy_all_normalize', 'svm_node2vec+_without_accuracy_all_normalize',
            'svm_node2vec+_data_distance_normalize',

            # 'rf_node2vec_without_accuracy',
            # 'rf_node2vec+_without_accuracy',
            # 'rf_node2vec_without_accuracy_basic',
            # 'rf_node2vec+_without_accuracy_basic',
            # 'rf_homo_SAGEConv_trained_on_transfer',
            # 'rf_homo_SAGEConv_trained_on_transfer_basic',
            # 'rf_homo_SAGEConv_trained_on_transfer_all',
            'rf_data_distance_normalize',
            'rf_logme_data_distance_normalize',
            'rf_node2vec_normalize', 'rf_node2vec+_normalize',
            'rf_node2vec_basic_normalize', 'rf_node2vec+_basic_normalize',
            'rf_node2vec_all_normalize', 'rf_node2vec+_all_normalize',
            'rf_node2vec_without_accuracy_basic_normalize', 'rf_node2vec+_without_accuracy_basic_normalize',
            'rf_node2vec_without_accuracy_all_normalize', 'rf_node2vec+_without_accuracy_all_normalize',
            'rf_node2vec+_data_distance_normalize',
            'rf_node2vec_data_distance_normalize',

            # 'lr_homo_GATConv_trained_on_transfer', 
            # 'lr_homo_SAGEConv_trained_on_transfer',
            # 'lr_node2vec_without_accuracy',
            # 'lr_node2vec+_without_accuracy',
            # 'lr_node2vec_without_accuracy_basic',
            # 'lr_node2vec+_without_accuracy_basic',
            # 'lr_homo_GATConv_without_accuracy',
            # 'lr_homo_SAGEConv_trained_on_transfer',
            # 'lr_homo_SAGEConv_trained_on_transfer_all',
            'lr_data_distance_normalize',
            'lr_logme_data_distance_normalize',
            'lr_node2vec_normalize', 'lr_node2vec+_normalize',
            'lr_node2vec_all_normalize', 'lr_node2vec+_all_normalize',
            'lr_node2vec_basic_normalize', 'lr_node2vec+_basic_normalize',
            'lr_node2vec_without_accuracy_basic_normalize', 'lr_node2vec+_without_accuracy_basic_normalize',
            'lr_node2vec_without_accuracy_all_normalize', 'lr_node2vec+_without_accuracy_all_normalize',
            'lr_node2vec+_data_distance_normalize',
            'lr_node2vec_data_distance_normalize',

            'lr_homoGATConv_normalize', 
            'lr_homo_SAGEConv_normalize',
            'lr_homoGCNConv_normalize',
            'rf_homoGATConv_normalize', 
            'rf_homo_SAGEConv_normalize',
            'rf_homoGCNConv_normalize',
            'xgb_homoGATConv_normalize', 
            'xgb_homo_SAGEConv_normalize',
            'xgb_homoGCNConv_normalize',
            'lr_homo_SAGEConv_basic_normalize',
                    'lr_homoGATConv_basic_normalize',
                    'lr_homoGCNConv_basic_normalize',
                    'rf_homo_SAGEConv_basic_normalize',
                    'rf_homoGATConv_basic_normalize',
                    'rf_homoGCNConv_basic_normalize',
                    'xgb_homo_SAGEConv_basic_normalize',
                    'xgb_homoGATConv_basic_normalize',
                    'xgb_homoGCNConv_basic_normalize',

                    'lr_homo_SAGEConv_all_normalize',
                    'lr_homoGATConv_all_normalize',
                    'lr_homoGCNConv_all_normalize',
                    'rf_homo_SAGEConv_all_normalize',
                    'rf_homoGATConv_all_normalize',
                    'rf_homoGCNConv_all_normalize',
                    'xgb_homo_SAGEConv_all_normalize',
                    'xgb_homoGATConv_all_normalize',
                    'xgb_homoGCNConv_all_normalize',
                    'lr_homo_SAGEConv_without_accuracy_all_normalize',
                    'lr_homoGATConv_without_accuracy_all_normalize',
                    'rf_homo_SAGEConv_without_accuracy_all_normalize',
                    'rf_homoGATConv_without_accuracy_all_normalize',
                    'xgb_homo_SAGEConv_without_accuracy_all_normalize',
                    'xgb_homoGATConv_without_accuracy_all_normalize',
                    'lr_homo_SAGEConv_without_accuracy_basic_normalize',
                    'lr_homoGATConv_without_accuracy_basic_normalize',
                    'rf_homo_SAGEConv_without_accuracy_basic_normalize',
                    'rf_homoGATConv_without_accuracy_basic_normalize',
                    'xgb_homo_SAGEConv_without_accuracy_basic_normalize',
                    'xgb_homoGATConv_without_accuracy_basic_normalize',
            'LogME',
            'NCE'
            ]

ratios = [1.0] #,0.7, 0.5, 0.3] # 0.8, 0.6 ,  
 
thres = '0.5'
top_neg_K = 0.5



top_K_config = {'correlation': [0],'rank':[5,10,15,20],'apk':[5,10,15,20]}

# top_K = 5

for ratio in ratios:
    print(f'\n ratio: {ratio}')
    ratio_addition = f'_r_{ratio}'
    for dataset_embed_method in ['domain_similarity']: #['domain_similarity']: #,  task2vec
        df_list = [] 
        if dataset_embed_method == 'domain_similarity':
            dataset_reference_model = 'google_vit_base_patch16_224'
            embed_addition = f''
        else:
            dataset_reference_model = 'resnet34'
            embed_addition = f'_{dataset_embed_method}'
        
        for top_K in top_K_config[metric_type]:
            for data_feature in data_feature_config:
                for hidden_channels in [128]: #32,64,
            
                    results = {k:{} for k in test_datasets}
                    for dataset in test_datasets:
                        print(f'\n ---- {dataset} ------')
                        if ratio > 1:
                            correlation_file = f'{dataset}/{metric_type}{embed_addition}_index.csv'
                        else:
                            correlation_file = f'{dataset}/{metric_type}{embed_addition}_index_ratio.csv'
                        # print(f'\n correlation_file: {correlation_file}')
                        df = pd.read_csv(correlation_file,index_col=0)

                        for method in methods:
                            top_neg_K = 0.5
                            filename = f'not_contain_model_feature/{dataset_embed_method}/\
metric,dataset_reference_model={dataset_reference_model},\
dataset_edge_distance_method=euclidean,model_dataset_edge_method=LogMe,hidden_channels={hidden_channels},\
top_pos_K=0.5,top_neg_K={top_neg_K},accu_pos_thres={thres},accu_neg_thres=0.5,\
distance_thres=-1.0,finetune_ratio={ratio}.csv' 
                            
                            if metric_type == 'correlation':     
                                metric_addition = ''
                            elif metric_type == 'rank' or metric_type == 'apk':
                                metric_addition = f'top_{top_K}:'

                            config = f'{metric_addition}0,../rank_final/{dataset}/{data_feature}/{filename}'
                            # print(f'\n config: {config}')
                            
                            try:
                                value = df.loc[config][method]
                            except Exception as e:
                                print(e)
                                # print('constant')
                                value = -20
                            if value != -20:
                                results[dataset][method] = value
                    
                    # print(f'\n results: {results}')
                    df_results = pd.DataFrame.from_dict(results)
                    df_results = df_results.astype('float64')
                    df_results['mean'] = df_results.mean(axis=1,numeric_only=True)
                    # print('\n df_results: ')
                    # print(df_results)
                    
                    # Reorganize the results: sort, reorder columns
                    df_results = df_results.sort_values(by=['mean'],ascending=False,na_position='last')
                    cols = df_results.columns.tolist()[::-1]
                    df_results = df_results[cols] 
                    if metric_type == 'correlation':
                        df_results.to_csv(f'results/{metric_type}{embed_addition}_{data_feature}_{hidden_channels}_{thres}{ratio_addition}.csv')
                    elif metric_type == 'rank' or metric_type =='apk':
                        df_results.to_csv(f"results/{metric_type}_{metric_addition.replace(':','')}{embed_addition}_{data_feature}_{hidden_channels}_{thres}{ratio_addition}.csv")
                    df_list.append(df_results)


                final = pd.concat(df_list)
                # final = final.convert_objects(convert_numeric=True)
                final = final.astype('float64').reset_index()
                # print(final)
                print(final.dtypes)
                final = final.groupby('index',as_index=False).mean()
                print(final)
                final.to_csv(f'results/{metric_type}{embed_addition}_{hidden_channels}_{thres}{ratio_addition}.csv')



        