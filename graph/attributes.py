import torch
import os
import json
import pickle

import itertools
import scipy.spatial.distance as distance
import pandas as pd
import numpy as np

import sys

class GraphAttributes():
    PRINT = True
    FEATURE_DIM = 2048 #768 #2048
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

    def __init__(self,args):
        self.args = args
        self.root = '../'
        self.record_path = 'doc/records.csv'

        self.finetune_records = self.get_finetuned_records()
        # get node id
        self.unique_model_id, self.unique_dataset_id = self.get_node_id()
        
        # get dataset-dataset edge index
        if args.dataset_reference_model == 'resnet50' or args.dataset_reference_model == 'google_vit_base_patch16_224':
            self.base_dataset = 'imagenet'
        elif args.dataset_reference_model == 'Ahmed9275_Vit-Cifar100':
            self.base_dataset = 'cifar100'
        elif args.dataset_reference_model == 'johnnydevriese_vit_beans':
            self.base_dataset = 'beans'


    def get_dataset_edge_index(self,threshold=0.3,base_dataset='imagenet',sim_method='cosine'):
        n = self.data_features.shape[0]
        # print(f'len(dataset_features):{n}')
        # print(f'emb_method: {self.args.dataset_embed_method}')

        distance_matrix = np.zeros([n,n])
        data_source = []
        data_target = []
        attr = []
        # print(features)
        for i, e1 in enumerate(self.data_features):
            if sim_method == 'correlation':
                similarity = distance.correlation(e1,e1) #cosine(e1,e1) #1 - distance.cosine(e1,e1)
            elif sim_method == 'euclidean':
                similarity = distance.euclidean(e1,e1)
            # print(f'similarity: {similarity}')
            distance_matrix[i,i] = similarity
        for (i, e1), (j, e2) in itertools.combinations(enumerate(self.data_features), 2):
            if sim_method == 'correlation':
                similarity = distance.correlation(e1,e1) #cosine(e1,e1) #1 - distance.cosine(e1,e1)
            elif sim_method == 'euclidean':
                similarity = distance.euclidean(e1,e2)
            # similarity = kl(e1,e2)
            distance_matrix[i, j] = similarity
            distance_matrix[j, i] = similarity
            if threshold == -1 or similarity < threshold:
                data_source.append(i)
                data_target.append(j)
                attr.append(1-similarity) ## distance (smaller the better); similarity (higher the better ~ accuracy)
            # if similarity < threshold:
            # if i == 0:
            #     print(f'\n-- similarity between datasets: {similarity}')
        ## normalization
        attr = [(float(i)-min(attr))/(max(attr)-min(attr)) for i in attr]

        ## Filter distance with top K

        if not os.path.exists(f'../doc/corr_{self.args.dataset_embed_method}_{self.args.dataset_reference_model}_{base_dataset}.csv'):
            # if True:
            dict_distance = {}
            for i,name in enumerate(self.unique_dataset_id['dataset'].values):#
            # for i,name in enumerate(unique_dataset_id['classname_name'].values):
                dict_distance[name] = list(distance_matrix[i,:])
            df_tmp = pd.DataFrame(dict_distance)
            df_tmp.index = df_tmp.columns
            df_tmp.to_csv(f'../doc/corr_{self.args.dataset_embed_method}_{self.args.dataset_reference_model}_{base_dataset}.csv')# _class
        # data_source = np.asarray(data_source)
        # data_target = np.asarray(data_target)
        print(f'len(connected_dataset):{len(data_source)}')
        return torch.stack([torch.tensor(data_source), torch.tensor(data_target)]), torch.tensor(attr) #dim=0

    def get_edge_index(self,method='accuracy',ratio=1.0): # method='score'
        # if ratio != 1:
        #     df = df.sample(frac=ratio, random_state=1)
        if method == 'accuracy':
            df = self.finetune_records.copy()
            df['mean'] = df.groupby('dataset')['accuracy'].transform(lambda x: x.mean())
            df_neg = df[df['accuracy']<=self.args.accu_neg_thres]#df['mean']]
            if self.args.accu_pos_thres == -1:
                df = df[df['accuracy']> df['mean']]
            else:
                df = df[df['accuracy']> self.args.accu_pos_thres]
            # df = df[df['accuracy']>=self.args.accuracy_thres]
        elif method == 'score':
            df, df_neg = self.get_transferability_scores()
            #if self.args.gnn_method == 'node2vec':
            # df['mean'] = df.groupby('dataset')['score'].transform(lambda x: x.mean())
            # df_neg = df[df['score']<=df['mean']]
            # df = df[df['score']>=df['mean']]
            # print(f'df_neg: {df_neg}')
        
        edge_index_model_to_dataset, edge_attributes = self.get_edges(df,method,type='positive')
        negative_edges, _ = self.get_edges(df_neg,method,type='negative')
        
        return edge_index_model_to_dataset, edge_attributes, negative_edges

    def get_edges(self,df,method,type='positive'):
        print()
        print('==========')
        # print(f'len(df): {len(df)}')
        print(f'len(df) after filtering models by {method}: {len(df)}')
        mapped_model_id = pd.merge(df['model'],self.unique_model_id,on='model',how='left')['mappedID'].values# how='left
        mapped_dataset_id = pd.merge(df['dataset'],self.unique_dataset_id,on='dataset',how='left')['mappedID'].values
        print(f'mapped_model_id.len: {len(mapped_model_id)}, mapped_dataset_id.len: {len(mapped_dataset_id)}')
        edge_index_model_to_dataset = torch.stack([torch.from_numpy(mapped_model_id), torch.from_numpy(mapped_dataset_id)], dim=0)
        # if type == 'positive':
        #     edge_index_model_to_dataset = torch.stack([torch.from_numpy(mapped_model_id), torch.from_numpy(mapped_dataset_id)], dim=0)
        if type == 'negative':
            edge_index_model_to_dataset = torch.stack([torch.from_numpy(mapped_model_id), torch.from_numpy(mapped_dataset_id)], dim=1)
        # print(f'== edge_index_model_to_dataset')
        edge_attr = torch.from_numpy(df[method].values)
        # print()
        # print(df)
        # print(f'edge_attr: {edge_attr}')
        return edge_index_model_to_dataset, edge_attr

    def del_node(self,unique_id,entity_list,entity_type):
        ## Drop rows that do not produce dataset features
        print(f'len(unique_id): {len(unique_id)}')
        # unique_id = unique_id.drop(labels=delete_row_idx, axis=0)
        unique_id = unique_id[unique_id[entity_type].isin(entity_list)]
        print(f'len(unique_id): {len(unique_id)}')
        return unique_id

    def drop_nodes(self):
        # reallocate the node id
        a = set(self.unique_dataset_id['dataset'].unique())
        b = set(self.dataset_list)
        print('a-b',a-b)
        self.unique_dataset_id = self.get_unique_node(self.del_node(self.unique_dataset_id,self.dataset_list.keys(),'dataset')['dataset'],'dataset')
        self.unique_model_id = self.get_unique_node(self.del_node(self.unique_model_id,self.model_list,'model')['model'],'model')

        ## Perform merge to obtain the edges from models and datasets:
        self.finetune_records = self.finetune_records[self.finetune_records['model'].isin(self.unique_model_id['model'].values)]
        self.finetune_records = self.finetune_records[self.finetune_records['dataset'].isin(self.unique_dataset_id['dataset'].values)]
        print(f'len(df): {len(self.finetune_records)}')

    ## Retrieve the embeddings of the model
    def get_model_features(self,complete_model_features=False):
        model_feat = []
        DATA_EMB_METHOD = 'attribution_map'
        ATTRIBUTION_METHOD = 'input_x_gradient' #'input_x_gradient'#'saliency'
        INPUT_SHAPE = 128 #64 # #224
        model_list = []
        for i, row in self.unique_model_id.iterrows():
            print(f"======== i: {i}, model: {row['model']} ==========")
            model_match_rows = self.finetune_records.loc[self.finetune_records['model']==row['model']]
            # model_match_rows = df_config.loc[df['model']==row['model']]
            if model_match_rows.empty:
                if complete_model_features:
                    # delete_model_row_idx.append(i)
                    model_list.append(row['model'])
                else:
                    features = np.zeros(INPUT_SHAPE*INPUT_SHAPE)
                    model_feat.append(features)
                continue
            if model_match_rows['model'].values[0] == np.nan: 
                # delete_model_row_idx.append(i)
                model_list.append(row['model'])
                continue
            try:
                dataset_name = model_match_rows['dataset'].values[0].replace('/','_').replace('-','_')
                ds_name = dataset_name
                dataset_name = self.dataset_map[dataset_name] if dataset_name in self.dataset_map.keys() else dataset_name
            except:
                print('fail to retrieve model')
                continue
            if isinstance(dataset_name,list):
                # print(dataset_name)
                configs = self.finetune_records[self.finetune_records['dataset']==ds_name]['configs'].values[0].replace("'",'"')
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
                # delete_model_row_idx.append(i)
                model_list.append(row['model'])
                continue
            IMAGE_SHAPE = int(sorted(model_match_rows['input_shape'].values,reverse=True)[0])
            model_name = row['model']
            # if model_name in ['AkshatSurolia/BEiT-FaceMask-Finetuned','AkshatSurolia/ConvNeXt-FaceMask-Finetuned','AkshatSurolia/DeiT-FaceMask-Finetuned','AkshatSurolia/ViT-FaceMask-Finetuned','Amrrs/indian-foods','Amrrs/south-indian-foods']: 
            #     continue
            path = os.path.join(f'../model_embed/{DATA_EMB_METHOD}/feature',dataset_name,model_name.replace('/','_')+f'_{ATTRIBUTION_METHOD}.npy')
            print(dataset_name,model_name)
            
            # load model features
            try:
                features = np.load(path)
            except Exception as e:
                print('----------')
                print(e)
                if complete_model_features:
                    print(f'== Skip this model and delete it')
                    # delete_model_row_idx.append(i)
                    model_list.append(row['model'])
                    continue 
                else:
                    features = np.zeros((INPUT_SHAPE,INPUT_SHAPE))
                # features = np.zeros((INPUT_SHAPE,INPUT_SHAPE))
            print(f'features.shape: {features.shape}')
            if features.shape == (INPUT_SHAPE,INPUT_SHAPE):
                print('Try to obtain missing features')
                sys.path.append('..')
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
                    # delete_model_row_idx.append(i)
                    model_list.append(row['model'])
                    print(f'--- fail - skip row {row["model"]}')
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
        return model_feat, model_list #delete_model_row_idx

    def get_dataset_list(self):
        dataset_list = {}
        # delete_dataset_row_idx = []
        for i, row in self.unique_dataset_id.iterrows():
            ds_name = row['dataset']
            dataset_name = ds_name.replace('/','_').replace('-','_')
            # self._print('dataset_name',dataset_name)
            if dataset_name in ['davanstrien_iiif_manuscripts_label_ge_50',
                                'dsprites',
                                'age_prediction',
                                'FastJobs_Visual_Emotional_Analysis',
                            ]:
                # delete_dataset_row_idx.append(i)
                continue
            dataset_name = self.dataset_map[dataset_name] if dataset_name in self.dataset_map.keys() else dataset_name
            if isinstance(dataset_name,list):
                # print(dataset_name)
                configs = self.finetune_records[self.finetune_records['dataset']==ds_name]['configs'].values[0].replace("'",'"')
                print(configs)
                if ds_name == 'clevr':
                    dataset_name = json.loads(configs)['preprocess']
                else:
                    dataset_name = f"{ds_name}_{json.loads(configs)['label_name']}"
            # cannot load imagenet-21k and make them equal
            if dataset_name == 'imagenet_21k': 
                dataset_name = 'imagenet'
            dataset_list[ds_name] = dataset_name
        return dataset_list #, delete_dataset_row_idx


    def _print(self,name,value,level=2):
        if self.PRINT:
            print()
            if level == 1:
                print('=====================')
            elif level > 1:
                print('---------------')
            print(f'== {name}: {value}')

    ## Node idx
    def get_node_id(self):
        unique_model_id = self.get_unique_node(self.finetune_records['model'],'model')
        unique_dataset_id = self.get_unique_node(self.finetune_records['dataset'],'dataset')
        print()
        print(f"len(unique_model_id): {len(unique_model_id)}")
        print(f'len(unique_dataset_id): {len(unique_dataset_id)}')
        return unique_model_id, unique_dataset_id

    def get_unique_node(self,col,name):
        tmp_col = col.copy().dropna()
        unique_id = tmp_col.unique()
        unique_id = pd.DataFrame(data={
                name: unique_id,
                'mappedID': pd.RangeIndex(len(unique_id)),
            })
        return unique_id

    def get_transferability_scores(self):
        df = self.finetune_records.copy()[['dataset','model','accuracy']]
        # print(len(self.dataset_list),self.dataset_list)
        df_list = []
        df_neg_list = []
        for ori_dataset_name, dataset_name in self.dataset_list.items():
            df_sub = df[df['dataset']==ori_dataset_name]
            try:
                # path = f'../doc/{self.args.model_datset_edge_method}/scores/{dataset_name.replace(" ","-")}.csv'
                path = f'baselines/LogME_scores/{dataset_name.replace(" ","-")}.csv'
                # print(f'path: {path}')
                df_score = pd.read_csv(path,index_col=0)
                df_score = df_score[df_score['model']!='time']
                
                # normalize
                # norm = np.linalg.norm(df_score['score'].values)     # To find the norm of the array
                # print(norm)                        # Printing the value of the norm
                score = df_score['score']
                normalized_pred = (score-score.min())/(score.max()-score.min()) #df_score['score'].values/norm 
                df_score['score'] = normalized_pred
                # print('df_score')
                # print(df_score.sort_values(['score'],ascending=False).head())

                # top K = 20
                # K = 50
                # largest
                if self.args.top_pos_K <= 1:
                    df_large = df_score[df_score['score']>=self.args.top_pos_K]
                elif self.args.top_pos_K > 1:
                    df_large = df_score.nlargest(self.args.top_pos_K, 'score')
                # df_large = df_score
                # print(df_large.head())
                df_ = pd.merge(df_sub,df_large,on='model',how='left')
                # print(f'df_')
                # print(df_.sort_values(['score'],ascending=False).head())

                # smallest
                if self.args.top_neg_K <= 1:
                    df_small = df_score[df_score['score']<=self.args.top_neg_K]
                elif self.args.top_neg_K <= 1:
                    df_small = df_score.nsmallest(self.args.top_neg_K,'score')
                df_s = pd.merge(df_sub,df_small,on='model',how='left')

                # print(df_)
                df_list.append(df_)
                df_neg_list.append(df_s)
                # if 'score' in df.columns: 
                #     df = pd.merge(df,df_,on=['dataset','model','accuracy','score'],how='left')
                # else:
                #     df = pd.merge(df,df_,on=['dataset','model','accuracy'],how='left')
            except:
                df_list.append(df_sub)
                # continue
        
        df = pd.concat(df_list)
        # print(df.head())
        df = df.dropna()
        if 'score' not in df.columns: df['score'] = 0
        df_neg = pd.concat(df_neg_list).dropna()
        
        # print(f'df: {df}')
        # print(f'df_neg: {df_neg}')
        
        return df, df_neg
        
    def get_finetuned_records(self):
        file = 'doc/model_config_dataset.csv'
        # model configuration
        config = pd.read_csv(os.path.join(self.root,file))
        config['dataset'] = config['labels']
        config['configs'] = {}
        config['accuracy'] = 0
        available_models = config['model'].unique()

        # finetune results
        finetune_records = pd.read_csv(os.path.join(self.root,self.record_path))
        finetune_records['model'] = finetune_records['model']
        finetune_records['dataset'] = finetune_records['finetuned_dataset'] #finetune_records['train_dataset_name']
        finetune_records['accuracy'] = finetune_records['test_accuracy']
        finetune_records['input_shape'] = 0

        print()
        print(f'---- len(finetune_records_raw): {len(finetune_records)}')
        
        ##### Ignore pre-trained information
        ######################
        ## Delete the finetune records of the test datset
        ######################
        finetune_records = finetune_records[finetune_records['dataset']!=self.args.test_dataset]
        finetune_records = pd.concat([ config[['dataset','model','input_shape','accuracy']],
                                        finetune_records[['dataset','model','input_shape','accuracy']]],
                                            ignore_index=True)
        
        print(f'---- len(finetune_records_after_concatenating_model_config): {len(finetune_records)}')
        finetune_records['config'] = ''
        # filter models that are contained in the config file
        finetune_records = finetune_records[finetune_records['model'].isin(available_models)]
        
        # ######################
        # ## Add an empty row to indicate the dataset
        # ######################
        finetune_records.loc[len(finetune_records)] = {'dataset':self.args.test_dataset}
        finetune_records.index = range(len(finetune_records))
        
        return finetune_records

class GraphAttributesWithDomainSimilarity(GraphAttributes):
    def __init__(self,args,approach='domain_similarity'): 
        # invoking the __init__ of the parent class
        GraphAttributes.__init__(self, args)
        self.dataset_list = self.get_dataset_list()
        # print()
        # print('========= Extracting Graph Attributes =========')
        # print(self.unique_model_id.sort_values(['mappedID'],ascending=True).head())
        # print(self.unique_dataset_id.sort_values(['mappedID'],ascending=False).head())
        # print()

        # print(f'args.dataset_reference_model: {self.args.dataset_reference_model}')
        self.data_features = self.get_dataset_features(self.args.dataset_reference_model)
        if 'node2vec' in args.gnn_method or (not args.contain_model_feature):
            self.model_features = []
            self.model_list = self.unique_model_id['model'].unique()
        else:
            self.model_features, self.model_list = self.get_model_features()
        # get common nodes
        self.drop_nodes()
        self.max_dataset_idx = self.unique_dataset_id['mappedID'].max()
        ##########
        # get specific dataset index
        ##########
        if args.test_dataset != '':
            try:
                self.test_dataset_idx = self.unique_dataset_id[self.unique_dataset_id['dataset']==args.test_dataset]['mappedID'].values[0]
            except:
                # pass
                print('==== fail ====')
                print(args.test_dataset)
                # print(self.unique_dataset_id['dataset'].unique())
            ##### !!! make the indeces of the dataset and the model different
            if 'homo' in self.args.gnn_method or 'node2vec' in self.args.gnn_method:
                self.unique_model_id['mappedID'] += self.max_dataset_idx + 1
            # print(f'unique_model_id: {self.unique_model_id}')
            self.model_idx = self.unique_model_id['mappedID'].values
        else:
            self.test_dataset_idx = -1
            self.model_idx = -1

        self.node_ID = list(self.model_idx) + list(self.unique_dataset_id['mappedID'].values)

        # get edge index
        self.edge_index_accu_model_to_dataset, self.edge_attr_accu_model_to_dataset, self.accu_negative_pairs = self.get_edge_index(method='accuracy',ratio=args.finetune_ratio)#score
        
        self.edge_index_tran_model_to_dataset, self.edge_attr_tran_model_to_dataset, self.tran_negative_pairs = self.get_edge_index(method='score',ratio=args.finetune_ratio)#score
        if 'without_accuracy' in args.gnn_method or 'trained_on_transfer' in args.gnn_method:
            self.negative_pairs = self.tran_negative_pairs
        else:
            self.negative_pairs = self.accu_negative_pairs
        print()
        print('=========')
        print(f'-- max node index: {torch.max(self.edge_index_accu_model_to_dataset),0}, {torch.max(self.edge_index_tran_model_to_dataset),0}')
        
        self.dataset_reference_model = args.dataset_reference_model
        self.edge_index_dataset_to_dataset, self.edge_attr_dataset_to_dataset = self.get_dataset_edge_index(base_dataset=self.base_dataset,threshold=args.distance_thres,sim_method=args.dataset_distance_method)

    def get_dataset_features(self,reference_model):
        # sys.path.append(os.path.abspath('../'))
        # sys.path.append(os.getcwd())
        from dataset_embed.domain_similarity.embed import embed
        data_feat = []
        # print('\n ---')
        # print('len(dataset_list)',len(self.dataset_list))
        # print(f'reference_model: {reference_model}')
        dataset_list = self.dataset_list.copy()
        for ori_dataset_name, dataset_name in dataset_list.items():
            ds_name = dataset_name.replace(' ','-')
            path = os.path.join(f'../dataset_embed/domain_similarity/feature',reference_model,f'{ds_name}_feature.npy')
            try:
                features = np.load(path)
            except Exception as e:
                # print('----------')
                # print(e)
                features = np.zeros((1,self.FEATURE_DIM))
            # print()
            # print(f'feature:{features.shape}')
            x = features == np.zeros((1,self.FEATURE_DIM))
            # print(f'x.type: {type(x)}')
            # if (isinstance(x,bool) and x):
            if not os.path.exists(path):
                print(f'\nTry to obtain missing features of {dataset_name}')
                # features = embed('../',dataset_name)
                try:
                    features = embed('../',dataset_name,reference_model)
                except FileNotFoundError as e:
                    # print(e)
                    # print(f'== fail to retrieve features and delete row {i}')
                    # self.delete_dataset_row_idx.append(i)
                    del self.dataset_list[ori_dataset_name]
                    continue
            features = np.mean(features,axis=0)
            # print(f"shape of {dataset_name} is {features.shape}")
            data_feat.append(features)
        data_feat = np.stack(data_feat)
        # print(f'== data_feat.shape:{data_feat.shape}')
        return data_feat

class GraphAttributesWithTask2Vec(GraphAttributes):
    def __init__(self,args,approach='task2vec'): 
        # invoking the __init__ of the parent class
        GraphAttributes.__init__(self, args)
        
        self.dataset_list = self.get_dataset_list()
        # if args.contain_dataset_feature:
        self.data_features = self.get_dataset_features(args.dataset_reference_model)

        if args.gnn_method == 'node2vec' or (not args.contain_model_feature):
            self.model_features = []
            self.model_list = self.unique_model_id['model'].unique()
        else:
            self.model_features, self.model_list = self.get_model_features()
        # get common nodes
        self.drop_nodes()
        self.reference_model = 'resnet50'
        self.max_dataset_idx = self.unique_dataset_id['mappedID'].max()

        ##########
        # get specific dataset index
        ##########
        if args.test_dataset != '':
            self.test_dataset_idx = self.unique_dataset_id[self.unique_dataset_id['dataset']==args.test_dataset]['mappedID'].values[0]
            ##### !!! make the indeces of the dataset and the model different
            if 'homo' in self.args.gnn_method or 'node2vec' in self.args.gnn_method:
                self.unique_model_id['mappedID'] += self.max_dataset_idx + 1
            print(f'unique_model_id: {self.unique_model_id}')
            self.model_idx = self.unique_model_id['mappedID'].values
        else:
            self.test_dataset_idx = -1
            self.model_idx = -1

        self.node_ID = list(self.model_idx) + list(self.unique_dataset_id['mappedID'].values)

        # get edge index
        self.edge_index_accu_model_to_dataset, self.edge_attr_accu_model_to_dataset, self.accu_negative_pairs = self.get_edge_index(method='accuracy',ratio=args.finetune_ratio)#score
        self.edge_index_tran_model_to_dataset, self.edge_attr_tran_model_to_dataset, self.tran_negative_pairs = self.get_edge_index(method='score',ratio=args.finetune_ratio)#score
        if 'without_accuracy' in args.gnn_method:
            self.negative_pairs = self.tran_negative_pairs
        else:
            self.negative_pairs = self.accu_negative_pairs
        
        # get dataset-dataset edge index
        self.edge_index_dataset_to_dataset, self.edge_attr_dataset_to_dataset = self.get_dataset_edge_index(base_dataset=self.base_dataset,threshold=args.distance_thres,sim_method=args.dataset_distance_method)

    
    def get_dataset_features(self,reference_model='resnet50'):
        from dataset_embed.task2vec_embed.embed_task import embed
        data_feat = []
        dataset_list = self.dataset_list.copy()
        for ori_dataset_name, dataset_name in dataset_list.items():
            ds_name = dataset_name.replace(' ','-')
            path = os.path.join(f'../dataset_embed/task2vec_embed/feature',f'{ds_name}_feature.p')
            try:
                with open(path, 'rb') as f:
                    features = pickle.load(f).hessian
                features = features.reshape((1,features.shape[0]))
                FEATURE_DIM = features.shape[1]
            except Exception as e:
                # print('----------')
                # print(e)
                features = np.zeros((1,self.FEATURE_DIM))
            # print()
            # print(f'feature:{features.shape}')
            
            x = features == np.zeros((1,FEATURE_DIM))
            # print(f'x.type: {type(x)}')
            
            # if (isinstance(x,bool) and x):
            if not os.path.exists(path):
                print('Try to obtain missing features')
                # features = embed('../',dataset_name)
                try:
                    features = embed('../',dataset_name,reference_model)
                except FileNotFoundError as e:
                    # print(e)
                    # print(f'== fail to retrieve features and delete row {ds_name}')
                    # self.delete_dataset_row_idx.append(i)
                    del self.dataset_list[ori_dataset_name]
                    continue
            features = np.mean(features,axis=0)
            # print(f"shape of {dataset_name} is {features.shape}")
            data_feat.append(features)
        data_feat = np.stack(data_feat)
        # print(f'== data_feat.shape:{data_feat.shape}')
        return data_feat
