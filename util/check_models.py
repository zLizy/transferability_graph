import os
import torch
import pandas as pd
from datasets import get_dataset_config_names
from datasets import load_dataset

from transformers import AutoConfig
from transformers import AutoModel,AutoTokenizer


# label2dataset = {v:k for k,v in dataset2label.items()}

df = pd.read_csv('../doc/ftrecords.csv',index_col=0)
df = df[df['framework']=='HuggingFaceText']
models = list(df['model_identifier'].value_counts().index)
print(f'====== len(models): {len(models)}')
# print(models)


file = '../doc/text_model_config.csv'
if os.path.exists(file):
#    df = pd.read_csv(file)
   df_new = pd.read_csv(file,index_col=0).copy()
else:
   df_new = pd.DataFrame(columns=[
                'model','input_shape','output_shape','architectures',
                'task','dataset','#labels','labels','task_specific_params',
                'problem_type','finetuning_task'])

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


for model_name in models:
    if model_name in list(df_new['model'].values): 
        continue
    else:
        print(f'======== model: {model_name}')

    try:
        config = AutoConfig.from_pretrained(model_name)
    except Exception as e:
        print(e)
        continue

    df_tmp = pd.DataFrame(columns=[
                'model','input_shape','output_shape','architectures',
                'task','dataset','#labels','labels','task_specific_params',
                'problem_type','finetuning_task'])

    df_tmp['architectures'] = config.architectures
    df_tmp['finetuning_task'] = config.finetuning_task
    df_tmp['#labels'] = config.num_labels
    df_tmp['labels'] = str(list(config.id2label.values()))
    df_tmp['task_specific_params'] = config.task_specific_params
    df_tmp['problem_type'] = config.problem_type
    df_tmp['model'] = model_name
    # df_tmp['input_shape'] = ''
    # df_tmp['task'] = ''
    # df_tmp['dataset'] = ''
    print(df_tmp)

    try:
        model = AutoModel.from_pretrained(model_name)
    except:
        model = AutoModel.from_pretrained(model_name,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer("I love AutoNLP", return_tensors="pt")
    print(f'== inputs.keys: {inputs.keys()}')

    try:
        outputs = model(**inputs)
    except:
        continue
    
    # Perform pooling. In this case, mean pooling.
    try:
        _embeddings = mean_pooling(outputs, inputs['attention_mask'])
        print(_embeddings.shape)
        df_tmp['output_shape'] = _embeddings.shape[1]
    except Exception as e:
        print(e)
        print('no attention_mask in inputs')
        continue
    
    df_new = pd.concat([df_new,df_tmp],ignore_index=True)
    print(df_new)
    df_new.to_csv(file)

############## check visual models
# hg_file = '../doc/hgpics_keyword.csv'
# df_hgpics = pd.read_csv(hg_file)
# keywords = df_hgpics['keywords'].values

# for i,row in df.iterrows():
#    print('-----------')
#    print(f"i, model_name: {i}, {row['model']}")
#    labels = row['labels']
#    print(f'labels: {labels}')
#    if labels in label2dataset.keys():
#       dataset = label2dataset[labels]
#       print(f'found {dataset}')
#       df_new.loc[i,'dataset'] = dataset
#       df_new.loc[i,'labels'] = dataset
#       # df.to_csv(file,index=False)
#    elif labels in keywords:
#       print(f'found {labels} -- huggingface pics')
#       df_new.loc[i,'dataset'] = 'hfpics'
#    else:
#       df_new.loc[i,'labels'] = row['dataset']
    
#    # configs = get_dataset_config_names(dataset_name)
#    # print(f'configs: {configs}')
#    # if dataset_name == 'cats_vs_dogs': continue
#    # try:
#    #     ds = load_dataset(dataset_name, split="test")
#    # except:
#    #     ds = load_dataset(dataset_name, 'full')
#    # try:
#    #     copy_of_features = ds.features.copy()
#    # except:
#    #     print(ds.keys())
#    #     copy_of_features = ds['test'].features.copy()
#    # # print(f'features: {copy_of_features}')
#    # print(f'feature_key: {copy_of_features.keys()}')
#    # key = [k for k in copy_of_features.keys() if 'label' in k]
#    # labels = copy_of_features[key[0]].names
#    # print(f'labels:{labels}')
# print(df_new.head())
# df_new.to_csv(file,index=False)
# remove_model = [
#                 'davanstrien/convnext_manuscript_iiif,davanstrien/davanstrien/flyswot_iiif',
#                 'amitkayal/ak-vit-base-patch16-224-in21k-image_classification','davanstrien/flyswot_iiif'
#                ]
# df_new = df_new[~df_new['model'].isin(remove_model)]


# df_dataset = df_new[df_new['dataset'].notna()]
# df_null_dataset = df_new[df_new['dataset'].isna()]
# print(f'df_dataset: {len(df_dataset)},df_null_dataset: {len(df_null_dataset)}')
# df_null_dataset.to_csv('../doc/model_config_null_dataset.csv',index=False)
# df_dataset.to_csv('../doc/model_config_dataset.csv',index=False)


