import os
import sys
sys.path.append('../../')
from util import dataset
import pickle
# import task_similarity
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_NUM_SAMPLES = 5000

# dataset_names = ('imagenet','cifar10', 'cifar100', 'food101')
name_list = ['clevr_count_all','clevr_closest_object_distance']
# name_list = [
#                 'food101','cifar10','cifar100','caltech101',
#                 'stanfordcars','eurosat','clevr_count_all','clevr_closest_object_distance',
#                 'dmlab', 'kitti','oxford_flowers102','oxford_iiit_pet',
#                 'pcam','sun397','smallnorb_label_azimuth','smallnorb_label_elevation',
#                 'svhn','resisc45','diabetic_retinopathy_detection',
#                 'cats_vs_dogs','keremberke/pokemon-classification','beans','poolrf2001/mask',
#                 'Matthijs/snacks','chest-xray-classification'
#             ]
# dataset_list = [dataset.__dict__[name]('../../datasets/')[0] for name in name_list] 

def embed(root,dataset_name,input_shape=224,save=True):
    try:
        from task2vec import Task2Vec
        from models import get_model
    except:
        from dataset_embed.task2vec_embed.task2vec import Task2Vec
        from dataset_embed.task2vec_embed.models import get_model
    if ']' in dataset_name:
        classes = dataset_name.strip('][').replace("'","").split(', ')
        ds_type = 'hfpics'
        ds = dataset.__dict__[ds_type](os.path.join(root,'datasets'),classes,input_shape)[0]
    else:
        ds = dataset.__dict__[dataset_name.lower()](os.path.join(root,'datasets'))[0]
    print('-- finish loading dataset')
    probe_network = get_model('resnet34', pretrained=True, num_classes=int(len(ds.classes))).to(device)
    emb = Task2Vec(probe_network, max_samples=MAX_NUM_SAMPLES, skip_layers=6).embed(ds)
    print(emb)
    # embeddings.append(emb)
    # task_similarity.plot_distance_matrix(embeddings, dataset_names)
    dataset_name = dataset_name.replace(' ','-')
    if save == False:
        return emb
    elif save == True:
        with open(os.path.join(root,'dataset_embed/task2vec_embed/feature',f'{dataset_name}_feature.p'), 'wb') as f:
            pickle.dump(emb, f)
        return emb.hessian
    
def embed_offline(root,dataset_name,input_shape=224):
    emb = embed(root,dataset_name,input_shape,save=False)
    with open(f'{root}/feature/{dataset_name}_feature.p', 'wb') as f:
        pickle.dump(emb, f)
    return emb.hessian

if __name__ == '__main__':
    embeddings = []
    # for name, dataset in zip(name_list, dataset_list):
    for i,name in enumerate(name_list[1:]):
        print('====================')
        print(f"Embedding {name}")
        name = name.replace('/','_').replace('-','_')
        if os.path.exists(f'./feature/{name}.p'): continue    

        emb = embed('.',name,input_shape=224)
        