from task2vec import Task2Vec
from models import get_model
import sys
sys.path.append('../../')
from util import dataset
import task_similarity
import pickle
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_NUM_SAMPLES = 5000

# dataset_names = ('imagenet','cifar10', 'cifar100', 'food101')
name_list = [
                    'food101','cifar10','cifar100','caltech101',
                    'stanfordcars','eurosat','clevr_count_all','clevr_closest_object_distance',
                    'dmlab', 'kitti_closest_vehicle_distance','flowers','pets',
                    'pcam','sun397','smallnorb_label_azimuth','smallnorb_label_elevation',
                    'svhn','resisc45','diabetic_retinopathy',
                    'cats_vs_dogs','keremberke/pokemon-classification','beans','poolrf2001/mask',
                    'Matthijs/snacks','keremberke/chest-xray-classification'
                ]
# dataset_list = [dataset.__dict__[name]('../../datasets/')[0] for name in name_list] 

embeddings = []
# for name, dataset in zip(name_list, dataset_list):
for i,name in enumerate(name_list[13:]):
    print('====================')
    print(f"Embedding {name}")
    name = name.replace('/','_').replace('-','_')
    ds = dataset.__dict__[name]('../../datasets/')[0]
    print('-- finish loading dataset')
    probe_network = get_model('resnet34', pretrained=True, num_classes=int(len(ds.classes))).to(device)
    emb = Task2Vec(probe_network, max_samples=MAX_NUM_SAMPLES, skip_layers=6).embed(ds)
    print(emb)
    embeddings.append(emb)
    # task_similarity.plot_distance_matrix(embeddings, dataset_names)
    with open(f'feature/{name}_feature.p', 'wb') as f:
        pickle.dump(embeddings, f)