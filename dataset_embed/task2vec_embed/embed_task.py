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
datasets_list = [
                    'food101','cifar10','cifar100','caltech101',
                    'stanfordcars','eurosat','clevr_count_all','clevr_closest_object_distance',
                    'dmlab', 'kitti_closest_vehicle_distance','flowers','pets',
                    'pcam','sun397','smallnorb_label_azimuth','smallnorb_label_elevation',
                    'svhn','resisc45','diabetic_retinopathy',
                    'cats_vs_dogs','keremberke/pokemon-classification','beans','poolrf2001/mask',
                    'Matthijs/snacks','keremberke/chest-xray-classification'
                ]
dataset_list = [dataset.__dict__[name]('../../datasets/')[0] for name in dataset_names] 

embeddings = []
for name, dataset in zip(dataset_names, dataset_list):
    print(f"Embedding {name}")
    probe_network = get_model('resnet34', pretrained=True, num_classes=int(max(dataset.targets)+1)).to(device)
    emb = Task2Vec(probe_network, max_samples=MAX_NUM_SAMPLES, skip_layers=6).embed(dataset)
    print(emb)
    embeddings.append(emb)
# task_similarity.plot_distance_matrix(embeddings, dataset_names)
with open('embedding.p', 'wb') as f:
    pickle.dump(embeddings, f)