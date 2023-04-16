from task2vec import Task2Vec
from models import get_model
import sys
sys.path.append('../../')
from util import dataset
import task_similarity
import pickle

dataset_names = ('imagenet','cifar10', 'cifar100', 'food101')
dataset_list = [dataset.__dict__[name]('../../datasets/')[0] for name in dataset_names] 

embeddings = []
for name, dataset in zip(dataset_names, dataset_list):
    print(f"Embedding {name}")
    probe_network = get_model('resnet34', pretrained=True, num_classes=int(max(dataset.targets)+1)).cuda()
    emb = Task2Vec(probe_network, max_samples=1000, skip_layers=6).embed(dataset)
    print(emb)
    embeddings.append(emb)
# task_similarity.plot_distance_matrix(embeddings, dataset_names)
with open('embedding.p', 'wb') as f:
    pickle.dump(embeddings, f)