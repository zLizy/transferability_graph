import string
import sys

from dataset_embed.task2vec_embed.models import get_model
from dataset_embed.task2vec_embed.task2vec import Task2Vec

sys.path.append('../../')
from enum import Enum
from util import dataset
class DatasetEmbeddingMethod(Enum):
    TASK2VEC = 1
    DOMAIN_SIMILARITY = 2

def embed_dataset_by_method(embeddingMethod: DatasetEmbeddingMethod, dataset_name: string, model_name: string):
    dataset_loaded = dataset.__dict__[dataset_name]('../../datasets/')[0]
    probe_network = get_model(model_name, pretrained=True, num_classes=int(max(dataset_loaded.targets)+1)).cuda()

    if embeddingMethod.__eq__(DatasetEmbeddingMethod.TASK2VEC):
        emb = Task2Vec(probe_network, max_samples=1000, skip_layers=6).embed(dataset_loaded)
    elif embeddingMethod.__eq__(DatasetEmbeddingMethod.DOMAIN_SIMILARITY):
        # TODO: Need to implement this similarly to Task2Vec
        raise RuntimeError("Need to extract this better")
    else:
        raise RuntimeError(f"Unsupported embedding method: '{embeddingMethod}'")

    print(emb)