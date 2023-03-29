import torch
from torch import Tensor
print(torch.__version__)
import os
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.data import download_url, extract_zip

import tqdm
import pandas as pd
import torch.nn.functional as F

from graph import Graph 

def construct():
    graph = Graph()


if __name__ == '__main__':
    construct()