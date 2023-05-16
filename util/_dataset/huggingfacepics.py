import time

import dateutil.utils
import requests
import math
import matplotlib.pyplot as plt
import shutil
from getpass import getpass
from PIL import Image, UnidentifiedImageError
from requests.exceptions import HTTPError
from io import BytesIO
from pathlib import Path
import torch
# import pytorch_lightning as pl
from huggingface_hub import HfApi, HfFolder, Repository, notebook_login
from torch.utils.data import DataLoader
# from torchmetrics import Accuracy
from torchvision.datasets import ImageFolder
from transformers import ViTFeatureExtractor, ViTForImageClassification

SEARCH_URL = "https://huggingface.co/api/experimental/images/search"


def get_image_urls_by_term(search_term: str, count=150):
    params = {"q": search_term, "license": "public", "imageType": "photo", "count": count}
    response = requests.get(SEARCH_URL, params=params)
    response.raise_for_status()
    response_data = response.json()
    image_urls = [img['thumbnailUrl'] for img in response_data['value']]
    return image_urls


def gen_images_from_urls(urls):
    num_skipped = 0
    for url in urls:
        response = requests.get(url)
        if not response.status_code == 200:
            num_skipped += 1
        try:
            img = Image.open(BytesIO(response.content))
            yield img
        except UnidentifiedImageError:
            num_skipped += 1

    print(f"Retrieved {len(urls) - num_skipped} images. Skipped {num_skipped}.")


def urls_to_image_folder(urls, save_directory):
    for i, image in enumerate(gen_images_from_urls(urls)):
        image.save(save_directory / f'{i}.jpg')


def get_huggingfacepics_data_set_by_all_search_term(root,all_search_term,transform=None):
    all_search_term = all_search_term
    data_dir = Path(
        f'{root}/hfpics/' + all_search_term.__str__())
        # '../../datasets/huggingfacepics/' + dateutil.utils.today().strftime('%Y-%m-%d') + all_search_term.__str__())

    if data_dir.exists():
        print("Already searched this huggingfacepics keyword combination today, using that.")
    else:
        for search_term in all_search_term:
            search_term_dir = data_dir / search_term
            search_term_dir.mkdir(exist_ok=True, parents=True)
            urls = get_image_urls_by_term(search_term)
            print(f"Saving images of {search_term} to {str(search_term_dir)}...")
            urls_to_image_folder(urls, search_term_dir)

    dataset = ImageFolder(data_dir,transform=transform)
    dataset.classes = all_search_term
    # indices = torch.randperm(len(dataset)).tolist()
    # n_val = math.floor(len(indices) * .15)
    # train_dataset = torch.utils.data.Subset(dataset, indices[:-n_val])
    # test_dataset = torch.utils.data.Subset(dataset, indices[-n_val:])
    # print(dataset)

    return dataset, dataset,'hfpics'