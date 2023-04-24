import os
import time

import requests
import math
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from pathlib import Path
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder

from util.config import dataset_collection

SEARCH_URL = "https://huggingface.co/api/experimental/images/search"


def get_image_urls_by_term(search_term: str, count=1500):
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


def get_huggingfacepics_data_set_by_all_search_term(root, all_search_term, transform=None):
    all_data_set = []

    for search_term in all_search_term:
        data_dir_string = f'{root}/hfpics/' + search_term
        search_term_dir = Path(data_dir_string + "/" + search_term)

        if search_term_dir.exists():
            print("Reusing saved search term directory: '" + search_term + "', skipping")
        else:
            search_term_dir.mkdir(exist_ok=True, parents=True)
            urls = get_image_urls_by_term(search_term)
            print(f"Saving images of {search_term} to {str(search_term_dir)}...")
            urls_to_image_folder(urls, search_term_dir)

        if len(os.listdir(search_term_dir)) == 0:
            print(f"No images for {search_term}, skipping...")
        else:
            all_data_set.append(ImageFolder(data_dir_string, transform=transform))

    print(f"Combining {'-'.join(all_search_term)} into one dataset.")
    dataset = ConcatDataset(all_data_set)
    indices = torch.randperm(len(dataset)).tolist()
    n_val = math.floor(len(indices) * .15)
    train_dataset = torch.utils.data.Subset(dataset, indices[:-n_val])
    test_dataset = torch.utils.data.Subset(dataset, indices[-n_val:])

    return train_dataset, test_dataset

