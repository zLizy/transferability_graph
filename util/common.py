from abc import abstractmethod
from enum import Enum
from typing import Optional

import numpy as np
from datasets import load_dataset, Dataset as HuggingFaceDataset
import torch
from torch import tensor
from torch.utils.data import TensorDataset as TorchTensorDataset, TensorDataset
from torchvision.transforms import transforms
from transformers import DistilBertTokenizer


class GransferEnumDatasetSource(Enum):
    HUGGINGFACE = 1


class GransferEnumDatasetType(Enum):
    IMAGE = 1
    TEXT = 2
    TABULAR = 3


# Draft catchy name for the project, so we can identify our classes
class GransferDataset:

    def __init__(
            self,
            identifier: str,
            dataset_source: GransferEnumDatasetSource,
            dataset_type: GransferEnumDatasetType,
            train_set: TorchTensorDataset,
            test_set: Optional[TorchTensorDataset] = None,
    ) -> None:
        self.identifier = identifier
        self.dataset_source = dataset_source
        self.dataset_type = dataset_type
        self.train_set = train_set
        self.test_set = test_set

    @staticmethod
    @abstractmethod
    def load(identifier: str):
        pass


class GransferHuggingFaceImageDataset(GransferDataset):
    @staticmethod
    def load(identifier: str) -> GransferDataset:
        train_dataset_huggingface = load_dataset(identifier, name='full')['train']
        train_dataset = GransferHuggingFaceImageDataset.hfds2tvds(train_dataset_huggingface)

        return GransferHuggingFaceImageDataset(
            identifier=identifier,
            dataset_source=GransferEnumDatasetSource.HUGGINGFACE,
            dataset_type=GransferEnumDatasetType.IMAGE,
            train_set=train_dataset
        )

    @staticmethod
    def hfds2tvds(dataset_huggingface: HuggingFaceDataset, input_shape=224):
        print(dataset_huggingface.features)
        try:
            labels = dataset_huggingface['labels']
        except:
            print('labels not in the columns')
            labels = dataset_huggingface['label']
        transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((input_shape, input_shape)),
            transforms.ToTensor(),
        ])
        images = [transform(image.convert('RGB')) for image in dataset_huggingface['image']]
        print(f'type(images):{type(images)},type(images[0]): {type(images[0])}, type(labels):{type(labels)}')
        # ds.set_transform(hf_transform)
        try:
            classes = dataset_huggingface.features.copy()['labels'].names
        except:
            print('labels not in the columns')
            classes = dataset_huggingface.features.copy()['label'].names
        print(f'classes: {classes}')
        from torch.utils.data import TensorDataset
        dataset_torch = TensorDataset(torch.stack(images), torch.from_numpy(np.stack(labels).reshape(len(labels))))
        dataset_torch.classes = classes

        return dataset_torch


class GransferHuggingFaceTextDataset(GransferDataset):
    @staticmethod
    def hftxtds2tvds(dataset_huggingface: HuggingFaceDataset) -> TorchTensorDataset:
        print(dataset_huggingface.features)
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        text = tokenizer(dataset_huggingface["text"], return_tensors="pt", padding=True)["input_ids"]
        labels = dataset_huggingface["coarse_label"]
        dataset_torch = TensorDataset(text, tensor(labels))

        return dataset_torch

    @staticmethod
    def load(identifier: str) -> GransferDataset:
        train_dataset_huggingface = load_dataset(identifier, name='full')['train']
        train_dataset_torch = GransferHuggingFaceTextDataset.hftxtds2tvds(train_dataset_huggingface)
        test_dataset_huggingface = load_dataset(identifier, split='test')
        test_dataset_torch = GransferHuggingFaceTextDataset.hftxtds2tvds(test_dataset_huggingface)

        return GransferHuggingFaceTextDataset(
            identifier=identifier,
            dataset_source=GransferEnumDatasetSource.HUGGINGFACE,
            dataset_type=GransferEnumDatasetType.IMAGE,
            train_set=train_dataset_torch,
            test_set=test_dataset_torch
        )