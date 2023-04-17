# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.


import collections
import torchvision.transforms as transforms
import os
import json
from tfds import VTABIterableDataset


try:
    from IPython import embed
except:
    pass

_DATASETS = {}

Dataset = collections.namedtuple(
    'Dataset', ['trainset', 'testset'])


def _add_dataset(dataset_fn):
    _DATASETS[dataset_fn.__name__] = dataset_fn
    return dataset_fn


def _get_transforms(augment=True, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    basic_transform = [transforms.ToTensor(), normalize]

    transform_train = []
    if augment:
        transform_train += [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transform_train += [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    transform_train += basic_transform
    transform_train = transforms.Compose(transform_train)

    transform_test = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    transform_test += basic_transform
    transform_test = transforms.Compose(transform_test)

    return transform_train, transform_test

def _load_classnames(dataset_name, current_folder=os.path.dirname(__file__)):
    with open(os.path.join(current_folder, "en_classnames.json"), "r") as f:
        classnames = json.load(f)
    return classnames

def _get_torch_ds(tfds_dataset,transform,classes):
    train_ds =  VTABIterableDataset(
        tfds_dataset, 
        input_name="image", label_name="label", 
        transform=transform, 
        target_transform=int,
        split='train',
        classes=classes,
    )
    test_ds =  VTABIterableDataset(
        tfds_dataset, 
        input_name="image", label_name="label", 
        transform=transform, 
        target_transform=int,
        split='test',
        classes=classes,
    )
    return train_ds, test_ds

def _get_mnist_transforms(augment=True, invert=False, transpose=False):
    transform = [
        transforms.ToTensor(),
    ]
    if invert:
        transform += [transforms.Lambda(lambda x: 1. - x)]
    if transpose:
        transform += [transforms.Lambda(lambda x: x.transpose(2, 1))]
    transform += [
        transforms.Normalize((.5,), (.5,)),
        transforms.Lambda(lambda x: x.expand(3, 32, 32))
    ]

    transform_train = []
    transform_train += [transforms.Pad(padding=2)]
    if augment:
        transform_train += [transforms.RandomCrop(32, padding=4)]
    transform_train += transform
    transform_train = transforms.Compose(transform_train)

    transform_test = []
    transform_test += [transforms.Pad(padding=2)]
    transform_test += transform
    transform_test = transforms.Compose(transform_test)

    return transform_train, transform_test


def _get_cifar_transforms(augment=True):
    transform = [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
    transform_train = []
    if augment:
        transform_train += [
            transforms.Pad(padding=4, fill=(125, 123, 113)),
            transforms.RandomCrop(32, padding=0),
            transforms.RandomHorizontalFlip()]
    transform_train += transform
    transform_train = transforms.Compose(transform_train)
    transform_test = []
    transform_test += transform
    transform_test = transforms.Compose(transform_test)
    return transform_train, transform_test


def set_metadata(trainset, testset, config, dataset_name):
    trainset.metadata = {
        'dataset': dataset_name,
        'task_id': config.task_id,
        'task_name': trainset.task_name,
    }
    testset.metadata = {
        'dataset': dataset_name,
        'task_id': config.task_id,
        'task_name': testset.task_name,
    }
    return trainset, testset

@_add_dataset
def inat2018(root, config):
    from dataset.inat import iNat2018Dataset
    transform_train, transform_test = _get_transforms()
    trainset = iNat2018Dataset(root, split='train', transform=transform_train, task_id=config.task_id)
    testset = iNat2018Dataset(root, split='val', transform=transform_test, task_id=config.task_id)
    trainset, testset = set_metadata(trainset, testset, config, 'inat2018')
    return trainset, testset


def load_tasks_map(tasks_map_file):
    assert os.path.exists(tasks_map_file), tasks_map_file
    with open(tasks_map_file, 'r') as f:
        tasks_map = json.load(f)
        tasks_map = {int(k): int(v) for k, v in tasks_map.items()}
    return tasks_map


@_add_dataset
def cub_inat2018(root, config):
    """This meta-task is the concatenation of CUB-200 (first 25 tasks) and iNat (last 207 tasks).

    - The first 10 tasks are classification of the animal species inside one of 10 orders of birds in CUB-200
        (considering all orders except passeriformes).
    - The next 15 tasks are classification of species inside the 15 families of the order of passerifomes
    - The remaining 207 tasks are classification of the species inside each of 207 families in iNat

    As noted above, for CUB-200 10 taks are classification of species inside an order, rather than inside of a family
    as done in the iNat (recall order > family > species). This is done because CUB-200 has very few images
    in each family of bird (expect for the families of passeriformes). Hence, we go up step in the taxonomy and
    consider classification inside a orders and not families.
    """
    NUM_CUB = 25
    NUM_CUB_ORDERS = 10
    NUM_INAT = 207
    assert 0 <= config.task_id < NUM_CUB + NUM_INAT
    transform_train, transform_test = _get_transforms()
    if 0 <= config.task_id < NUM_CUB:
        # CUB
        from dataset.cub import CUBTasks, CUBDataset
        tasks_map_file = os.path.join(root, 'cub/CUB_200_2011', 'final_tasks_map.json')
        tasks_map = load_tasks_map(tasks_map_file)
        task_id = tasks_map[config.task_id]

        if config.task_id < NUM_CUB_ORDERS:
            # CUB orders
            train_tasks = CUBTasks(CUBDataset(root, split='train'))
            trainset = train_tasks.generate(task_id=task_id,
                                            use_species_names=True,
                                            transform=transform_train)
            test_tasks = CUBTasks(CUBDataset(root, split='test'))
            testset = test_tasks.generate(task_id=task_id,
                                          use_species_names=True,
                                          transform=transform_test)
        else:
            # CUB passeriformes families
            train_tasks = CUBTasks(CUBDataset(root, split='train'))
            trainset = train_tasks.generate(task_id=task_id,
                                            task='family',
                                            taxonomy_file='passeriformes.txt',
                                            use_species_names=True,
                                            transform=transform_train)
            test_tasks = CUBTasks(CUBDataset(root, split='test'))
            testset = test_tasks.generate(task_id=task_id,
                                          task='family',
                                          taxonomy_file='passeriformes.txt',
                                          use_species_names=True,
                                          transform=transform_test)
    else:
        # iNat2018
        from dataset.inat import iNat2018Dataset
        tasks_map_file = os.path.join(root, 'inat2018', 'final_tasks_map.json')
        tasks_map = load_tasks_map(tasks_map_file)
        task_id = tasks_map[config.task_id - NUM_CUB]

        trainset = iNat2018Dataset(root, split='train', transform=transform_train, task_id=task_id)
        testset = iNat2018Dataset(root, split='val', transform=transform_test, task_id=task_id)
    trainset, testset = set_metadata(trainset, testset, config, 'cub_inat2018')
    return trainset, testset


@_add_dataset
def imat2018fashion(root, config):
    NUM_IMAT = 228
    assert 0 <= config.task_id < NUM_IMAT
    from dataset.imat import iMat2018FashionDataset, iMat2018FashionTasks
    transform_train, transform_test = _get_transforms()
    train_tasks = iMat2018FashionTasks(iMat2018FashionDataset(root, split='train'))
    trainset = train_tasks.generate(task_id=config.task_id,
                                    transform=transform_train)
    test_tasks = iMat2018FashionTasks(iMat2018FashionDataset(root, split='validation'))
    testset = test_tasks.generate(task_id=config.task_id,
                                  transform=transform_test)
    trainset, testset = set_metadata(trainset, testset, config, 'imat2018fashion')
    return trainset, testset


@_add_dataset
def split_mnist(root, config):
    assert isinstance(config.task_id, tuple)
    from dataset.mnist import MNISTDataset, SplitMNISTTask
    transform_train, transform_test = _get_mnist_transforms()
    train_tasks = SplitMNISTTask(MNISTDataset(root, train=True))
    trainset = train_tasks.generate(classes=config.task_id, transform=transform_train)
    test_tasks = SplitMNISTTask(MNISTDataset(root, train=False))
    testset = test_tasks.generate(classes=config.task_id, transform=transform_test)
    trainset, testset = set_metadata(trainset, testset, config, 'split_mnist')
    return trainset, testset


@_add_dataset
def split_cifar(root, config):
    assert 0 <= config.task_id < 11
    from dataset.cifar import CIFAR10Dataset, CIFAR100Dataset, SplitCIFARTask
    transform_train, transform_test = _get_cifar_transforms()
    train_tasks = SplitCIFARTask(CIFAR10Dataset(root, train=True), CIFAR100Dataset(root, train=True))
    trainset = train_tasks.generate(task_id=config.task_id, transform=transform_train)
    test_tasks = SplitCIFARTask(CIFAR10Dataset(root, train=False), CIFAR100Dataset(root, train=False))
    testset = test_tasks.generate(task_id=config.task_id, transform=transform_test)
    trainset, testset = set_metadata(trainset, testset, config, 'split_cifar')
    return trainset, testset


@_add_dataset
def cifar10_mnist(root, config):
    from dataset.cifar import CIFAR10Dataset
    from dataset.mnist import MNISTDataset
    from dataset.expansion import UnionClassificationTaskExpander
    transform_train, transform_test = _get_cifar_transforms()
    trainset = UnionClassificationTaskExpander(merge_duplicate_images=False)(
        [CIFAR10Dataset(root, train=True), MNISTDataset(root, train=True, expand=True)], transform=transform_train)
    testset = UnionClassificationTaskExpander(merge_duplicate_images=False)(
        [CIFAR10Dataset(root, train=False), MNISTDataset(root, train=False, expand=True)], transform=transform_test)
    return trainset, testset

@_add_dataset
def cars(root):
    from torchvision.datasets import StanfordCars
    transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )
    train_dataset = StanfordCars(root=root, split="train", transform=transform, download=True, **kwargs)
    test_dataset = StanfordCars(root=root, split="test", transform=transform, download=True, **kwargs)
    return train_dataset, test_dataset

@_add_dataset
def dtd(root):
    from task_adaptation.data.dtd import DTDData
    tfds_dataset = DTDData(data_dir=root)
    classes = _load_classnames("dtd")
    transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )
    return _get_torch_ds(tfds_dataset,transform=transform,classes=classes)

@_add_dataset
def eurosat(root):
    from torchvision.datasets import EuroSAT
    transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )
    train_dataset = EuroSAT(root=root, split="train", transform=transform, download=True, **kwargs)
    test_dataset = EuroSAT(root=root, split="test", transform=transform, download=True, **kwargs)
    train_dataset.classes = _load_classnames("eurosat")
    test_dataset.classes = _load_classnames("eurosat")
    return train_dataset, test_dataset

@_add_dataset
def flowers(root):
    from task_adaptation.data.oxford_flowers102 import OxfordFlowers102Data
    transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )
    tfds_dataset = OxfordFlowers102Data(data_dir=root)
    return _get_torch_ds(tfds_dataset,transform=transform,classes=_load_classnames('flowers'))

@_add_dataset
def pets(root):
    from task_adaptation.data.oxford_iiit_pet import OxfordIIITPetData
    transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )
    tfds_dataset = OxfordIIITPetData(data_dir=root)
    return _get_torch_ds(tfds_dataset,transform=transform,classes=_load_classnames('pets'))


@_add_dataset
def dmlab(root):
    from task_adaptation.data.dmlab import DmlabData
    transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )
    download_tfds_dataset("dmlab", data_dir=root) # it's not called in the original VTAB code, so we do it explictly
    tfds_dataset = DmlabData(data_dir=root)
    return _get_torch_ds(tfds_dataset,transform=transform,classes=_load_classnames('dmlab'))

@_add_dataset
def clevr(root,task):
    transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )
    from task_adaptation.data.clevr import CLEVRData
    # task = _extract_task(dataset_name)
    assert task in ("count_all", "closest_object_distance")
    tfds_dataset = CLEVRData(task=task, data_dir=root)
    if task == "count_all":
        classes = _load_classnames("clevr_count_all")
    elif task == "closest_object_distance":
        classes = _load_classnames("clevr_closest_object_distance")
    else:
        raise ValueError(f"non supported: {task}")
    return _get_torch_ds(tfds_dataset,transform=transform,classes=classes)

@_add_dataset
def caltech101(root):
    from torchvision.datasets import Caltech101
    transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )
    train_dataset = StanfordCars(root=root, split="train", transform=transform, download=True, **kwargs)
    test_dataset = StanfordCars(root=root, split="test", transform=transform, download=True, **kwargs)
    return train_dataset, test_dataset

@_add_dataset
def imagenet(root):
    from dataset.imagenet import ImageNetKaggle
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    val_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    dataset = ImageNetKaggle(root, "val", val_transform)
    # trainset = CIFAR10(root, train=True, transform=transform, download=True)
    # testset = CIFAR10(root, train=False, transform=transform)
    return dataset, dataset


@_add_dataset
def cifar10(root):
    from torchvision.datasets import CIFAR10
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = CIFAR10(root, train=True, transform=transform, download=True)
    testset = CIFAR10(root, train=False, transform=transform)
    return trainset, testset


@_add_dataset
def cifar100(root):
    from torchvision.datasets import CIFAR100
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = CIFAR100(root, train=True, transform=transform, download=True)
    testset = CIFAR100(root, train=False, transform=transform)
    return trainset, testset

@_add_dataset
def pcam(root):
    from torchvision.datasets import PCAM
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset =  PCAM(root=root, split="train", transform=transform, download=True, **kwargs)
    testset =  PCAM(root=root, split="test", transform=transform, download=True, **kwargs)
    classes = _load_classnames("pcam")
    trainset.classes = classes
    testset.classes = classes
    return trainset, testset


@_add_dataset
def sun397(root):
    from torchvision.datasets import SUN397
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = SUN397(root, train=True, transform=transform, download=True)
    testset = SUN397(root, train=False, transform=transform)
    # ds = SUN397(root=root, transform=transform, download=True, **kwargs)
    # trainset.classes = [cl.replace("_", " ").replace("/", " ") for cl in ds.classes]
    return trainset, testset




@_add_dataset
def mnist(root):
    from torchvision.datasets import MNIST
    transform = transforms.Compose([
        lambda x: x.convert("RGB"),
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
    ])
    trainset = MNIST(root, train=True, transform=transform, download=True)
    testset = MNIST(root, train=False, transform=transform)
    return trainset, testset


@_add_dataset
def letters(root):
    from torchvision.datasets import EMNIST
    transform = transforms.Compose([
        lambda x: x.convert("RGB"),
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
    ])
    trainset = EMNIST(root, train=True, split='letters', transform=transform, download=True)
    testset = EMNIST(root, train=False, split='letters', transform=transform)
    return trainset, testset

@_add_dataset
def food101(root):
    from torchvision.datasets import Food101
    transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    trainset = Food101(root, split='train', transform=transform, download=True)
    testset = Food101(root, split='test', transform=transform)
    trainset.targets = trainset._labels
    testset.targets = testset._labels
    return trainset, testset

@_add_dataset
def kmnist(root):
    from torchvision.datasets import KMNIST
    transform = transforms.Compose([
        lambda x: x.convert("RGB"),
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    trainset = KMNIST(root, train=True, transform=transform, download=True)
    testset = KMNIST(root, train=False, transform=transform)
    return trainset, testset


@_add_dataset
def stl10(root):
    from torchvision.datasets import STL10
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = STL10(root, split='train', transform=transform, download=True)
    testset = STL10(root, split='test', transform=transform)
    trainset.targets = trainset.labels
    testset.targets = testset.labels
    return trainset, testset


def get_dataset(root, config=None):
    return _DATASETS[config.name](os.path.expanduser(root), config)


