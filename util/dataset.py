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
import json
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from datasets import Dataset as HuggingFaceDataset
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from torch import tensor
from torch.utils.data import DataLoader

from util.common import GransferHuggingFaceTextDataset, GransferHuggingFaceImageDataset
from util.tfds import VTABIterableDataset

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


def _get_transforms(input_shape,augment=False, normalize=None):
    if normalize is None:
        basic_transform = [transforms.ToTensor()]
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        basic_transform = [transforms.ToTensor(), normalize]

    transform_train = []
    if augment:
        transform_train += [
            transforms.RandomResizedCrop(input_shape),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transform_train += [
            # transforms.Resize(256),
            transforms.CenterCrop(input_shape),
        ]
    transform_train += basic_transform
    transform_train = transforms.Compose(transform_train)

    transform_test = [
        # transforms.Resize(256),
        transforms.CenterCrop(input_shape),
    ]
    transform_test += basic_transform
    transform_test = transforms.Compose(transform_test)

    return transform_train, transform_test

def _extract_task(dataset_name):
    prefix, *task_name_list = dataset_name.split("_")
    task = "_".join(task_name_list)
    return task

def _load_classnames(current_folder=os.path.dirname(__file__)):
    with open(os.path.join(current_folder, "en_classnames.json"), "r") as f:
        classnames = json.load(f)
    return classnames

def _get_torch_ds(tfds_dataset,transform,classes,batch_size=1):
    # print(f"tfds_dataset.size: {tfds_dataset.get_num_samples('train')}")
    train_ds =  VTABIterableDataset(
        tfds_dataset, 
        input_name="image", label_name="label", 
        transform=transform, 
        target_transform=int,
        split='train',
        classes=classes,
        batch_size=batch_size,
    )
    test_ds =  VTABIterableDataset(
        tfds_dataset, 
        input_name="image", label_name="label", 
        transform=transform, 
        target_transform=int,
        split='test',
        classes=classes,
        batch_size=batch_size,
    )
    return train_ds, test_ds

def tfds2tvds(ds,input_shape):
        ## Convert tfds to torchvision dataset
        print('is tfds dataset type')
        batch_size = 1
        transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((input_shape,input_shape)),
                            transforms.ToTensor(),
                        ])
        iterator = ds.get_tf_data('train', batch_size=1, epochs=1, for_eval=True)# self.batch_size = 1
        inputs = []
        labels = []
        for data in iterator:
            image = data['image'].numpy()
            shape = image.shape
            image = np.reshape(image,(shape[1],shape[2],shape[3]))
            inputs.append(transform(image))
            labels.append(data['label'].numpy())
        # Generate torchvision dataset
        from torch.utils.data import TensorDataset
        ds = TensorDataset(torch.stack(inputs),torch.from_numpy(np.stack(labels).reshape(len(labels))))
        print(f'tf_data: {type(ds)}')
        torch.cuda.empty_cache()
        return ds

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


def _get_cifar_transforms(input_shape,augment=True):
    transform = [
        transforms.Resize((input_shape,input_shape)),
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
def age_prediction(root,input_shape):
    # https://www.aicrowd.com/showcase/solution-for-submission-175171
    from dataset.age import DatasetAge
    IMAGE_PATH = ...
    TRAIN_CSV_PATH = ...
    TEST_CSV_PATH = ...
    transforms, _ = _get_transforms(input_shape,normalize=True)
    train_dataset = DatasetAge(csv_path=TRAIN_CSV_PATH,
                               img_dir=IMAGE_PATH,
                               split="train",
                               transform=transforms)
    test_dataset = DatasetAge(csv_path=TEST_CSV_PATH,
                               img_dir=IMAGE_PATH,
                               split="test",
                               transform=transforms)

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
    transform_train, transform_test = _get_cifar_transforms(input_shape)
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
    return trainset, testset, 'tvds'

@_add_dataset
def stanfordcars(root,input_shape=224):
    from torchvision.datasets import StanfordCars
    transform, _ = _get_transforms(input_shape,augment=False)
    train_dataset = StanfordCars(root=root, split="train", transform=transform, download=True)
    test_dataset = StanfordCars(root=root, split="test", transform=transform, download=True)
    return train_dataset, test_dataset, 'tvds'

@_add_dataset
def dtd(root,input_shape=224):
    from task_adaptation.data.dtd import DTDData
    tfds_dataset = DTDData(data_dir=root)
    classes = tfds_dataset._dataset_builder.info.features['label'].names
    tfds_dataset = tfds2tvds(tfds_dataset,input_shape)
    tfds_dataset.classes = classes
    return tfds_dataset, tfds_dataset, 'tfds'
    # return _get_torch_ds(tfds_dataset,transform=transform,classes=classes)

@_add_dataset
def eurosat(root,input_shape=224):
    # from torchvision.datasets import EuroSAT
    # transform, _ = _get_transforms(augment=False)
    # train_dataset = EuroSAT(root=root, transform=transform, download=True)
    # # test_dataset = EuroSAT(root=root,transform=transform, download=True)
    # train_dataset.classes = _load_classnames("eurosat")
    # # test_dataset.classes = _load_classnames("eurosat")
    # return train_dataset, '', 'tvds'
    from task_adaptation.data.eurosat import EurosatData
    tfds_dataset = EurosatData(subset="rgb", data_key="image", data_dir=root)
    tfds_dataset = tfds2tvds(tfds_dataset,input_shape)
    classes = _load_classnames()["eurosat"]
    tfds_dataset.classes = classes
    return tfds_dataset, '', 'tfds'

@_add_dataset
def flowers(root,input_shape=224):
    return oxford_flowers102(root,input_shape)

@_add_dataset
def oxford_flowers102(root,input_shape=224):
    from task_adaptation.data.oxford_flowers102 import OxfordFlowers102Data
    transform, _ = _get_transforms(augment=False,input_shape=input_shape)
    tfds_dataset = OxfordFlowers102Data(data_dir=root)
    tfds_dataset = tfds2tvds(tfds_dataset,input_shape)
    classes = _load_classnames()["flowers"]
    tfds_dataset.classes = classes
    return tfds_dataset, '', 'tfds'

@_add_dataset
def pets(root,input_shape=224):
    return oxford_iiit_pet(root,input_shape)

@_add_dataset
def oxford_iiit_pet(root,input_shape=224):
    from task_adaptation.data.oxford_iiit_pet import OxfordIIITPetData
    # transform, _ = _get_transforms(input_shape,augment=False)
    tfds_dataset = OxfordIIITPetData(data_dir=root)
    tfds_dataset = tfds2tvds(tfds_dataset,input_shape)
    classes = _load_classnames()["pets"]
    tfds_dataset.classes = classes
    return tfds_dataset, '', 'tfds'
    # return _get_torch_ds(tfds_dataset,transform=transform,classes=_load_classnames('pets'))


@_add_dataset
def dmlab(root,input_shape=224):
    from task_adaptation.data.dmlab import DmlabData
    transform, _ = _get_transforms(input_shape,augment=False)
    from util.tfds import download_tfds_dataset
    download_tfds_dataset("dmlab", data_dir=root) # it's not called in the original VTAB code, so we do it explictly
    tfds_dataset = DmlabData(data_dir=root)
    tfds_dataset = tfds2tvds(tfds_dataset,input_shape)
    classes = _load_classnames()["dmlab"]
    tfds_dataset.classes = classes
    return tfds_dataset, tfds_dataset, 'tfds'
    # return _get_torch_ds(tfds_dataset,transform=transform,classes=_load_classnames('dmlab'))

@_add_dataset
def resisc45(root,input_shape=224):
    # Needs download from OneDrive: https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs
    # The archive needs to to be put at <DATASET_ROOT>/downloads/manual then extracted
    if not os.path.exists(root):
        os.makedirs(os.path.join(data_dir, "downloads", "manual"))
        call(f"wget 'https://onedrive.live.com/download?resid=5C5E061130630A68!107&authkey=!AHHNaHIlzp_IXjs' --output-document={root}/downloads/manual/resisc45.rar", shell=True)
        call(f"cd {data_dir}/downloads/manual;unrar x resisc45.rar", shell=True)
    from task_adaptation.data.resisc45 import Resisc45Data
    tfds_dataset = Resisc45Data(data_dir=root)
    classes = tfds_dataset._dataset_builder.info.features['label'].names
    tfds_dataset = tfds2tvds(tfds_dataset,input_shape)
    tfds_dataset.classes = classes
    return tfds_dataset, tfds_dataset, 'tfds'

@_add_dataset
def clevr_count(root,input_shape=224):
    return clevr_count_all(root,input_shape)

@_add_dataset
def clevr_count_all(root,input_shape=224):
    task = _extract_task('clevr_count_all')
    return clevr(root,task,input_shape)

@_add_dataset
def clevr_closest_object_distance(root,input_shape=224):
    return clevr_distance(root,input_shape)

@_add_dataset
def clevr_distance(root,input_shape=224):
    task = _extract_task('clevr_closest_object_distance')
    return clevr(root,task,input_shape)


def clevr(root,task,input_shape=224):
    transform, _ = _get_transforms(input_shape,augment=False)
    from task_adaptation.data.clevr import CLEVRData
    # task = _extract_task(dataset_name)
    assert task in ("count_all", "closest_object_distance")
    tfds_dataset = CLEVRData(task=task, data_dir=root)
    if task == "count_all":
        classes = _load_classnames()["clevr_count_all"]
    elif task == "closest_object_distance":
        classes = _load_classnames()["clevr_closest_object_distance"]
    else:
        raise ValueError(f"non supported: {task}")
    tfds_dataset = tfds2tvds(tfds_dataset,input_shape)
    # import random
    # random.seed(10)
    # idx = random.sample(range(len(tfds_dataset)), k=10000)
    # # randomly sample MAX_NUM_SAMPLES
    # tfds_dataset= torch.utils.data.Subset(tfds_dataset, idx)
    tfds_dataset.classes = tfds_dataset 
    return tfds_dataset, tfds_dataset, 'tfds'
    # return _get_torch_ds(tfds_dataset,transform=transform,classes=classes)

@_add_dataset
def svhn(root,input_shape=224):
    from task_adaptation.data.svhn import SvhnData
    tfds_dataset = SvhnData(data_dir=root)
    classes =_load_classnames()["svhn"]
    tfds_dataset = tfds2tvds(tfds_dataset,input_shape)
    tfds_dataset.classes = classes
    # transform, _ = _get_transforms(augment=False)
    return tfds_dataset, '', 'tfds'
    # return _get_torch_ds(tfds_dataset,transform=transform,classes=classes)

@_add_dataset
def smallnorb_label_azimuth(root,input_shape=224):
    task = _extract_task('smallnorb_label_azimuth')
    return smallnorb(root,task,input_shape)

def smallnorb_label_elevation(root,input_shape=224):
    task = _extract_task('smallnorb_label_elevation')
    return smallnorb(root,task,input_shape)

def smallnorb(root,task,input_shape,transform=None):
    from task_adaptation.data.smallnorb import SmallNORBData
    assert task in ("label_category", "label_elevation", "label_azimuth", "label_lighting")
    tfds_dataset = SmallNORBData(predicted_attribute=task, data_dir=root)
    classes = tfds_dataset._dataset_builder.info.features[task].names
    # transform, _ = _get_transforms(augment=False)
    tfds_dataset = tfds2tvds(tfds_dataset,input_shape)
    tfds_dataset.classes = classes
    return tfds_dataset, '', 'tfds'
    # return _get_torch_ds(tfds_dataset,transform=transform,classes=classes)

@_add_dataset
def dsprites_label_x_position(root,input_shape=224):
    task = _extract_task('dsprites_label_x_position')
    return dsprites(root,task,input_shape)

@_add_dataset
def dsprites_label_orientation(root,input_shape=224):
    task = _extract_task('dsprites_label_orientation')
    return dsprites(root,task,input_shape)

def dsprites(root,task,input_shape,transform=None):
    from task_adaptation.data.dsprites import DSpritesData
    assert task in ("label_shape", "label_scale", "label_orientation", "label_x_position", "label_y_position")
    tfds_dataset = DSpritesData(task, data_dir=root)
    classes = tfds_dataset._dataset_builder.info.features[task].names
    # transform, _ = _get_transforms(augment=False)
    tfds_dataset = tfds2tvds(tfds_dataset,input_shape)
    tfds_dataset.classes = classes
    return tfds_dataset, tfds_dataset, 'tfds'
    # return _get_torch_ds(tfds_dataset,transform=transform,classes=classes)

@_add_dataset
def kitti(root,input_shape=224):
    # from util._dataset.kitti import KittiData
    task = _extract_task('kitti_closest_vehicle_distance')
    return _kitti(root,task,input_shape)
    # assert task in (
    #         "count_all", "count_left", "count_far", "count_near", 
    #         "closest_object_distance", "closest_object_x_location", 
    #         "count_vehicles", "closest_vehicle_distance",
    #     )
    # tfds_dataset = KittiData(task=task, data_dir=root)
    # if task == "closest_vehicle_distance":
    #     classes = _load_classnames["kitti_closest_vehicle_distance"]
    # else:
    #     raise ValueError(f"Unsupported task: {task}")
    # tfds_dataset.classes = classes
    # return tfds_dataset, '', 'tfds'

def _kitti(root,task,input_shape=224):
    from util._dataset.kitti import KittiData
    
    assert task in (
            "count_all", "count_left", "count_far", "count_near", 
            "closest_object_distance", "closest_object_x_location", 
            "count_vehicles", "closest_vehicle_distance",
        )
    tfds_dataset = KittiData(task=task, data_dir=root)
    if task == "closest_vehicle_distance":
        classes = _load_classnames()["kitti_closest_vehicle_distance"]
    else:
        raise ValueError(f"Unsupported task: {task}")
    # transform, _ = _get_transforms(augment=False)
    tfds_dataset = tfds2tvds(tfds_dataset,input_shape)
    tfds_dataset.classes = classes
    return tfds_dataset, '', 'tfds'
    # return _get_torch_ds(tfds_dataset,transform=transform,classes=classes)
    
@_add_dataset
def caltech101(root,input_shape=224):
    from task_adaptation.data.caltech import Caltech101
    tfds_dataset = Caltech101(data_dir=root)
    classes = _load_classnames()["caltech101_vtab"]
    # transform, _ = _get_transforms(augment=False)
    tfds_dataset = tfds2tvds(tfds_dataset,input_shape)
    tfds_dataset.classes = classes
    return tfds_dataset, tfds_dataset, 'tfds'
    # tfds_dataset = tfds_dataset.get_tf_data('train', batch_size=64, epochs=1, for_eval=True)
    # return tfds_dataset._dataset_builder.as_dataset(split='train'),tfds_dataset._dataset_builder.as_dataset(split='test'),'tfds'
    # return _get_torch_ds(tfds_dataset,transform=transform,classes=classes)
    # train_set = tfds.as_numpy(tfds_dataset.batch(batch_size))
    # tfds_dataset.get_tf_data('train', batch_size=self.batch_size, epochs=1, for_eval=True)
    # return train_set, train_set

@_add_dataset
def imagenet(root,input_shape=224):
    from util._dataset.imagenet import ImageNetKaggle
    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)
    transform, _ = _get_transforms(input_shape,augment=False,normalize=True)
    dataset = ImageNetKaggle(root, "val", transform)
    dataset.classes = _load_classnames()['imagenet1k']
    # trainset = CIFAR10(root, train=True, transform=transform, download=True)
    # testset = CIFAR10(root, train=False, transform=transform)
    return dataset, '', 'imagenet'


@_add_dataset
def cifar10(root,input_shape=224):
    from torchvision.datasets import CIFAR10
    transform = transforms.Compose([
        transforms.Resize((input_shape,input_shape)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = CIFAR10(root, train=True, transform=transform, download=True)
    testset = CIFAR10(root, train=False, transform=transform)
    return trainset, testset, 'tvds'


@_add_dataset
def cifar100(root,input_shape=224):
    from torchvision.datasets import CIFAR100
    transform = transforms.Compose([
        transforms.Resize((input_shape,input_shape)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = CIFAR100(root, train=True, transform=transform, download=True)
    return trainset, '', 'tvds'

@_add_dataset
def pcam(root,input_shape=224):
    from torchvision.datasets import PCAM
    transform = transforms.Compose([
        transforms.Resize((input_shape,input_shape)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset =  PCAM(root=root, split="train", transform=transform, download=True)
    testset =  PCAM(root=root, split="test", transform=transform, download=True)
    classes = _load_classnames()["pcam"]
    trainset.classes = classes
    testset.classes = classes
    return trainset, testset, 'tvds'


@_add_dataset
def sun397(root,input_shape=224):
    # from task_adaptation.data.sun397 import Sun397Data
    # tfds_dataset = Sun397Data(config="tfds", data_dir=root)
    from torchvision.datasets import SUN397
    tfds_dataset = SUN397(root=root, transform=_get_transforms(input_shape,augment=False,normalize=True)[0], download=True)
    tfds_dataset.classes = [cl.replace("_", " ").replace("/", " ") for cl in tfds_dataset.classes]
    return tfds_dataset, '', 'tfds'


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
    trainset.classes = _load_classnames()['mnist']
    return trainset, testset, 'tvds'


@_add_dataset
def letters(root,input_shape=224):
    from torchvision.datasets import EMNIST
    transform = transforms.Compose([
        lambda x: x.convert("RGB"),
        transforms.Resize(input_shape),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
    ])
    trainset = EMNIST(root, train=True, split='letters', transform=transform, download=True)
    testset = EMNIST(root, train=False, split='letters', transform=transform)
    return trainset, testset, 'tvds'

@_add_dataset
def food101(root,input_shape=224):
    from torchvision.datasets import Food101
    transform, _ = _get_transforms(input_shape,augment=False,normalize=True)
    trainset = Food101(root, split='train', transform=transform, download=True)
    testset = Food101(root, split='test', transform=transform)
    # self.classes exist
    return trainset, testset, 'tvds'

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
    return trainset, testset, 'tvds'


@_add_dataset
def stl10(root,input_shape=224):
    from torchvision.datasets import STL10
    transform = transforms.Compose([
        transforms.Resize((input_shape,input_shape)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = STL10(root, split='train', transform=transform, download=True)
    testset = STL10(root, split='test', transform=transform)
    trainset.targets = trainset.labels
    testset.targets = testset.labels
    return trainset, testset, 'tvds'

@_add_dataset
def diabetic_retinopathy_detection(root,input_shape=224):
    # Needs manual download from Kaggle
    # 1) `kaggle competitions download -c diabetic-retinopathy-detection` on $ROOT/downloads/manual
    # 2) extract archives  on $ROOT/downloads/manual
    if not os.path.exists(root):
        # Automatic download
        print("Downloading diabetic_retinopathy...")
        if not has_kaggle():
            print("Kaggle is needed to download the dataset. Please install it via `pip install kaggle`")
            sys.exit(1)
        os.makedirs(os.path.join(data_dir, "downloads", "manual"))
        call(f"kaggle competitions download -c diabetic-retinopathy-detection -p {root}/downloads/manual", shell=True)
        call(f"cd {root}/downloads/manual;unzip diabetic-retinopathy-detection.zip;cat train.zip*>train.zip;cat test.zip*>test.zip;unzip train.zip; unzip test.zip;unzip sample.zip;unzip trainLabels.csv.zip", shell=True)
        
    from task_adaptation.data.diabetic_retinopathy import RetinopathyData
    tfds_dataset = RetinopathyData(config="btgraham-300", data_dir=root)
    classes = _load_classnames()["diabetic_retinopathy"]
    tfds_dataset = tfds2tvds(tfds_dataset,input_shape)
    tfds_dataset.classes = classes
    # transform, _ = _get_transforms(augment=False)
    return tfds_dataset, '', 'tfds'
    # return _get_torch_ds(tfds_dataset,transform=transform,classes=classes)

@_add_dataset
def huggingfacepics_corgi(root):
    from util._dataset.huggingfacepics import get_huggingfacepics_data_set_by_all_search_term

    return get_huggingfacepics_data_set_by_all_search_term(["corgi"])

def _get_dataset(root, name,input_shape):
    return _DATASETS[name](root, input_shape)

#########################
# HuggingFace Pics

def _get_transform(input_shape=224):
    transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(input_shape),
    transforms.Resize((input_shape,input_shape)),
    transforms.ToTensor(),
    transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
    ])
    return transform

@_add_dataset
def hfpics(root,classes,input_shape=224):
    from util._dataset.huggingfacepics import get_huggingfacepics_data_set_by_all_search_term
    return get_huggingfacepics_data_set_by_all_search_term(root,classes,transform=_get_transform(input_shape))

#### HuggingFace Datasets

def hfds2tvds(ds,input_shape=224):
    print(ds.features)
    try:
        labels = ds['labels']
    except:
        print('labels not in the columns')
        labels = ds['label']
    transform = transforms.Compose([
                            # transforms.ToPILImage(),
                            transforms.Resize((input_shape,input_shape)),
                            transforms.ToTensor(),
                        ])
    images = [transform(image.convert('RGB')) for image in ds['image']]
    print(f'type(images):{type(images)},type(images[0]): {type(images[0])}, type(labels):{type(labels)}')
    # ds.set_transform(hf_transform)
    try:
        classes = ds.features.copy()['labels'].names
    except:
        print('labels not in the columns')
        classes = ds.features.copy()['label'].names
    print(f'classes: {classes}')
    from torch.utils.data import TensorDataset
    ds = TensorDataset(torch.stack(images),torch.from_numpy(np.stack(labels).reshape(len(labels))))
    ds.classes = classes
    return ds

@_add_dataset
def cats_vs_dogs(root,input_shape=224):
    train_dataset = load_dataset('cats_vs_dogs', split='train')
    # train_dataset.set_format(type='torch', columns=['image', 'labels'])
    train_dataset = hfds2tvds(train_dataset,input_shape)    
    # test_dataset = load_dataset('cats_vs_dogs', split='test')
    return train_dataset, '','hfds'


@_add_dataset
def beans(root,input_shape=224):
    train_dataset = load_dataset('beans', split='train')
    # train_dataset.set_format(type='torch', columns=['image_file_path','image','labels'])
    train_dataset = hfds2tvds(train_dataset,input_shape)    
    test_dataset = load_dataset('beans', split='test')
    test_dataset = hfds2tvds(test_dataset,input_shape)    
    return train_dataset, test_dataset,'hfds'

@_add_dataset
def keremberke_pokemon_classification(root,input_shape=224):
    train_dataset = load_dataset('keremberke/pokemon-classification', name='full')['train']
    train_dataset = hfds2tvds(train_dataset,input_shape) 
    test_dataset = load_dataset('keremberke/pokemon-classification', name='mini')['test']
    test_dataset = hfds2tvds(test_dataset,input_shape) 
    return train_dataset, test_dataset,'hfds'

@_add_dataset
def poolrf2001_facemask(root,input_shape=224):
    from torchvision.datasets import ImageFolder
    train_set = ImageFolder(os.path.join(root,'poolrf2001_facemask','train'),transform=_get_transforms(input_shape,augment=False)[0])
    test_set = ImageFolder(os.path.join(root,'poolrf2001_facemask','test'),transform=_get_transforms(input_shape,augment=False)[0])
    return train_set,test_set,'tvds'

@_add_dataset
def matthijs_snacks(root,input_shape=224):
    train_dataset = load_dataset("Matthijs/snacks", split='train')
    train_dataset = hfds2tvds(train_dataset,input_shape) 
    test_dataset = load_dataset("Matthijs/snacks", split='test')
    test_dataset = hfds2tvds(test_dataset,input_shape) 
    return train_dataset, test_dataset,'hfds'

@_add_dataset
def fastjobs_visual_emotional_analysis(root,input_shape=224):
    train_dataset = load_dataset("FastJobs/Visual_Emotional_Analysis", split='train')
    # train_dataset.set_format(type='torch', columns=['image', 'label'])
    # test_dataset = load_dataset("FastJobs/Visual_Emotional_Analysis", split='validation')
    # test_dataset.set_format(type='torch', columns=['image', 'label'])
    train_dataset = hfds2tvds(train_dataset,input_shape=input_shape) 
    return train_dataset, '','hfds'

@_add_dataset
def chest_xray_classification(root,input_shape=224):
    return GransferHuggingFaceImageDataset.load('keremberke/chest-xray-classification')

@_add_dataset
def trec(root, input_shape=224):
    return GransferHuggingFaceTextDataset.load('trec')


def hftxtds2tvds(ds: HuggingFaceDataset):
    print(ds.features)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    text = model.encode(ds["text"], convert_to_tensor=True)
    labels = ds["coarse_label"]
    from torch.utils.data import TensorDataset
    ds = TensorDataset(text, tensor(labels))

    return ds

def get_dataset(root,dataset_name,input_shape=224,splits=[''],return_classes=False):
    if ']' in dataset_name:
        classes = dataset_name.strip('][').replace("'","").split(', ')
        # print(f'classes: {classes}')
        ds_type = 'hfpics'
        raw_ds,_, ds_type  = _DATASETS[ds_type](os.path.join(root,'datasets'),classes,input_shape)
    else:
        raw_ds, _, ds_type = _DATASETS[dataset_name.lower()](os.path.join(root,'datasets'),input_shape=input_shape)
    
    try:
        length = len(raw_ds)
    except:
        length = raw_ds.get_num_samples('train')
    LEN = min(length,5000)
    print(f'dataset size: {length}')
    print(type(raw_ds))

    classes = raw_ds.classes

    ## Load dataset
    # randomly sample MAX_NUM_SAMPLES
    # if isinstance(splits,str): splits = list(splits)
    print(splits)
    data = []
    for i, spl in enumerate(splits):
        torch.cuda.empty_cache()
        print(f'split: {spl}')
        if spl == '':
            idx = random.sample(range(length), k=LEN)
            ds = torch.utils.data.Subset(raw_ds, idx)
        elif spl == ':800':
            # ds = ds.select(range(min(800,LEN)))
            # ds = torch.utils.data.Subset(ds, range(min(800,LEN))) # 800
            idx = random.sample(range(length), k=int(length*0.5) )# min(int(length*0.2),8000))
            # idx = random.sample(range(length), k=8000) 
            ds = torch.utils.data.Subset(raw_ds, idx)
        elif spl == '800:1000':
            # ds = torch.utils.data.Subset(ds, range(min(800,LEN)-200,min(800,LEN)))
            idx = random.sample(range(length), k=2000)
            ds = torch.utils.data.Subset(raw_ds, idx)
        elif spl == ':100%':
            ds = torch.utils.data.Subset(raw_ds, range(min(500,LEN)))
        data.append(ds)
    print(f'len(data_sets): {len(data)}')
    if return_classes:
        return data, classes
    torch.cuda.empty_cache()
    return data, ''

# root,dataset_name,input_shape,batch_size,train_slice,val_slice
def get_dataloader(root,dataset_name,data_sets=[],input_shape=224,batch_size=16,splits=[''],return_classes=False):
    classes = []
    if data_sets == []:
        data_sets, classes = get_dataset(root,dataset_name,input_shape=input_shape,splits=splits,return_classes=return_classes)
    
    loader_list = []
    print(f'len(data_sets): {len(data_sets)}')
    for data in data_sets:
        dataloader = DataLoader(
                    data,
                    batch_size=batch_size, # may need to reduce this depending on your GPU 
                    num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True
                )
        print(f'dataloader size: {len(dataloader)}')
        loader_list.append(dataloader)
    if return_classes:
        return loader_list, classes
    return loader_list