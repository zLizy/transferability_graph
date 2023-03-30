import pandas as pd
from datasets import get_dataset_config_names
from datasets import load_dataset

dataset_list = ['cifar10','cifar100','food101','beans','cats_vs_dogs',
                'nelorth/oxford-flowers','fashion_mnist',
                'rajistics/indian_food_images','sasha/dog-food','keremberke/pokemon-classification',
                'mnist','svhn','poolrf2001/mask']

label_map = {'eurosat':"['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']",
             'cifar10': "['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']"
            }


for dataset_name in dataset_list[9:]:
    print('-----------')
    print(f'dataset_name: {dataset_name}')
    configs = get_dataset_config_names(dataset_name)
    print(f'configs: {configs}')
    if dataset_name == 'cats_vs_dogs': continue
    try:
        ds = load_dataset(dataset_name, split="test")
    except:
        ds = load_dataset(dataset_name, 'full')
    try:
        copy_of_features = ds.features.copy()
    except:
        print(ds.keys())
        copy_of_features = ds['test'].features.copy()
    # print(f'features: {copy_of_features}')
    print(f'feature_key: {copy_of_features.keys()}')
    key = [k for k in copy_of_features.keys() if 'label' in k]
    labels = copy_of_features[key[0]].names
    print(f'labels:{labels}')


