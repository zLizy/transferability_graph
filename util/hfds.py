from datasets import load_dataset

Dataset = collections.namedtuple(
    'Dataset', ['trainset', 'testset'])


def _add_dataset(dataset_fn):
    _DATASETS[dataset_fn.__name__] = dataset_fn
    return dataset_fn

@_add_dataset
def cats_vs_dogs(root):
    train_dataset = load_dataset('cats_vs_dogs', split='train')
    train_dataset.set_format(type='torch', columns=['image', 'labels'])
    test_dataset = load_dataset('cats_vs_dogs', split='test')
    test_dataset.set_format(type='torch', columns=['image', 'labels'])
    return train_dataset, test_dataset


@_add_dataset
def beans(root):
    train_dataset = load_dataset('beans', split='train')
    train_dataset.set_format(type='torch', columns=['image_file_path','image','labels'])
    test_dataset = load_dataset('beans', split='test')
    test_dataset.set_format(type='torch', columns=['image_file_path','image','labels'])
    return train_dataset, test_dataset

@_add_dataset
def keremberke_pokemon_classification(root):
    train_dataset = load_dataset('keremberke/pokemon-classification', split='train')
    train_dataset.set_format(type='torch', columns=['image_file_path','image','labels'])
    test_dataset = load_dataset('keremberke/pokemon-classification', split='validation')
    test_dataset.set_format(type='torch', columns=['image_file_path','image','labels'])
    return train_dataset, test_dataset

@_add_dataset
def poolrf2001_mask(root):
    train_dataset = load_dataset("poolrf2001/mask", split='train')
    train_dataset.set_format(type='torch', columns=['image', 'labels'])
    test_dataset = load_dataset("poolrf2001/mask", split='validation')
    test_dataset.set_format(type='torch', columns=['image', 'labels'])
    return train_dataset, test_dataset

@_add_dataset
def Matthijs_snacks(root):
    train_dataset = load_dataset("Matthijs/snacks", split='train')
    train_dataset.set_format(type='torch', columns=['image', 'label'])
    test_dataset = load_dataset("poolrf2001/mask", split='validation')
    test_dataset.set_format(type='torch', columns=['image', 'label'])
    return train_dataset, test_dataset

@_add_dataset
def FastJobs_Visual_Emotional_Analysis(root):
    train_dataset = load_dataset("FastJobs/Visual_Emotional_Analysis", split='train')
    train_dataset.set_format(type='torch', columns=['image', 'label'])
    test_dataset = load_dataset("FastJobs/Visual_Emotional_Analysis", split='validation')
    test_dataset.set_format(type='torch', columns=['image', 'label'])
    return train_dataset, test_dataset

@_add_dataset
def keremberke_chest_xray_classification(root):
    train_dataset = load_dataset('keremberke/chest-xray-classification', split='train')
    train_dataset.set_format(type='torch', columns=['image_file_path','image','labels'])
    test_dataset = load_dataset('keremberke/chest-xray-classification', split='validation')
    test_dataset.set_format(type='torch', columns=['image_file_path','image','labels'])
    return train_dataset, test_dataset