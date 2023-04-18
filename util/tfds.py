import torch
from PIL import Image
import torchvision.transforms as transforms
import tensorflow_datasets as tfds


def download_tfds_dataset(name, data_dir=None):
    import tensorflow_datasets as tfds
    import timm
    builder = tfds.builder(name, data_dir=data_dir)
    builder.download_and_prepare()

def disable_gpus_on_tensorflow():
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')  

class VTABIterableDataset(torch.utils.data.IterableDataset):

    def __init__(self, tfds_dataset, split="test", input_name="image", label_name="label", input_mode="RGB", transform=None, target_transform=None, classes=None,batch_size=1):
        self.tfds_dataset = tfds_dataset
        self.input_name = input_name
        self.label_name = label_name
        self.transform = transform
        self.target_transform = target_transform
        self.input_mode = input_mode
        self.num_examples = tfds_dataset.get_num_samples(split)
        self.split = split
        self.batch_size = batch_size
        if classes is None:
            self.classes = tfds_dataset._dataset_builder.info.features['label'].names
        else:
            self.classes = classes

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        print(f'worker_info: {worker_info}')
        iterator = self.tfds_dataset.get_tf_data(self.split, batch_size=self.batch_size, epochs=1, for_eval=True)# self.batch_size = 1
        if worker_info is not None:
            iterator = iterator.shard(index=worker_info.id, num_shards=worker_info.num_workers)
        nb = 0
        self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                        ])
        for data in iterator:
            inputs = (data[self.input_name].numpy())
            # print(f'type inputs: {type(inputs)}, inputs shape: {inputs.shape}')
            labels = data[self.label_name].numpy()
            for input, label in zip(inputs, labels):
                # input = Image.fromarray(input, mode=self.input_mode)
                if self.transform is not None:
                    # print('transform is not None')
                    input = self.transform(input)
                # print(f'type input: {type(input)}, inputs shape after transforms: {input.shape}')
                if self.target_transform is not None:
                    label = self.target_transform(label)
                yield input, label

        # return iterator
                
    def __len__(self):
        return self.num_examples