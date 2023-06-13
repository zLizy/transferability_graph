# Graph Learning for Transfer Learning

## Data Preperation

Create a folder name 'datasets'
```
mkdir datasets
```
You may implement the dataset loading method in: 
`util/dataset.py`
For example, you may define a new dataset function as in the follwoing.

```python
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
```

### Data Sources
- [torchvision.datasets](https://pytorch.org/vision/stable/datasets.html)

- [huggingface](https://huggingface.co/datasets?task_categories=task_categories:image-classification&sort=downloads)

## Dataset Embeddings
To get dataset embeddings, two ways are implemented:
- Task2VEC

`./dataset_embed/task2vec_embed/embed_task.py`
- Domain Similarity

`./dataset_embed/domain_similarity/embed_offline.py`

## Model Embeddings
- Attribution Map
- `model_embed/attribution_map/embed_offline.py'

## Graph Construction

## Graph Neural Network Learning

## Link Prediction

