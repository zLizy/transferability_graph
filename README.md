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

For HuggingFace datasets:
```python
train_dataset = load_dataset('beans', split='train') # name of the dataset
```
### Dataset name mapping
- If dataset names start with **tweet_eval** or **glue**, it is a subset of either **tweet_eval** or **glue**
```json
{"tweet_eval_sentiment":[
        "negative", "neutral", "positive"
    ]}
```
For example, `tweet_eval_sentiment` is the subset, `sentiment`, within the larget dataset collection, `tweet_eval`

### Data Sources
- [torchvision.datasets](https://pytorch.org/vision/stable/datasets.html)

- [huggingface](https://huggingface.co/datasets?task_categories=task_categories:image-classification&sort=downloads)

## Dataset Embeddings
To get dataset embeddings, two ways are implemented:
- Task2VEC

`./dataset_embed/task2vec_embed/embed_task.py`

The features of the datasets are stored under foler: `heterogeneous_graph/dataset_embed/task2vec_embed/feature/` with name as `[dataset_name]_feature.p`, e.g., `caltech101_feature.p`.

- Domain Similarity

`./dataset_embed/domain_similarity/embed_offline.py`

The features of the datasets are stored under foler: `heterogeneous_graph/dataset_embed/domain_similarity/feature/resnet50/` with name as `[dataset_name]_feature.npy`, e.g., `caltech101_feature.npy`.

## Model Embeddings
- Attribution Map
Path of the script: `model_embed/attribution_map/embed_offline.py'

The features of the models are stored under folder: `heterogeneous_graph/model_embed/attribution_map/feature/`, e.g., `cifar10/aaraki_vit-base-patch16-224-in21k-finetuned-cifar10_input_x_gradient.npy` 

## Graph Construction
```cd ./graph ```
```python
python3 leave_one_out.py \
                                                        -contain_data_similarity ${CONTAIN_DATA_SIMILARITY} \
                                                        -contain_dataset_feature ${CONTAIN_DATASET_FEATURE} \
                                                        -embed_dataset_feature ${EMBED_DATASET_FEATURE} \
                                                        -contain_model_feature ${CONTAIN_MODEL_FEATURE} \
                                                        -embed_model_feature ${EMBED_MODEL_FEATURE} \
                                                        -complete_model_features ${complete_model_features} \
                                                        -accuracy_thres ${ACCU_THRES} \
                                                        -gnn_method ${GNN_METHOD} \
                                                        -finetune_ratio ${FINETUE_RATIOS} \
                                                        -hidden_channels ${hidden_channels} \
                                                        -test_dataset ${dataset}
```

## Graph Neural Network Learning

## Link Prediction

