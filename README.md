# Model Selection with Model Zoo via Graph Learning
Under review in ICDE 2024

In this study, we introduce **TransferGraph**, a novel framework that reformulates model selection as a graph learning problem. TransferGraph constructs a graph using extensive metadata extracted from models and datasets, while capturing their intrinsic relationships. Through comprehensive experiments across 12 real datasets, we demonstrate TransferGraph’s effectiveness in capturing essential model-dataset relationships, yielding up to a 21.8% improvement in correlatio between predicted performance and the actual fine-tuning results compared to the state-of-the-art methods.

![image](https://github.com/zLizy/transferability_graph/blob/main/img/overview.jpg)

## Model zoo and data collection
**Datasets**: We use 11 vision datasets, including 10 datasets from the public transfer learning benchmark [VTAB](https://github.com/google-research/task_adaptation), and [StanfordCars](https://pytorch.org/vision/stable/generated/torchvision.datasets.StanfordCars.html). All the datasets are publicly available online.

**Models**: We construct a model zoo with 185 heterogeneous pre-trained models. These models vary in terms of various aspects, e.g., pre-trained dataset, architectures, and various other metadata features. All the models are available from [HuggingFace](https://huggingface.co/models).

## Instructions
### Data preparation
* Collect metadata (dataset, model), e.g., attributes, performance. The files are under `doc/`
* Obtain **Transferability score** - **LogMe**.
```console
  python3 ./LogMe/LogMe.py
```   
*  Run **TransferGraph** to map model-dataset relationships in a graph and use GNN to train node representations.
```console
cd graph
./run_graph.sh
``` 
