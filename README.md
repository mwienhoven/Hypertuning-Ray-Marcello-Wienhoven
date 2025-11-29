# Hypertuning-Ray-Marcello-Wienhoven
The repository for assignment 4-hypertuning-ray for the Portfolio-Marcello-Wienhoven

## Used dataset

The dataset used in this repository can be found in the 
[**CIFAR-10 dataset**](https://www.cs.toronto.edu/~kriz/cifar.html). The dataset used is the CIFAR-10 classification dataset. This dataset contains 60000 samples (50000 training samples and 10000 test samples). There are 10 classes (airplanes, automobiles, birds, cats, deers, dogs, frogs, horses, ships, and trucks.) The images are relatively small also with a 32x32 pixel size that helps with the computational speed.


### Flowers102 dataset (beginning of the project)
In the beginning of developing the code for the assignment, I relied on the Flowers102 dataset, which can be found on [**Flowers Recognition**](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition). More info can also be found on [**102 Category Flower Dataset**](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/). This dataset contains 102 classes, and each class contains of between 40 and 258 images (samples). The CNN used in this research just could not perform on this dataset, this dataset is too complex for the simple CNN structure. This is the reason I switched to the CIFAR-10 dataset.


### Activate the virtual environment (.venv)
To activate the virtual environment, run the following command:
```bash 
source .venv/bin/activate
```

### Loading the data
The data can be loaded with the command:

```bash
uv run load_data.py
```

The data is downloaded with a batch size of 32, and validation split of 0.2. This results in 40000 training samples, 10000 validation samples, and 10000 test samples.

### Training the CNN model
The CNN model can be trained with the command:

```bash
uv run train.py
```

This will train the CNN model (that can be found in src/models/cnn.py). The configurations can be changed in the config.toml file. Based on the CNN model and the configurations, the model will be trained. The results will be stored in MLFlow (mlflow.db).

### Open MLFlow GUI
To open the MLFlow GUI, run the following command
```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --host 127.0.0.1 \
    --port 5000
```

In a single line:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

## Testing phase
### Filters
### Kernel size
### Maxpool
### Number of layers
### Dropout
### Units 1 
### Units 2 
### Learning rate
### Optimizer

## Report
