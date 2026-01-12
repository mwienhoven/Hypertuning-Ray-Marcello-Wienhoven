# Hypertuning-Ray-Marcello-Wienhoven
The repository for assignment 4-hypertuning-ray for the Portfolio-Marcello-Wienhoven

## Used dataset

The dataset used in this repository can be found in the 
[**CIFAR-10 dataset**](https://www.cs.toronto.edu/~kriz/cifar.html). The dataset used is the CIFAR-10 classification dataset. This dataset contains 60000 samples (50000 training samples and 10000 test samples). There are 10 classes (airplanes, automobiles, birds, cats, deers, dogs, frogs, horses, ships, and trucks.) The images are relatively small also with a 32x32 pixel size that helps with the computational speed.

In the beginning of developing the code for the assignment, I relied on the Flowers102 dataset, which can be found on [**Flowers Recognition**](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition). More info can also be found on [**102 Category Flower Dataset**](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/). This dataset contains 102 classes, and each class contains of between 40 and 258 images (samples). The CNN used in this research just could not perform on this dataset, this dataset is too complex for the simple CNN structure. This is the reason I switched to the CIFAR-10 dataset.

The images in CIFAR-10 have an image size of 32 pixels (height and width)

## Sync the environment
Run the command:

```bash
uv sync
```

Also to clear the uv cache, run the following command

```bash
uv cache clean
```

## Open the MLFlow Dashboard
Run the command:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

## Using the code

### Loading the data
The data can be loaded with the command 

```bash
uv run load_data.py
```

The data is downloaded with a batch size of 32, and validation split of 0.2. This results in 40000 training samples, 10000 validation samples, and 10000 test samples.

### Training the data (testing manually)
The model in src/models/cnn.py is used for the training. The settings are written in config.toml. To train the model given the model and the settings, run the following command.

```bash
uv run train.py
```

### Hypertuning
The hypertuning can be done using the following command:

```bash
uv run hypertune.py
```

### Ray analysis
The ray analysis provides contour heat plots of the trials trained while hypertuning. The images are saved in the /img folder. The ray analysis can be performed using the following command:

```bash
uv run ray_analysis.py
```

## Manual testing
Some manual test will be ran before hypertuning to gain insights for optimal search spaces. The test will be described in the following subsections. The baseline settings of training are saved in DONT_CHANGE_orig_training.toml

### Original model
The original model, with the settings from DONT_CHANGE_orig_training.toml, achieved a highest accuracy of 0.53.

### Filters
The original model used 32 filters.

When 64 filters are used, the model achieved a highest accuracy of 0.50. When 128 filters are used, the model achieved a highest accuracy of 0.53. When 256 filters are used, the model achieved a highest accuracy of 0.44.

The filters are als saved in the original model experiment name by accident.

### Units of layer 1
The original model used 64 units for layer 1. 

When 32 units are used, the model achieved a highest accuracy of 0.59. When 128 units are used, the model achieved a highest accuracy of 0.47. When 256 units are used, the model achieved a highest accuracy of 0.56.

### Units of layer 2
The original model used 32 units for layer 2.

When 16 units are used, the model achieved a highest accuracy of 0.44. When 64 units are used, the model achieved a highest accuracy of 0.50. When 128 units are used, the model achieved a highest accuracy of 0.53.

### Number of layers
The original model used 2 layers.

When 1 layer is used, the model achieved a highest accuracy of 0.50. When 3 layers are used, the model achieved a highest accuracy of 0.41. When 5 layers are used, the model achieved a highest accuracy of 0.50.

### Kernel size
The original model used a kernel size of 3.

When a kernel size of 2 is used, the model achieved a highest accuracy of 0.41. When a kernel size of 4 is used, the model achieved a highest accuracy of 0.50. When a kernel size of 5 is used, the model achieved a highest accuracy of 0.44.

### Dropout
The original model used a dropout of 0.2.

When a dropout of 0.0 is used, the model achieved a highest accuracy of 0.44. When a dropout of 0.4 is used, the model achieved a highest accuracy of 0.34. When a dropout of 0.7 is used, the model achieved a highest accuracy of 0.34. When a dropout of 1.0 is used, the model achieved a highest accuracy of 0.16. The last one was a test for myself, because a dropout of 1.0 is strange and not practical.

## Hyperparameter tuning
Based on the test ran above, the hyperparameter tuning ranges were set to:
```python
search_space = {
        "filters": tune.choice([16, 32, 64, 128]),
        "units1": tune.choice([64, 128, 256]),
        "units2": tune.choice([32, 64, 128]),
        "num_layers": tune.randint(2, 4),
        "kernel_size": tune.choice([3, 5]),
        "dropout": tune.uniform(0.0, 0.5),
    }
```

200 samples of max 10 epochs were ran using Ray (using HyperOpt). The result were analyzed by contour heat plots. These are saved in the /img subfolder.

## Best model
