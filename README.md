# Model Agnostic Meta Learning
> This is a fork repository from the main MAML code.


## Installation
In-order to run the code you must have the following softwares and packages available:
* Conda
* Tensorflow 1.15
* Keras


To install Conda please follow [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

```
conda install -c conda-forge tensorflow==1.15 keras
```

## Pre-processing

* First you need to download the MiniImageNet dataset from [here](https://drive.google.com/file/d/1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk/view).
* Extact the file and move it to `./data_preprocessing/` folder.
* Navigate to `./data_preprocessing`
* Run `python proc_images.py`
> This will resize and cluster your images into train and testset.


## Run
### Training
This source code is optimised to run on both GPU and CPU. 
Is you have GPU available make sure you have CUDA >= 9.2.
To train a meta-learning network, please take a look at `main_train.py`. 
You change the files `flags` in-order to train on `20-shot`, `5-shot`. 
To start training run: 
```
python main_train.py
```


### Testing
To start testing run: 
```
python main_test.py
```
> Make sure you have trained your model and saved at `./logs` folder.




