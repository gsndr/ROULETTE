# neuRal attentiOn MULti-Output ModEl for explainable InTrusion DeTEction  (ROULETTE)


The repository contains code refered to the work:

_Giuseppina Andresini, Annalisa Appice, Francesco Paolo Caforio, Donato Malerba, Gennario Vessio_

[ROULETTE: A Neural Attention Multi-Output Modelfor Explainable Network Intrusion Detection]() 

Please cite our work if you find it useful for your research and work.
```
 @article{ANDRESINI2021108,

```

![ROULETTE]()

## Code requirements
The code relies on the following python3.7+ libs.
Packages needed are:
* Tensorflow 2.4.0
* Pandas 1.3.2
* Numpy 1.19.5
* Matplotlib 3.4.2
* Hyperopt 0.2.5
* Keras 2.4.3
* Scikit-learn 0.24.2
* [Visual-attention-tf 1.2 ] (https://pypi.org/project/visual-attention-tf/)


## Data
The following [DATASETS](https://drive.google.com/drive/folders/15IYzxZLt8C02kQ13sqPDCGXuJeQobP8N?usp=sharing).
The datasets used are:
* NSL-KDD
* UNSW-NB15


## How to use

The repository contains the following scripts:
* main.py:  script to execute ROULETTE
* TrainModel.py: script to execute the learning task with Hyperopt using a Multi-output classifier
* TrainSingleOutput.py: script to execute the learning task with Hyperopt using a Single-ouput classifier
* ExplanationMap.py: script to create the Attention maps

The description of the other scripts can be found [here](https://github.com/Kyanji/MAGNETO/)

## Replicate the experiments
Modify the following code in the main.py script to change the beaviour of ROULETTE

# Parameters
```python
param = {"Max_A_Size": 10,  # Heigth and Weight of the images
         "Max_B_Size": 10, 
         "Dynamic_Size": False,  # search the minimum A and B to create 0 Collisions
         'Metod': 'tSNE',   # {tSNE, kpca, pca} to create the mapping between examples and images 
         "ValidRatio": 0.1, 
         "seed": 180,
         "dir": "dataset/dataset4/",  # path of dataset
         "Mode": "MultiBAttention",  # Mode : MultiAttention for Single Output, MultiBAttention is the multi-output
         "LoadFromPickle": False, # load dataset images from pickle
         "mutual_info": True,  # Mean or MI
         "hyper_opt_evals": 20, 
         "epoch": 150, #maximum epochs to retrina the model
         "No_0_MI": False,  # True : remove 0 MI Features
         "autoencoder": False, # use autoencoder to reduce the number of features
         "enhanced_dataset": "gan"  # gan, smote, adasyn, ""None""
         "dataset": 'NSL-KDD',  #Name of the dataset used
         "trainModel": False, #if true train a new model if false load a pretrained model
         "attentionLayer": True, #if True the attention layer is created, otherwise a model without Attention layer is created
         "classes":5,  #number of classes in multi-class strategy
         "classification": "classification", # name of multi-class label column 
         "classificationBinary": "classificationBinary", # binary label column  
         "createImage": 1, #if 1 create and save the Attention maps
         }
```








