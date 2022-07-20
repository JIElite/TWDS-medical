# TWDS Group20 Medical Project

Note!!
**This repository is currently deprecated and I have moved it to an organization to continue the development, please refer to our [new repository](https://github.com/TWDS-2022-Group20/model-trainer)**

## Pre-requisite
In this project, we use mlflow to track the records of each training experiments. Before you step into this project, please make sure your are familar with building a mlflow localhost server.

## Introduction
Nowadays, many people feel unhappy in their daily life. Moreover, someone have suffered the depression for a while. We collect the Quetionnaires from the CDC of United States to explore what kind of people may be in the risk of depression.


## Dataset
We selected the Questionnaires released annually from The CDC in United States to study our topic, due to its completeness. Each questionnaires contain 300 questions to answer, the questions contain either catebgorical or numeric information,

source of the dataset: https://www.cdc.gov/brfss/questionnaires/index.htm


## Model Training

We provided two kinds of method to do model seletion:
1. Hold-Out Training
2. Cross-Validation

The corresponding command lines are as follows:


Train a model with hold-out validation
```
$ python train_model.py
```

Train a model with cross-validation
```
$ python train_model_cv.py
```

In both `train_model.py`, and `train_model_cv.py`, please setup
the parameters to the experiement along with model parameters and our
dedicated Training Framework will train the model and report the performance.
