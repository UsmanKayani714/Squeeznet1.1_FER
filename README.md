# SqueezeNet 1.1: Lightweight, Fast and Efficient Recognition of Driver's Facial Expression

This is the **official repository** for the [**paper**](https://arxiv.org/abs/) "_Shuffle Vision Transformer: Lightweight, Fast and Efficient Recognition of Driver’s Facial Expression_".

## SqueezeNet 1.1 Architecture

<div style="display: flex; justify-content: flex-start;">
  <img width=680 src="figures/shuffarch.png"/>
</div>

## Datasets

- FER2013 dataset (facial expression recognition, 6 classes: angry, disgust, fear, happy, sad, surprise)

### Preprocessing

-_For FER2013 dataset_: 'python preprocess_fer2013.py' to save the train and test data in .h5 format.

### Train and Test model

- _FER2013 dataset_: python combinemodelkmu.py --model Ourmodel --bs 32 --lr 0.0001

### plot confusion matrix

- python confmatrixkmu.py --model Ourmodel
- python confmatrixkdef.py --model Ourmodel

### FER2013 Accuracy

- Model： SqueezeNet 1.1 ; Accuracy： TBD <Br/>

### Confusion matrices

<div style="display: flex; justify-content: flex-start;">
  <img width=400 src="figures/ok12.png"/>
  <img width=400 src="figures/ok11.png"/>
</div>
