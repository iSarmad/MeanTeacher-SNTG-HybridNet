# Implementation of Mean Teacher, SNTG and HybridNet 
This Repo is the implementation of the following three papers

* [Mean Teacher](https://arxiv.org/abs/1703.01780) Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results 
* [SNTG](https://arxiv.org/pdf/1711.00258.pdf)   Smooth Neighbors on Teacher Graphs for Semi-supervised Learning
* [HybridNet](https://arxiv.org/abs/1807.11407) HybridNet: Classification and Reconstruction Cooperation for Semi-Supervised Learning


I only used Cifar-10 Dataset. I have used the following architectures:

1. Convlarge ( Mean Teacher and SNTG)
2. Convlarge based HybridNet ( For Hybrid Net ) 



## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

What things you need to install:
1. Pytorch 0.4.1
2. Python 3.6.6
3. TensorboardX

I used 2 Titan Xp GPUs. Average training time is 3 hours for hybrid network and 1 hour for other networks.
 
### Dataset 
 You may set up CIFAR-10 inside the repository by running the following command.

```
./data-local/bin/prepare_cifar10.sh
```

###  Accuracy Achieved on Test Dataset

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
1. Supervised Only without BN : 76.6% 
2. Mean Teacher without BN: 
    a) Student Model : 83.58%
    b) Teacher Model : 86.78%
3. Mean Teacher with BN
    a) Student Model : 84.4%
    b) Teacher Model : 87.07%
4. Mean Teacher + SNTG with BN
    a) Student Model : 84.6%
    b) Teacher Model : 87.28%
5. Hybrid Network
    a) Student Model : 84.18%
    b) Teacher Model : 87.00%
```


## Running the Training 


### Supervised Model Only (4000 labels of Cifar-10)
Go the parameters.py and change the following flags as follows:

1. supervised_mode = True ( To use only 4000 labels for training)
2. lr = 0.15  ( setting the learning rate)
3. BN = False  ( for turning batch Normalization on or off)
4. sntg = False ( Do not use any SNTG loss )
5. Do not change any other settings and run main.py
 Note that my baseline has not Batch Normalization in it. 
### Mean Teacher Only 
Go the parameters.py and change the following flags as follows:

1. supervised_mode = False ( To use only 4000 labels for training)
2. lr = 0.2  ( setting the learning rate)
3. BN = False or True  ( for turning batch Normalization on or off)
4. sntg = False ( Do not use any SNTG loss )
5. Do not change any other settings and run main.py

Note that my baseline has not Batch Normalization in it. However I tested mean teacher with both a BN and without BN
 


### Mean Teacher + SNTG Loss 
Go the parameters.py and change the following flags as follows:

1. supervised_mode = False ( To use only 4000 labels for training)
2. lr = 0.2  ( setting the learning rate)
3. BN = True  ( for turning batch Normalization on or off)
4. sntg = True ( Do not use any SNTG loss )
5. Do not change any other settings and run main.py

### HybridNet  
Go the parameters.py and change the following flags as follows:

1. supervised_mode = False ( To use only 4000 labels for training)
2. lr_hybrid = 0.2  ( setting the learning rate)
3. BN = True  ( for turning batch Normalization on or off)
4. sntg = False ( Do not use any SNTG loss )
5. Do not change any other settings and run main_hybrid.py



## Tensorboard Visualization
To Visualize on Tensorboard, use the following command 
```
tensorboard --logdir=”path to ./ckpt”
```
Note that all the checkpoints are in the ./ckpt folder so simply start a tensorboard session to visualize it. Also all the saved checkpoints for student models are also saved there.
```
1. Baseline : 12-03-18:09/convlarge,Adam,200epochs,b256,lr0.15/test
2. Mean teacher without BN :
   12-03-20:12/convlarge,Adam,200epochs,b256,lr0.15/test
   12-03-23:38/convlarge,Adam,200epochs,b256,lr0.2/test
3. Mean Teacher with BN : 12-05-11:55/convlarge,Adam,200epochs,b256,lr0.2/test
4. Hybrid Net : 12-06-10:58/hybridnet,Adam,200epochs,b256,lr0.2/test
5. SNTG + Meant Teacher: 12-07-00:36/convlarge,Adam,200epochs,b256,lr0.2/test
```


## License

This project is licensed under the MIT License. 
For specific helper function used in this repository please see the license agreement of the Repo linked in Acknowledgement section
## Acknowledgments
My implementation has been inspired from the following sources.

* [Mean Teacher](https://github.com/CuriousAI/mean-teacher) : I have mainly followed the Pytorch Version of this Repo
* [SNTG](https://github.com/xinmei9322/SNTG) - I have understood the concept of SNTG and converted Theano Implementation to Pytorch
* [Hybrid Network](https://github.com/dakshitagrawal97/HybridNet) - I have followed this repository to incorporate reconstruction loss in my implementation. 
