# MSCeleb1MResNetTorchModel
This is the test code for a resnet 18-layer model for [MS-Celeb-1M] (http://www.msceleb.org/). 

The training code is from [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch). The total training epochs are reduced to 30 epochs. The learning rate is decreased from 0.1 to 0.01 after 10 epochs, and to 0.001 after 20 epochs. All other settings are the same

## Requirement
torch

matio & matio-ffi.torch


## Model download

A ResNet-18 layer model: [Google Drive](https://drive.google.com/open?id=10bUQVY3BhBRl6fmTfuIgtbAmevqfkMne)

## Usage
th classify.lua model/model_best.t7 test.txt 

Results will be saved in the folder ./images with a '.mat' suffix.


