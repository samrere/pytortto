# PyTortto
- [Intro](#Intro)
- [Installation](#Installation)
- [Prerequisites](#Prerequisites)
- [Quick start](#Quick-start)
- [Examples](#Examples)
  - [Resnet](#Resnet)
  - [UNet](#UNet)
  - [Vision Transformer](#Vision-Transformer)
  - [DCGAN](#DCGAN)
- [Supported functionalities](#Supported-functionalities)


## Intro
This is a pytorch style machine learning framework implemented entirely in numpy, with GPU acceleration from cupy.

Similar to the pokemon "ditto", pytortto works exactly like pytorch, although inferior in terms of speed. The purpose of this project is to understand how pytorch, or how general machine learning algorithms work under the hood. Max effort was given to correctness, calculation efficiency (like simpler Jacobian in logsoftmax, efficient implementation of convolution etc.), numerical stability (log-sum-exp used in logsigmoid, BCEWithLogitsLoss etc.), and memory efficiency (implementation of caching, view etc.).  

When compared in GPU, Tortto is around 1.5(vision transformers) ~ 3(CNNs) times slower than pytorch. It also achieves the same complexity as pytorch, which means tortto can be used to train relatively larger models such as `resnet101` and vision transformer `ViT-B/16` with same speed ratio.

Tortto implements reverse mode automatic differentiation and supports dynamic computation graph like pytorch.


## Installation
```python
pip install tortto
```

## Prerequisites
**numpy** (required): Only use its basic functions. The "highest-level" function used are FFT, but I also implemented those in another repo: **[FFT from scratch](https://github.com/samrere/fft-from-scratch)**. To use FFT in `Conv2d` and `ConvTranspose2d`, set `os.environ['fft'] = 'True'` before `import tortto`  

**scipy** (optional): Can be installed to improve efficiency in some functions:  
* `scipy.fft.rfft2` and `scipy.fft.irfft2`. Numpy equivalence will be used if scipy is not installed, but `np.fft` only works in complex128 so it's slow.
* `scipy.special.erf` used in `nn.GELU`. approximation of erf is used when scipy is not installed.
* `scipy.sparse` used in `nn.Embedding` when setting `sparse=True`. Can't use sparse if neither scipy or cupy is installed.   

**cupy** (optional): Compute Numpy functions in GPU. To send a tensor/module to GPU, use `.cuda()`.  

**pytorch** (optional): Only use dataset, dataloder and transforms, required in actual training to load and preprocess data. In each iteration, torch tensors (i.e. data, label) will be converted to tortto tensors by `tortto_tensor = tortto.tensor(torch_tensor.numpy()).cuda()`. All computations after that is done in tortto. See examples below.

## Quick start
pytortto uses the same syntax as pytorch. Here shows the forward and backward propagation of some functions.

```python
import tortto as tt

x = tt.tensor([[1.1, 2.0, 3.6], [4.7, 3.14, 2.718]], requires_grad=True)
y = tt.tensor([[2., 3.99], [8.4, 1.5], [2.5, 7.8]], requires_grad=True)
output = tt.tanh(1/(tt.cat([tt.sin((tt.exp(x ** 1.4) / 3.1 ** tt.log(x)) @ y), y]).mean()))
output.backward()

print(output)
# tensor(0.3489, grad_fn=<tanhBackward>)
print(x.grad)
# [[ -0.0276   0.6477 -11.827 ]
#  [ 27.2554  -5.3062   1.0978]]
print(y.grad)
# [[-10.8005   8.2755]
#  [ -0.3336   0.2187]
#  [  0.8972  -0.9684]]
```
Tortto also implements common Modules. Let's try `Conv2d` and compare its result and speed with pytorch:  
prep common weight, bias, input data and grad
```python
import numpy as np

weight = np.random.randn(16, 2, 3, 2).astype('f')
bias = np.random.randn(16).astype('f')
data = np.random.randn(128, 4, 64, 64).astype('f')
grad = np.random.randn(128, 16, 66, 23).astype('f')
```
tortto `conv2d`:  
```python
# defaults to 'False'. Switch to 'True' to use fft. Need to restart kernel to take effect
import os
os.environ['fft'] = 'False'
import tortto as tt
import tortto.nn as nn

m = nn.Conv2d(in_channels=4, 
              out_channels=16, 
              kernel_size=(3, 2), 
              stride=(1, 3), 
              padding=(2, 3), 
              dilation=(1, 2),
              groups=2, 
              bias=True, 
              padding_mode='zeros')  # only support zero padding for now
m.weight.data = weight.copy() # replace weight and bias
m.bias.data = bias.copy()
x = tt.tensor(data, requires_grad=True)
y = m(x)
y.backward(tt.tensor(grad))
```
pytorch `Conv2d`:  
```python
import torch

m_torch = torch.nn.Conv2d(in_channels=4, 
                          out_channels=16, 
                          kernel_size=(3, 2), 
                          stride=(1, 3), 
                          padding=(2, 3),
                          dilation=(1, 2), 
                          groups=2, 
                          bias=True, 
                          padding_mode='zeros')
m_torch.weight.data = torch.tensor(weight) # replace weight and bias
m_torch.bias.data = torch.tensor(bias)
x_torch = torch.tensor(data, requires_grad=True)
y_torch = m_torch(x_torch)
y_torch.backward(torch.tensor(grad))
```
Compare results:    
```python
print(np.allclose(y.detach().numpy(), y_torch.detach().numpy(), atol=1e-5, rtol=1e-5)) # output
# True

print(np.allclose(x.grad, x_torch.grad.detach().numpy(), atol=1e-5, rtol=1e-5)) # input grad
# True

print(np.allclose(m.weight.grad, m_torch.weight.grad.detach().numpy(), atol=1e-5, rtol=1e-3)) # weight grad
# True

print(np.allclose(m.bias.grad, m_torch.bias.grad.detach().numpy(), atol=1e-5, rtol=1e-3)) # bias grad
# True
```
Speed comparison with pytorch in GPU:  
```python
# cupy benchmark: https://docs.cupy.dev/en/stable/user_guide/performance.html
from cupyx.profiler import benchmark

m = m.cuda()
m_torch = m_torch.cuda()

def tortto_conv2d_gpu():
    x = tt.tensor(data, requires_grad=True).cuda()
    y = m(x)
    y.backward(tt.tensor(grad).cuda())

def torch_conv2d_gpu():
    x_torch = torch.tensor(data, requires_grad=True).cuda()
    y_torch = m_torch(x_torch)
    y_torch.backward(torch.tensor(grad).cuda())

print(benchmark(tortto_conv2d_gpu, (), n_repeat=50), '\n')
print(benchmark(torch_conv2d_gpu, (), n_repeat=50))
```
```python
tortto_conv2d_gpu   :    CPU:48152.444 us   +/-1573.327 (min:46592.200 / max:52764.700) us     GPU-0:48203.616 us   +/-1605.487 (min:46588.703 / max:52895.039) us 

torch_conv2d_gpu    :    CPU:24957.706 us   +/-1239.075 (min:23263.900 / max:29052.800) us     GPU-0:25032.415 us   +/-1243.638 (min:23319.712 / max:29204.128) us
```
As shown above, in this example, tortto `Conv2d` is about 2 times slower than pytorch.  
<br />
### [quick start: reverse prediction](https://github.com/samrere/pytortto/blob/main/examples/transformers/reverse_prediction.ipynb)
Next, Let's train a transformer encoder in 40 seconds, inspired from [UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html):  
The goal is to train a simple transformer encoder that can reverse the input sequence. i.e. if the input is `4,1,9,7,5,2,2`, the correct output would be `2,2,5,7,9,1,4`.  
<p align="left">
  <img src="https://github.com/samrere/pytortto/blob/main/img/sequence_reverse.png" width="250">
</p>


## Examples
**All examples trained on a Tesla T4 (16GB memory) GPU**  
**When comparing speed with pytorch, `torch.backends.cudnn.benchmark` is set to `False` for a fair comparison**
## Resnet  
* Trained on CIFAR10
* Each epoch = 45k train + 5k validation 
* Tested on full 10k test samples

model | test acc. | #.filters | n | epochs trained | speed (min/epoch) | pytorch speed (min/epoch) | speed comparison
--- | --- | --- | --- | --- | --- | --- | --- |
[small-preact_resnet110](https://github.com/samrere/pytortto/tree/main/examples/resnet/small_preact_resnet_110) | **94.08%**| (16,32,64) | basicBlock:(18,18,18) | 180 | 3.65 | 1.2 | 3.65/1.2=3.0
[preact_resnet18](https://github.com/samrere/pytortto/blob/main/examples/resnet/preact_resnet18) | **94.65%**| (64,128,256,512) | basicBlock:(2,2,2,2) | 180 | 2.0 | 0.75 | 2.0/0.75=2.7
[standard_resnet50 (finetune)](https://github.com/samrere/pytortto/blob/main/examples/resnet/resnet50_finetune) | **96.31%**| (64,128,256,512) | bottleNeck:(3,4,6,3) | 15 | 20.65 | 8.82 | 20.65/8.82=2.3
[preact_resnet101](https://github.com/samrere/pytortto/blob/main/examples/resnet/preact_resnet101) | **94.78%**| (64,128,256,512) | bottleNeck:(3,4,23,3) | 200 | 9.2 | 4.3 | 9.2/4.3=2.1
* `small_preact_resnet110`, `preact_resnet18` and `preact_resnet101` are trained from scratch. Kernel size of the first conv layer is 3, because CIFAR-10 images are 32x32 in size.
* `standard_resnet50` fintunes the full pretrained model. Kernel size of the first conv layer is 7. CIFAR-10 images are resized to 224x224 before feeding into the model.

<p align="center">
  <img src="https://github.com/samrere/pytortto/blob/main/img/cifar10.png" width="800">
</p>  

* Class activation map (CAM) of fintuned `standard_resnet50`  

<p align="center">
  <img src="https://github.com/samrere/pytortto/blob/main/img/cam.png" width="600">
</p>  

## UNet
* Trained on the carvana dataset
* GPU memory: 1.5GB
* Training time: 7 mins

model | image size | #.filters | batchsize | train/val/test | epochs
--- | --- | --- | --- | --- | --- |
[UNet](https://github.com/samrere/pytortto/tree/main/examples/unet) | resized to 3x64x64 | 32,64,128,256,512,256,128,64,32 | 32 | 3658/646/784 | 20 

<p align="center">
  <img src="https://github.com/samrere/pytortto/blob/main/img/carvana.png" width="600">
</p>

## Vision Transformer  
* Trained on CIFAR-10  
* Finetune: Each epoch = 4500 train + 500 validation  
* Tested on full 10k test samples

model | test acc. | layers | Hidden size | MLP size | Heads | epochs trained | speed (min/epoch) | pytorch speed (min/epoch) | speed comparison
--- | --- | --- | --- | --- | --- | --- | ---| ---| --- |
[ViT-B/16 (finetune)](https://github.com/samrere/pytortto/tree/main/examples/transformers/vision_transformer_finetune) | **97.42%**| 12 | 768 | 3072 | 12 | 15 | 4.2 | 2.7 | 4.2/2.7=1.56
* `ViT-B/16` Finetunes the full pretrained model. CIFAR-10 images are resized to 224x224 before feeding into the model.
<p align="center">
  <img src="https://github.com/samrere/pytortto/blob/main/img/attn.png" width="600">
</p>

## DCGAN
* Dataset adapted from https://github.com/bchao1/Anime-Face-Dataset
* Images resized to 3x64x64

model | epochs trained | speed (min/epoch) | pytorch speed (min/epoch) | speed comparison
--- | --- | ---| ---| --- |
[DCGAN](https://github.com/samrere/pytortto/tree/main/examples/gan/dcgan) | 100 | 6.27 | 2.38 | 6.27/2.38=2.6
<p align="center">
  <img src="https://github.com/samrere/pytortto/blob/main/img/fake.png" width="450">
</p>
<p align="center">
  <img src="https://github.com/samrere/pytortto/blob/main/img/transition.png" width="800">
</p>

## Supported functionalities
### Misc. 
#### variable functions
`manual_seed`, `ones`, `ones_like`, `zeros`, `zeros_like`, `randn`, `eye`, `arange`, `linspace`, `tensor`, ...   
 
### Autograd, forward and backward propagations of:
#### basic operations   
add: `+`, sub: `-`, mul: `*`, div: `/`, `sum`, `mean`, `var`      

#### ufunc  
`exp`, `log`, `sin`, `cos`, `tanh`, `sigmoid`, power: `x**2`, exponential: `2**x`,

#### tensor  
matmul: `@`, elementwise (Hadamard) mul: `*`, transpose: `.T` or `transpose(x, axes)`, concatenate: `cat`, view: `view`, slicing: `x[...,1:10:2,:]`, `flatten`, `swapaxis`, etc.       

#### nn.modules  
* activation: `Tanh`, `Sigmoid`, `LogSigmoid`, `ReLU`, `LeakyReLU`, `GELU`, `Softmax`, `LogSoftmax`, `MultiheadAttention`    
* batchnorm: `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d`   
* container: `Sequential`, `ModuleList`  
* conv: `Conv2d`  
* transposed conv: `ConvTranspose2d`
* dropout: `Dropout`    
* linear: `Linear`    
* loss: `MSELoss`, `BCELoss`, `BCEWithLogitsLoss`, `NLLLoss`  
* module: `Module`  
* pooling: `MaxPool2d`   
* sparse: `Embedding`  
* transformer: `TransformerEncoder`, `TransformerEncoderLayer` 

#### optim    
adapted from pytorch code, since it's already written in python.
* optimizer: `Optimizer`   
* sgd: `SGD`
* Adam: `Adam`  
* Adamw: `AdamW`  
* lr_scheduler: `LambdaLR`, `StepLR`, `MultiStepLR`, `CosineAnnealingLR`, ... 
