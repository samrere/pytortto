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
  - [YOLOv1](#YOLOv1)


## Intro
This is a pytorch style machine learning framework implemented entirely in numpy, with GPU acceleration from cupy.

Similar to the pokemon "ditto", pytortto works exactly like pytorch, although inferior in terms of speed. The purpose of this project is to understand how deep learning algorithms and frameworks like pytorch work under the hood. Max effort was given to correctness, calculation efficiency (like simpler Jacobian in logsoftmax, efficient implementation of convolution etc.), numerical stability (log-sum-exp used in logsigmoid, BCEWithLogitsLoss etc.), and memory efficiency (implementation of caching, view etc.).  

When computed in GPU, Tortto is around 1.5(vision transformers) ~ 3(CNNs) times slower than pytorch. It also achieves the same complexity as pytorch, which means tortto can be used to train relatively larger models such as `resnet101` and vision transformer `ViT-B/16` with the same speed ratio.

Tortto implements reverse mode automatic differentiation and supports dynamic computation graph like pytorch.

New in the latest version (1.3)  
* Computation graph now uses operations as nodes, instead of using tensors.
* Supports in-place operations.


## Installation
```python
pip install tortto
```

## Prerequisites
**numpy** (required): Only use its basic functions.  

**scipy** (optional): Can be installed to improve efficiency in some functions:  
* `scipy.special.erf` used in `nn.GELU`. approximation of erf is used when scipy is not installed.
* `scipy.sparse` used in `nn.Embedding` when setting `sparse=True`. Can't use sparse if neither scipy or cupy is installed.   

**cupy** (optional): Compute Numpy functions in GPU. Use `.cuda()`/`.cpu()` to send tensors or modules between CPU and GPU.  

**pytorch** (optional): Only use its dataset, dataloder and transforms, required in actual training to load and preprocess data. In each iteration, torch tensors (i.e. data, label) will be converted to tortto tensors by `tortto_tensor = tortto.tensor(torch_tensor.numpy()).cuda()`. All computations after that is done in tortto. See examples below.

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

in-place operations:
```python
import tortto as tt

x0 = tt.tensor([[-0.5722, -1.3110, -1.4200, -0.3545],
                [ 0.0945, -1.2108, -0.1059,  0.8041],
                [-0.5110,  1.4361, -1.1575, -0.7639]], requires_grad=True)
x = x0 + 0 # make it non-leaf for in-place operations
x[:, 2:] = tt.sign(x[:, 2:]) * tt.sqrt(tt.abs_(x[:, 2:]))
x.backward(x)

print(x._version)
# 2

print(x0.grad)
# [[-0.5722 -1.3110 -0.5000 -0.5000]
#  [ 0.0945 -1.2108 -0.5000  0.5000]
#  [-0.5110  1.4361 -0.5000 -0.5000]]
```

tortto also implements common Modules. Let's try `Conv2d` and compare its result and speed with pytorch:  
[conv2d comparison in GPU](https://github.com/samrere/pytortto/blob/main/examples/conv2d_result_speed_comparison.ipynb)  

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

## YOLOv1
[YOLO v1](https://github.com/samrere/pytortto/tree/main/examples/yolo) is work in progress...
