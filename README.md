# This is a fork of caffe that has ROI convolution, which was described in [this paper](https://arxiv.org/abs/1801.02722):
```
End-to-end detection-segmentation network with ROI convolution,
Zichen Zhang, Min Tang, Dana Cobzas, Dornoosh Zonoobi, Martin Jagersand
ISBI 2018
```
The main ROI layers are:

* roi_conv_layer.cu
* roi_conv_layer.cpp
* roi_pooling_layers.cu
* roi_pooling_layers.cpp
* softmax_loss_roi_layer.cu
* softmax_loss_roi_layer.cpp

These additional changes were added on top of this commit: 
https://github.com/BVLC/caffe/tree/691febcb83d6a3147be8e9583c77aefaac9945f8

If you want to update the caffe in case of any cuda compatibility reasons, please
git merge the [official caffe repo](https://github.com/BVLC/caffe/) into this modified repo.

