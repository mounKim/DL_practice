## Resnet

Resnet use residual(잔차) learning.
Network depth is crucial importance.
Visual recognition tasks are greatly benefited from deep models.
But vanishing/exploding gradients hamper convergence
and deeper network has higher training/test error.
To solve this problem, Resnet added identity mapping.

CIFAR-10 dataset consists of 50K training images
and 10K test images in 10 classes.
Network inputs are 32x32 images.
The first layer is 3x3 convolutions. Then resnet use a stack of 6n layers
with 3x3 convolutions on the feature map of size {32, 16, 8}
respectively. The number of filters are {16, 32, 64} respectively.

220515 : 81.93% at epoch 41<br>
220522 : 89.79% at epoch 195 (EarlyStopping/DataAugmentation 추가)<br>
