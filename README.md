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

benchmark : CIFAR-10<br>
220515 : val acc is 81.93% at epoch 41<br>
220522 : val acc is 89.79% at epoch 195 (Add EarlyStopping/DataAugmentation)<br>



## Vision Transformer

Transformers have become the model of choice in NLP. The dominant approach is to pretrain
on a large text corpus and fine-tune on a smaller task-specific dataset. Inspired by the 
Transformer`s successes in NLP, we experiment with applying a standard
Transformer directly to images, with the fewest possible modifications.
To do so, we split an image into patches and provide the sequence of linear embeddings 
of these patches as an input to a Transformer.
When pretrained on large amounts of data, Vision Transformer attains excellent results
compared to state-of-the-art convolutional networks.

benchmark : MNIST<br>
220526 : test acc is 94.58% after epoch 32
