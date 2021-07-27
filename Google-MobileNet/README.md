# Google MobileNet
This is an implenmentation of [MobileNet V1](https://arxiv.org/pdf/1704.04861.pdf) using Keras.

MobileNet is a CNN network supposed to be efficient enough to work on mobile, thus the name. Its efficiency comes from replacing convolution blocks by depthwise separable convolution block: A depthwise convolution followed by a pointwise convolution to reduce the number of computations.

# Depth Wise
The depthwise convolution has a filter per input channel.
The number of input channels is thus the same as the number of output channels because the fact of there are no interactions between channels.
# Point Wise
Is a normal Convolution a filter per output channel.
The filter size is 1x1.

Return the interaction between channels.
# Refrences
* [Google MobileNet Paper](https://arxiv.org/pdf/1704.04861.pdf)
