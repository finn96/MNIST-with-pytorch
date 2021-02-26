# MNIST Handwritten Digit Classification with Pytorch


## Thought Dump

I guess we need to start by finding the data. A quick google search finds it: http://yann.lecun.com/exdb/mnist/

I'm going to start by just downloading the labels data and see if I can play around with it.
`wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz`
`gunzip train-labels-idx1-ubyte.gz` 

So the data formatting is pretty clear from the website. The labels begin at the 3rd byte, and go until the end.

Alright. Parsing that was fairly easy. Let's download the images and see how that turns out.
`wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz`
`gunzip http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz`