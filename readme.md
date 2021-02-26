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

Unzipped, all the images are only about 45mb. Not too bad in the grand scheme of things

Alright. This byte schema gets a little more compicated, but still doesn't look so bad. The first 4bytes are the magic number, the second 4bytes is the number of images, the third 4bytse is the number of rows per image, and the fourth 4bytes is the number of columns per image. After that, each byte is a pixel.

I can read from the docs that "Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black)." So basically the 
psuedo code of what I need to do:

for _ in num_images:
    image = []
    for row in num_rows:
       pixel_data_for_row = read(num_cols)
       image.append(pixel_data_for_row)

Let's see how this goes.

Okay so the classic one from the pytorch tutorial has a whole bunch of layers. Plus, it looks in 2 dimensions. Let's see how accurate we can get it if we flatten each image to a 1d 784 vector. With only 1 conv layer, maybe 1 or 3 fc layers 


wget `http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz`
wget `http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz`