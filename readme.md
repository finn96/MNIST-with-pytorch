# MNIST Handwritten Digit Classification with Pytorch

## TODO: 
 - Fix target size and output size difference
 - Try CrossEntropy Loss (maybe add an argmax layer)
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

## Update 02/26

I'm not going to make this notebook about getting cuda to work in WSL2, but NVIDIA's posts make this sound way easier than it is. Currently on 3rd install of cuda-toolkit in the subsystem. Probably won't work, but here's what I've learned so far

- Do not install ANY `nvidia-*` packages. There's going to be some display drivers in there and they're going to mess up the gpu virtualization. The only thing you should be installing is `cuda-toolkit-{major}-{minor}

- Seriously, just running `sudo apt remove nvidia-*` helped me make some serious progress.

- Run some basic sanity checks in powershell `nvidia-smi`, `nvcc` etc. Make sure that your GPU is correctly hooked up to the host. If it's not, you're going to have a bad time.
    - There's some sort of link between `C:\Windows\System32\lxss\` and the subsystem. But I haven't figured out exactly what it is yet. I think there's a link between here and the usr/lib driver in the subsystem.


Alright, so I started getting an error about cuda runtime version not matching cuda driver version. I noticed on my windows host that the cuda version was 11-3, but the latest cuda available for Ubuntu is 11.2. I uninstalled nvidia `FrameView` and am currently installing cuda_11.2.1 on the host. Let's see if this works. If I don't come back, it means  I lost my display

Got cuda on. It's slower.

We got something fairly reproducible now. After a bunch of tries, it seems we have a network that can somewhat reliably converge to a solution with 70-80% accuracy. It's a 3 layer network with a maxpool and dropout layer. Conv1d(1,1,3) -> Sigmoid -> max_pool1d -> Linear(391,100) -> Dropout -> -> Linear(100,10). I actually can't remember if the max_pool1d is better or worse. I need to check that again. Here's where the real breakthroughs came on re-producibility:
 - Switching the activation function from RELU to Sigmoid. I kept getting stuck in these local gradients. I THINK some gradients were just explodingly negative and I was getting dying RELU, based on how significant of a change this made. Don't get me wrong, occasionally this model would come out 80-85% accurate off only a few epochs. But 9/10 times it got stuck and thought everything was a 7.

 - Manually settting the initial weights of the two linear layers to uniform([-1/sqrt(n), 1/sqrt(n)]). I need to look up how torch normally sets the initial weights.

- The Dropout I'm not sure about. I need to test more. I know at one point it helped, but I've changed a lot since then and need to test

I compared a number of configurations. It turns out that maxpool_1d appears to be worse, with a test accuracy of about 6-8% lower. For the Linear Networks, I toyed with the shapes (792, 100),(100,10) and (792,80),(80,10). Both seemed to perform about equal. I chose 100 because it's square.

LOL: Linear(784, 100) -> Sigmoid -> Linear(100, 10) kicks ass. 92%. But no Conv :(

Okay takeway from the Linear experiment -- we don't need the drop. Getting 88% now on the conv(1,1,3) -> Sigmoid -> Linear(782, 100) -> Sigmoid -> Linear(100,10)

uhh wrong name 3