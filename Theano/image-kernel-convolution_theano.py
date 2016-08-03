# SIMPLE IMAGE-KERNEL (CONVOLUTION) OPERATION
# created for stackoverflow case: http://stackoverflow.com/questions/38144525/how-to-perform-stencil-computations-element-wise-on-a-matrix-in-theano/38196969#38196969s
# PLEASE READ THE QUESTION CAREFULLY
# THIS PROGRAM IS A SOLUTION OF THE QUESTION

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d

# original image 3D (3x3x4) (RGB Channel, height, width)
img = [[[1, 2, 3, 4],
       [1, 1, 3, 1],
       [1, 3, 1, 1]],

      [[2, 2, 3, 4],
       [2, 2, 3, 2],
       [2, 3, 2, 2]],

      [[3, 2, 3, 4],
       [3, 3, 3, 3],
       [3, 3, 3, 3]]]

# separate and reshape each channel to 4D 
# because convolution works on each channel only
# (from question's specificaton)
R = np.asarray([[img[0]]], dtype='float32')
G = np.asarray([[img[1]]], dtype='float32')
B = np.asarray([[img[2]]], dtype='float32')       

# 4D kernel from the original : [1,0,1]
# rotated because convolution works only on column
# (from question's specificaton)
kernel = np.asarray([[[[1],[0],[1]]]], dtype='float32')

# theano convolution
t_img = T.ftensor4("t_img")
t_kernel = T.ftensor4("t_kernel")
result = conv2d(
            input = t_img,
            filters=t_kernel,
            filter_shape=(1,1,1,3),
            border_mode = 'half')
f = theano.function([t_img,t_kernel],result)

# compute each channel
R = f(R,kernel)
G = f(G,kernel)
B = f(B,kernel)

# merge and reshape again
img = np.asarray([R,G,B])
img = np.reshape(img,(3,3,4))
print img