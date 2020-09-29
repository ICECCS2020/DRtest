from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

def preprocess_image_1(x):
    if len(x.shape) < 3:
        x = np.expand_dims(x, axis=2)
    x /= 255.0
    x = x.astype('float32')
    return x

def deprocess_image_1(x):
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    if x.shape[3] == 1:
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
    else:
        x = x.reshape(x.shape[1], x.shape[2], x.shape[3])
    return x

def preprocess_image_3(x):
    # Remove zero-center by mean pixel
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    x = x.astype('float32')
    return x

def deprocess_image_3(x):
    x = x.reshape((32, 32, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x