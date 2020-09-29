"""
A pure TensorFlow implementation of a neural network. This can be
used as a drop-in replacement for a Keras model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from nmutant_model.network import *

def make_basic_cnn(input_shape=(None, 28, 28, 1), nb_classes=10):
    layers = [Conv2D(64, (8, 8), (2, 2), "SAME"),
              ReLU(),
              Conv2D(128, (6, 6), (2, 2), "VALID"),
              ReLU(),
              Conv2D(128, (5, 5), (1, 1), "VALID"),
              ReLU(),
              Flatten(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model

def make_basic_cnn_cifar10(input_shape=(None, 32, 32, 3), nb_classes=10):
    layers = [Conv2D(int(32), (5, 5), (1, 1), "SAME"),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "SAME"),
              Conv2D(64, (5, 5), (1, 1), "SAME"),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "SAME"),
              Flatten(),
              Linear(1024),
              ReLU(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model

def make_better_cnn_cifar10(keep_prob1, keep_prob2, input_shape=(None, 32, 32, 3), nb_classes=10):
    layers = [Conv2D(96, (3, 3), (1, 1), "SAME"),
              ReLU(),
              Dropout(keep_prob1),
              Conv2D(96, (3, 3), (1, 1), "SAME"),
              ReLU(),
              Conv2D(96, (3, 3), (2, 2), "SAME"),
              ReLU(),
              Dropout(keep_prob2),
              Conv2D(192, (3, 3), (1, 1), "SAME"),
              ReLU(),
              Conv2D(192, (3, 3), (1, 1), "SAME"),
              ReLU(),
              Conv2D(192, (3, 3), (2, 2), "SAME"),
              ReLU(),
              Dropout(keep_prob2),
              Conv2D(192, (3, 3), (1, 1), "SAME"),
              ReLU(),
              Conv2D(192, (1, 1), (1, 1), "VALID"),
              ReLU(),
              Conv2D(nb_classes, (1, 1), (1, 1), "VALID"),
              AvgPooling((8,8), (8,8), "VALID"),#GlobalAveragePooling2D()
              Flatten(),
              Softmax()]

    model = MLP(layers, input_shape)
    return model

def sub_model(input_shape=(None, 32, 32, 3), nb_classes=10, training=True):
    # Define a fully connected model (it's different than the black-box)
    layers = [Flatten(),
              Linear(200),
              ReLU(),
              Linear(200),
              ReLU(),
              Linear(nb_classes),
              Softmax()]

    return MLP(layers, input_shape)

def mnist_das_cnn(keep_prob1, input_shape=(None, 28, 28, 1), nb_classes=10):
    layers = [Conv2D(64, (3, 3), (1, 1), "VALID"),
              ReLU(),
              Conv2D(64, (3, 3), (1, 1), "VALID"),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "VALID"),
              Dropout(keep_prob1),
              Flatten(),
              Linear(128),
              ReLU(),
              Dropout(keep_prob1),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model

def cifar10_das_cnn(keep_prob1, input_shape=(None, 32, 32, 3), nb_classes=10):
    layers = [Conv2D(32, (3,3), (1,1), "SAME"),
              ReLU(),
              Conv2D(32, (3, 3), (1, 1), "SAME"),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "VALID"),
              Conv2D(64, (3, 3), (1, 1), "SAME"),
              ReLU(),
              Conv2D(64, (3, 3), (1, 1), "SAME"),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "VALID"),
              Conv2D(128, (3, 3), (1, 1), "SAME"),
              ReLU(),
              Conv2D(128, (3, 3), (1, 1), "SAME"),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "VALID"),
              Flatten(),
              Dropout(keep_prob1),
              # Dense(1024),
              Linear(1024),
              ReLU(),
              Dropout(keep_prob1),
              # Dense(512),
              Linear(512),
              ReLU(),
              Dropout(keep_prob1),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model

def svnh_das_cnn(keep_prob1, input_shape=(None, 32, 32, 3), nb_classes=10):
    layers = [Conv2D(64, (3,3), (1,1), "VALID"),
              ReLU(),
              Conv2D(64, (3, 3), (1, 1), "VALID"),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "VALID"),
              Dropout(keep_prob1),
              Flatten(),
              Linear(512),
              ReLU(),
              Dropout(keep_prob1),
              Linear(128),
              ReLU(),
              Dropout(keep_prob1),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model

def LeNet_1(input_shape=(None, 28, 28, 1), nb_classes=10, training=True):
    layers = [Conv2D(4, (5, 5), (1, 1), "SAME"),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "VALID"),
              Conv2D(12, (5, 5), (1, 1), "SAME"),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "VALID"),
              Flatten(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model

def LeNet_4(input_shape=(None, 28, 28, 1), nb_classes=10, training=True):
    layers = [Conv2D(6, (5, 5), (1, 1), "SAME"),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "VALID"),
              Conv2D(16, (5, 5), (1, 1), "SAME"),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "VALID"),
              Flatten(),
              Linear(84),
              ReLU(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model

def LeNet_5(input_shape=(None, 28, 28, 1), nb_classes=10, training=True):
    layers = [Conv2D(6, (5, 5), (1, 1), "SAME"),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "VALID"),
              Conv2D(16, (5, 5), (1, 1), "SAME"),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "VALID"),
              Flatten(),
              Linear(120),
              ReLU(),
              Linear(84),
              ReLU(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model

def VGG_test(input_shape=(None, 32, 32, 3), nb_classes=10, training=True):
    layers = [Conv2D(64, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              MaxPooling((2,2),(2,2), "SAME"),
              Flatten(),
              Linear(200),
              ReLU(),
              Linear(100),
              ReLU(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model

def VGG11(input_shape=(None, 32, 32, 3), nb_classes=10, training=True):
    layers = [Conv2D(64, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              MaxPooling((2,2),(2,2), "SAME"),
              Conv2D(128, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "SAME"),
              Conv2D(256, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(256, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "SAME"),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "SAME"),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "SAME"),
              Flatten(),
              Linear(4096),
              ReLU(),
              Linear(4096),
              ReLU(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model

def VGG13(input_shape=(None, 32, 32, 3), nb_classes=10, training=True):
    layers = [Conv2D(64, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(64, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              MaxPooling((2,2),(2,2), "SAME"),
              Conv2D(128, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(128, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "SAME"),
              Conv2D(256, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(256, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "SAME"),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "SAME"),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "SAME"),
              Flatten(),
              Linear(4096),
              ReLU(),
              Linear(4096),
              ReLU(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model

def VGG16(input_shape=(None, 32, 32, 3), nb_classes=10, training=True):
    layers = [Conv2D(64, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(64, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              MaxPooling((2,2),(2,2), "SAME"),
              Conv2D(128, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(128, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "SAME"),
              Conv2D(256, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(256, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(256, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "SAME"),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "SAME"),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "SAME"),
              Flatten(),
              Linear(4096),
              ReLU(),
              Linear(4096),
              ReLU(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model

def VGG19(input_shape=(None, 32, 32, 3), nb_classes=10, training=True):
    layers = [Conv2D(64, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(64, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              MaxPooling((2,2),(2,2), "SAME"),
              Conv2D(128, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(128, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "SAME"),
              Conv2D(256, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(256, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(256, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(256, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "SAME"),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "SAME"),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              Conv2D(512, (3, 3), (1, 1), "SAME"),
              BN(training),
              ReLU(),
              MaxPooling((2, 2), (2, 2), "SAME"),
              Flatten(),
              Linear(4096),
              ReLU(),
              Linear(4096),
              ReLU(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model

def block(in_channel, out_channel, num, block_size, training, stride=2):
    layers = []
    layers.append(bottleneck(in_channel, out_channel, block_size, training, stride))
    for i in range(1, num):
        layers.append(bottleneck(out_channel, out_channel, block_size, training))
    return layers

def bottleneck(in_channel, out_channel, block_size, training, stride=None):
    if stride is None:
        stride = 1 if in_channel == out_channel else 2
    layers = []
    if block_size == 3:
        l1 = []
        l1.append(Conv2D(out_channel // 4,(1,1,in_channel), (stride,stride),"SAME"))
        l1.append(BN(training))
        l1.append(ReLU())
        l1.append(Conv2D(out_channel // 4,(3,3), (1,1),"SAME"))
        l1.append(BN(training))
        l1.append(ReLU())
        l1.append(Conv2D(out_channel,(1,1), (1,1),"SAME"))
        l1.append(BN(training))
        layers.append(l1)
    elif block_size == 2:
        l1 = []
        l1.append(Conv2D(out_channel,(3, 3,in_channel), (stride,stride),"SAME"))
        l1.append(BN(training))
        l1.append(ReLU())
        l1.append(Conv2D(out_channel, (3, 3), (1, 1), "SAME"))
        l1.append(BN(training))
        layers.append(l1)
    if stride != 1 or in_channel != out_channel:
        l2 = []
        l2.append(Conv2D(out_channel,(1,1,in_channel), (stride,stride),"SAME"))
        l2.append(BN(training))
        layers.append(l2)
    else:
        layers.append([])
    layers.append(ReLU())
    return layers

def ResNet18(input_shape=(None, 28, 28, 1), nb_classes=10, training=True):
    layers = [Conv2D(64, (3,3), (1,1), "SAME"),
              BN(training),
              ReLU()]
    layers = layers + block(64, 64, 2, 2, training)
    layers = layers + block(64, 128, 2, 2, training)
    layers = layers + block(128, 256, 2, 2, training)
    layers = layers + block(256, 512, 2, 2, training)
    layers.append(AvgPooling((2, 2), (2, 2), "VALID"))
    layers.append(Flatten())
    layers.append(Linear(nb_classes))
    layers.append(Softmax())

    model = ResidualModel(layers, input_shape)
    return model

def ResNet34(input_shape=(None, 28, 28, 1), nb_classes=10, training=True):
    layers = [Conv2D(64, (3,3), (1,1), "SAME"),
              BN(training),
              ReLU()]
    layers = layers + block(64, 64, 3, 2, training)
    layers = layers + block(64, 128, 4, 2, training)
    layers = layers + block(128, 256, 6, 2, training)
    layers = layers + block(256, 512, 3, 2, training)
    layers.append(AvgPooling((2, 2), (2, 2), "VALID"))
    layers.append(Flatten())
    layers.append(Linear(nb_classes))
    layers.append(Softmax())

    model = ResidualModel(layers, input_shape)
    return model

def ResNet50(input_shape=(None, 28, 28, 1), nb_classes=10, training=True):
    layers = [Conv2D(64, (3,3), (1,1), "SAME"),
              BN(training),
              ReLU()]
    layers = layers + block(64, 256, 3, 3, training)
    layers = layers + block(256, 512, 4, 3, training)
    layers = layers + block(512, 1024, 6, 3, training)
    layers = layers + block(1024, 2048, 3, 3, training)
    layers.append(AvgPooling((2, 2), (2, 2), "VALID"))
    layers.append(Flatten())
    layers.append(Linear(nb_classes))
    layers.append(Softmax())

    model = ResidualModel(layers, input_shape)
    return model

def ResNet101(input_shape=(None, 28, 28, 1), nb_classes=10, training=True):
    layers = [Conv2D(64, (3,3), (1,1), "SAME"),
              BN(training),
              ReLU()]
    layers = layers + block(64, 256, 3, 3, training)
    layers = layers + block(256, 512, 4, 3, training)
    layers = layers + block(512, 1024, 23, 3, training)
    layers = layers + block(1024, 2048, 3, 3, training)
    layers.append(AvgPooling((2, 2), (2, 2), "VALID"))
    layers.append(Flatten())
    layers.append(Linear(nb_classes))
    layers.append(Softmax())

    model = ResidualModel(layers, input_shape)
    return model

def ResNet152(input_shape=(None, 28, 28, 1), nb_classes=10, training=True):
    layers = [Conv2D(64, (3,3), (1,1), "SAME"),
              BN(training),
              ReLU()]
    layers = layers + block(64, 256, 3, 3, training)
    layers = layers + block(256, 512, 8, 3, training)
    layers = layers + block(512, 1024, 36, 3, training)
    layers = layers + block(1024, 2048, 3, 3, training)
    layers.append(AvgPooling((2, 2), (2, 2), "VALID"))
    layers.append(Flatten())
    layers.append(Linear(nb_classes))
    layers.append(Softmax())

    model = ResidualModel(layers, input_shape)
    return model

def inception(n1x1, n3x3reduce, n3x3, n5x5reduce, n5x5, pool_planes, training):
    layer = []
    l1 = [Conv2D(n1x1, (1,1), (1,1), "SAME"), BN(training), ReLU()]
    layer.append(l1)
    l2 = [Conv2D(n3x3reduce, (1,1), (1,1), "SAME"), BN(training), ReLU(), Conv2D(n3x3, (3,3), (1,1), "SAME"), BN(training), ReLU()]
    layer.append(l2)
    l3 = [Conv2D(n5x5reduce, (1,1), (1,1), "SAME"), BN(training), ReLU(), Conv2D(n5x5, (5,5), (1,1), "SAME"), BN(training), ReLU()]
    layer.append(l3)
    l4 = [MaxPooling((3, 3), (1,1), "SAME"), Conv2D(pool_planes, (1,1), (1,1), "SAME"), BN(training), ReLU()]
    layer.append(l4)
    return layer

def GoogleNet12(input_shape=(None, 28, 28, 1), nb_classes=10, training=True):

    layers = [Conv2D(192, (3, 3), (1, 1), "SAME"), BN(training), ReLU()]
    layers.append(inception(64, 96, 128, 16, 32, 32, training))
    layers.append(inception(128, 128, 192, 32, 96, 64, training))
    layers.append(MaxPooling((3, 3), (2,2), "SAME"))
    layers.append(inception(192,  96, 208, 16,  48,  64, training))
    layers.append(AvgPooling((5, 5), (3, 3), "VALID"))
    layers.append(Conv2D(256, (1,1),(1,1),"SAME"))
    layers.append(BN(training))
    layers.append(ReLU())
    layers.append(Flatten())
    layers.append(Linear(1024))
    layers.append(Linear(nb_classes))
    layers.append(Softmax())

    model = EnsembleModel(layers, input_shape)
    return model

def GoogleNet16(input_shape=(None, 28, 28, 1), nb_classes=10, training=True):

    layers = [Conv2D(192, (3, 3), (1, 1), "SAME"), BN(training), ReLU()]
    layers.append(inception(64, 96, 128, 16, 32, 32, training))
    layers.append(inception(128, 128, 192, 32, 96, 64, training))
    layers.append(MaxPooling((3, 3), (2, 2), "SAME"))
    layers.append(inception(192, 96, 208, 16, 48, 64, training))
    layers.append(inception(160, 112, 224, 24, 64, 64, training))
    layers.append(inception(128, 128, 256, 24, 64, 64, training))
    layers.append(AvgPooling((5, 5), (3, 3), "VALID"))
    layers.append(Conv2D(256, (1, 1), (1, 1), "SAME"))
    layers.append(BN(training))
    layers.append(ReLU())
    layers.append(Flatten())
    layers.append(Linear(1024))
    layers.append(Linear(nb_classes))
    layers.append(Softmax())

    model = EnsembleModel(layers, input_shape)
    return model

def GoogleNet22(input_shape=(None, 28, 28, 1), nb_classes=10, training=True):

    layers = [Conv2D(192, (3, 3), (1, 1), "SAME"), BN(training), ReLU()]
    layers.append(inception(64, 96, 128, 16, 32, 32, training))
    layers.append(inception(128, 128, 192, 32, 96, 64, training))
    layers.append(MaxPooling((3, 3), (2,2), "SAME"))
    layers.append(inception(192,  96, 208, 16,  48,  64, training))
    layers.append(inception(160, 112, 224, 24,  64,  64, training))
    layers.append(inception(128, 128, 256, 24,  64,  64, training))
    layers.append(inception(112, 144, 288, 32,  64,  64, training))
    layers.append(inception(256, 160, 320, 32, 128, 128, training))
    layers.append(MaxPooling((3, 3), (2, 2), "SAME"))
    layers.append(inception(256, 160, 320, 32, 128, 128, training))
    layers.append(inception(384, 192, 384, 48, 128, 128, training))
    layers.append(AvgPooling((7,7), (1,1), "VALID"))
    layers.append(Flatten())
    layers.append(Linear(nb_classes))
    layers.append(Softmax())

    model = EnsembleModel(layers, input_shape)
    return model
