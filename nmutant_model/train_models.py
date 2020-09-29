from model_operation import model_training
import tensorflow as tf

#### train lenet
'''
tf.reset_default_graph() 
print('rebuild')#rebuild
print('training lenet1: ')

model_training(datasets='mnist',
                   model_name = 'lenet1',
                   nb_epochs=10,
                   batch_size=128,
                   learning_rate=0.001)


tf.reset_default_graph() 
print('rebuild')#rebuild
print('training lenet4: ')
model_training(datasets='mnist',
                   model_name = 'lenet4',
                   nb_epochs=10,
                   batch_size=128,
                   learning_rate=0.001)

tf.reset_default_graph() 
print('rebuild')#rebuild
print('training lenet5: ')
model_training(datasets='mnist',
                   model_name = 'lenet5',
                   nb_epochs=10,
                   batch_size=128,
                   learning_rate=0.001)
'''

'''
##train vgg
tf.reset_default_graph() 
print('rebuild')#rebuild
print('training vgg11: ')
model_training(datasets='cifar10',
                   model_name = 'vgg11',
                   nb_epochs=50,
                   batch_size=128,
                   learning_rate=0.001)

tf.reset_default_graph() 
print('rebuild')#rebuild
print('training vgg13: ')
model_training(datasets='cifar10',
                   model_name = 'vgg13',
                   nb_epochs=50,
                   batch_size=128,
                   learning_rate=0.001)
'''
'''
tf.reset_default_graph() 
print('rebuild')#rebuild
print('training vgg16: ')
model_training(datasets='cifar10',
                   model_name = 'vgg16',
                   nb_epochs=50,
                   batch_size=128,
                   learning_rate=0.001)

tf.reset_default_graph() 
print('rebuild')#rebuild
print('training vgg19: ')
model_training(datasets='cifar10',
                   model_name = 'vgg19',
                   nb_epochs=50,
                   batch_size=128,
                   learning_rate=0.001)
'''
'''
#resnet18
tf.reset_default_graph() 
print('rebuild')#rebuild
print('training resnet18: ')
model_training(datasets='cifar10',
                   model_name = 'resnet18',
                   nb_epochs=50,
                   batch_size=128,
                   learning_rate=0.001)

tf.reset_default_graph() 
print('rebuild')#rebuild
print('training resnet34: ')
model_training(datasets='cifar10',
                   model_name = 'resnet34',
                   nb_epochs=50,
                   batch_size=128,
                   learning_rate=0.001)

tf.reset_default_graph() 
print('rebuild')#rebuild
print('training resnet50: ')
model_training(datasets='cifar10',
                   model_name = 'resnet50',
                   nb_epochs=50,
                   batch_size=128,
                   learning_rate=0.001)

tf.reset_default_graph() 
print('rebuild')#rebuild
print('training resnet101: ')
model_training(datasets='cifar10',
                   model_name = 'resnet101',
                   nb_epochs=50,
                   batch_size=128,
                   learning_rate=0.001)
'''
'''
tf.reset_default_graph() 
print('rebuild')#rebuild
print('training resnet152: ')
model_training(datasets='cifar10',
                   model_name = 'resnet152',
                   nb_epochs=50,
                   batch_size=128,
                   learning_rate=0.001)

#googlenet
tf.reset_default_graph() 
print('rebuild')#rebuild
print('googlenet12: ')
model_training(datasets='cifar10',
                   model_name = 'googlenet12',
                   nb_epochs=50,
                   batch_size=128,
                   learning_rate=0.001)
'''

tf.reset_default_graph() 
print('rebuild')#rebuild
print('googlenet16: ')
model_training(datasets='cifar10',
                   model_name = 'googlenet16',
                   nb_epochs=50,
                   batch_size=128,
                   learning_rate=0.001)

tf.reset_default_graph() 
print('rebuild')#rebuild
print('googlenet22: ')
model_training(datasets='cifar10',
                   model_name = 'googlenet22',
                   nb_epochs=50,
                   batch_size=128,
                   learning_rate=0.001)


'''
#sub
tf.reset_default_graph() 
print('rebuild')#rebuild
print('sub: ')
model_training(datasets='mnist',
                   model_name = 'sub',
                   nb_epochs=10,
                   batch_size=128,
                   learning_rate=0.001)
'''