from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import sys

import numpy as np
from six.moves import xrange
from tensorflow.python.platform import flags

sys.path.append("../")

from nmutant_util.utils import to_categorical, set_log_level, TemporaryLogLevel
from nmutant_util.utils_tf import model_train, model_eval, batch_eval
from nmutant_attack.attacks_tf import jacobian_graph, jacobian_augmentation
from nmutant_model.tutorial_models import *
from nmutant_data.data import get_data, get_shape
from nmutant_util.configs import path
from nmutant_util.utils_imgproc import preprocess_image_1
from nmutant_util.utils_file import get_data_file

FLAGS = flags.FLAGS

model_dict = {"sub":sub_model,
              "vgg11":VGG11, "vgg13":VGG13, "vgg16":VGG16, "vgg19":VGG19, "vgg_test":VGG_test, #"restnet_test": restnet_test,
              "lenet1":LeNet_1, "lenet4":LeNet_4, "lenet5":LeNet_5,
              "resnet18":ResNet18, "resnet34":ResNet34, "resnet50":ResNet50, "resnet101":ResNet101, "resnet152":ResNet152,
              "googlenet12":GoogleNet12, "googlenet16":GoogleNet16, "googlenet22":GoogleNet22}

def model_training(datasets, model_name, samples_path=None, nb_epochs=6, batch_size=256,learning_rate=0.001, attack=None, mu=False, mu_var='gf'):
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.9

    # Create TF session and set as Keras backend session
    sess = tf.Session(config=config)
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    X_train, Y_train, X_test, Y_test = get_data(datasets)

    def y_one_hot(label):
        y = np.zeros(10)
        y[label] = 1
        return y

    if samples_path != None:
        [image_list, image_files, real_labels, predicted_labels] = get_data_file(samples_path)
        #samples_adv = np.asarray([preprocess_image_1(image.astype('float64')) for image in image_list])
        samples_adv=np.asarray(image_list)
        #print(samples_adv.shape)
        labels_adv = np.asarray([y_one_hot(int(label)) for label in real_labels])
        samples = np.concatenate((X_train, samples_adv))
        #print(samples.shape)
        labels = np.concatenate((Y_train, labels_adv))
        if mu==True:
            model_path = path.mude_model_path + mu_var + '/' + attack + '/'+ datasets + "_" + model_name
        else:
            model_path = path.de_model_path + attack + '/'+ datasets + "_" + model_name

    else:
        samples = X_train
        labels = Y_train
        model_path = path.model_path + datasets + "_" + model_name

    input_shape, nb_classes = get_shape(datasets)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    feed_dict = None

    model = model_dict[model_name](input_shape, nb_classes)

    preds = model(x)
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': model_path,
        'filename': model_name + '.model'
    }

    sess.run(tf.global_variables_initializer())
    rng = np.random.RandomState([2017, 8, 30])
    model_train(sess, x, y, preds, samples, labels, args=train_params,
                rng=rng, save=True)

    # Evaluate the accuracy of the model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params, feed=feed_dict)
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    #accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params, feed=feed_dict)
    # Close TF session
    sess.close()
    print('Finish model training.')

def submodel_training(datasets, submodel_name, target_model, batch_size=256,learning_rate=0.001, data_aug=6, lmbda=0.1,nb_epochs=10, holdout = 150):
    # Set TF random seed to improve reproducibility
    
    tf.set_random_seed(1234)

    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True
    # Create TF session and set as Keras backend session
    sess = tf.Session(config=config)
    print("Created TensorFlow session.")
    
    set_log_level(logging.DEBUG)

    model_path = path.model_path + datasets + "_" + submodel_name + "_" + target_model

    X_train, Y_train, X_test, Y_test = get_data(datasets)
    input_shape, nb_classes = get_shape(datasets)

    # Define input TF placeholder
    sess, preds, x, y, model, feed_dict = model_load(datasets, target_model)
	
    rng = np.random.RandomState([2017, 8, 30])
    # Initialize substitute training set reserved for adversary
    X_sub = X_test[:holdout]
    Y_sub = np.argmax(Y_test[:holdout], axis=1)

    model_sub = model_dict[submodel_name](input_shape, nb_classes)
    preds_sub = model_sub(x)
    print("Defined TensorFlow model graph for the substitute.")

    # Define the Jacobian symbolically using TensorFlow
    grads = jacobian_graph(preds_sub, x, nb_classes)

    # Train the substitute and augment dataset alternatively
    for rho in xrange(data_aug):
        print("Substitute training epoch #" + str(rho))
        train_params = {
            'nb_epochs': nb_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'train_dir': model_path,
            'filename': 'substitute.model'
        }
        with TemporaryLogLevel(logging.WARNING, "nmutant_util.utils.tf"):
            model_train(sess, x, y, preds_sub, X_sub,
                        to_categorical(Y_sub, nb_classes),
                        init_all=False, args=train_params, rng=rng, save=True)

        # If we are not at last substitute training iteration, augment dataset
        if rho < data_aug - 1:
            print("Augmenting substitute training data.")
            # Perform the Jacobian augmentation
            lmbda_coef = 2 * int(int(rho / 3) != 0) - 1
            X_sub = jacobian_augmentation(sess, x, X_sub, Y_sub, grads,
                                          lmbda_coef * lmbda)

            print("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box
            Y_sub = np.hstack([Y_sub, Y_sub])
            X_sub_prev = X_sub[int(len(X_sub) / 2):]
            eval_params = {'batch_size': batch_size}
            bbox_val = batch_eval(sess, [x], [preds], [X_sub_prev],
                                  args=eval_params, feed=feed_dict)[0]
            # Note here that we take the argmax because the adversary
            # only has access to the label (not the probabilities) output
            # by the black-box model
            Y_sub[int(len(X_sub) / 2):] = np.argmax(bbox_val, axis=1)
    # Close TF session
    sess.close()
    print('Finish model training.')

def model_load(datasets, model_name, de=False, epoch=9, attack='fgsm', mu=False, mu_var='gf'):

    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True
    # Create TF session and set as Keras backend session
    sess = tf.Session(config=config)
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)
    input_shape, nb_classes = get_shape(datasets)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    feed_dict = None

    model = model_dict[model_name](input_shape, nb_classes, False)

    preds = model(x)
    print("Defined TensorFlow model graph.")
    
    if mu==False:
        if True == de:
            model_path = path.de_model_path + attack + '/' + datasets + "_" + model_name +  '/' + str(epoch) + '/' + model_name + '.model'
        else:
            model_path=path.model_path+datasets+'_'+model_name+'/'+str(epoch)+'/'+model_name+'.model'
    else:
        model_path=path.mu_model_path+mu_var+'/'+datasets+'_'+model_name+'/0/'+ datasets + "_" + model_name + '.model'

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    return sess, preds, x, y, model, feed_dict

def sub_model_load(sess, datasets, submodel_name, target_model, epoch='9'):
    # This is only useful for blackbox
    input_shape, nb_classes = get_shape(datasets)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    model_sub = model_dict[submodel_name](input_shape, nb_classes, False)

    preds_sub = model_sub(x)
    print("Defined TensorFlow model graph for the substitute.")

    # Train the substitute and augment dataset alternatively
    model_path = path.model_path + datasets + "_" + submodel_name + "_" + target_model + '/' + epoch + "/substitute.model"
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    return model_sub, preds_sub

def get_model_dict():
    return model_dict

def main(argv=None):
    model_training(datasets=FLAGS.datasets,
                   model_name = FLAGS.model,
                   #samples_path="../mt_result/integration/blackbox/mnist",
                   nb_epochs=FLAGS.nb_epochs,
                   batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The name of datasets')  #mnist
    flags.DEFINE_string('model', 'vgg11', 'The name of model')  # mnist
    flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model')#6
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    # flags.DEFINE_string('method', 'blackbox', 'The method to use')
    # flags.DEFINE_integer('data_aug', 6, 'Nb of substitute data augmentations')
    # flags.DEFINE_float('lmbda', 0.1, 'Lambda from arxiv.org/abs/1602.02697')
    # flags.DEFINE_integer('nb_epochs_s', 10, 'Training epochs for substitute')
    # flags.DEFINE_integer('holdout', 150, 'Test set holdout for adversary')

    tf.app.run()
