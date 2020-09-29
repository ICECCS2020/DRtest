from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

sys.path.append("../")

from nmutant_model.model_operation import model_load
from coverage_criteria.utils import init_coverage_tables, neuron_covered, update_coverage
from nmutant_util.utils_imgproc import preprocess_image_1
from nmutant_data.mnist import data_mnist
from nmutant_data.cifar10 import data_cifar10
from nmutant_data.svhn import data_svhn
from nmutant_util.utils_file import get_data_file
from neuron_coverage import neuron_coverage
from multi_testing_criteria import multi_testing_criteria

# models=['lenet1', 'lenet4', 'lenet5', 'sub', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#	     'resnet152', 'googlenet12', 'googlenet16', 'googlenet22']

models = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
datasets = 'cifar10'
# datasets='mnist'
attacks = ['fgsm']

f = open('cifar_coverage_vgg11fgsm.txt', 'w')
# f=open('test.txt', 'w')

for model in ['vgg11']:
    '''
    result_ori_test=neuron_coverage(datasets, model, 'test', epoch=484)
    f.write(datasets+' neuron coverage using X_test of original model: '+model+' is: '+str(result_ori_test))
    f.write('\n')
    '''
    for attack in attacks:
        samples_path = '../adv_result/' + datasets + '/' + attack + '/' + model + '/test_data'
        '''
        result_ori_attack=neuron_coverage(datasets, model, samples_path, epoch=484)
        f.write(datasets+' neuron coverage using combined '+attack+' examples of original model '+model+' is: '+str(result_ori_attack))
        f.write('\n')
        '''
        result_adv_test = neuron_coverage(datasets, model, 'test', de=True, attack=attack, epoch=99)
        f.write(
            datasets + ' neuron coverage using X_test and ' + attack + ' adversarial training model: ' + model + ' is: ' + str(
                result_adv_test))
        f.write('\n')

        result_adv_attack = neuron_coverage(datasets, model, samples_path, de=True, attack=attack, epoch=99)
        f.write(
            datasets + ' neuron coverage using combined ' + attack + ' examples and ' + attack + ' adversarial training model: ' + model + ' is: ' + str(
                result_adv_attack))
        f.write('\n')

        '''
        result_ori_justadv=neuron_coverage(datasets, model, samples_path, just_adv=True, epoch=49)
        f.write(datasets+' neuron coverage using just '+attack+' examples of original model '+model+' is: '+str(result_ori_justadv))
        f.write('\n')
        result_adv_justadv=neuron_coverage(datasets, model, samples_path, de=True, attack=attack, just_adv=True, epoch=49)
        f.write(datasets+' neuron coverage using just '+attack+' examples and '+attack+' adversarial training model: '+model+' is: '+str(result_adv_justadv))
        f.write('\n')
        '''
    '''
    kmn, nb, sna, tknc, tknp=multi_testing_criteria(datasets, model, 'test', epoch=484)
    f.write('datasets: '+datasets+' model: orignal '+model+' test_set: X_test:')
    f.write('\n')
    f.write('KMN: '+str(kmn))
    f.write('\n')
    f.write('NB: '+str(nb))
    f.write('\n')
    f.write('SNA: '+str(sna))
    f.write('\n')
    f.write('TKNC: '+str(tknc))
    f.write('\n')
    f.write('TKNP: '+str(tknp))
    f.write('\n')
    '''
    for attack in attacks:
        samples_path = '../adv_result/' + datasets + '/' + attack + '/' + model + '/test_data'
        '''
        kmn, nb, sna, tknc, tknp=multi_testing_criteria(datasets, model, samples_path, epoch=484)
        f.write('datasets: '+datasets+' model: orignal '+model+' test_set: combined '+attack+' adv example:')
        f.write('\n')
        f.write('KMN: '+str(kmn))
        f.write('\n')
        f.write('NB: '+str(nb))
        f.write('\n')
        f.write('SNA: '+str(sna))
        f.write('\n')
        f.write('TKNC: '+str(tknc))
        f.write('\n')
        f.write('TKNP: '+str(tknp))
        f.write('\n')
        '''
        kmn, nb, sna, tknc, tknp = multi_testing_criteria(datasets, model, 'test', de=True, attack=attack, epoch=99)
        f.write('datasets: ' + datasets + ' model: ' + attack + ' adv_training ' + model + ' test_set: X_test:')
        f.write('\n')
        f.write('KMN: ' + str(kmn))
        f.write('\n')
        f.write('NB: ' + str(nb))
        f.write('\n')
        f.write('SNA: ' + str(sna))
        f.write('\n')
        f.write('TKNC: ' + str(tknc))
        f.write('\n')
        f.write('TKNP: ' + str(tknp))
        f.write('\n')

        kmn, nb, sna, tknc, tknp = multi_testing_criteria(datasets, model, samples_path, de=True, attack=attack,
                                                          epoch=99)

        f.write(
            'datasets: ' + datasets + ' model: ' + attack + ' adv_training ' + model + ' test_set: combined ' + attack + ' adv example:')
        f.write('\n')
        f.write('KMN: ' + str(kmn))
        f.write('\n')
        f.write('NB: ' + str(nb))
        f.write('\n')
        f.write('SNA: ' + str(sna))
        f.write('\n')
        f.write('TKNC: ' + str(tknc))
        f.write('\n')
        f.write('TKNP: ' + str(tknp))
        f.write('\n')

        '''
        kmn, nb, sna, tknc, tknp=multi_testing_criteria(datasets, model, samples_path, just_adv=True, epoch=49)
        f.write('datasets: '+datasets+' model: orignal '+model+' test_set: just '+attack+' adv example:')
        f.write('\n')
        f.write('KMN: '+str(kmn))
        f.write('\n')
        f.write('NB: '+str(nb))
        f.write('\n')
        f.write('SNA: '+str(sna))
        f.write('\n')
        f.write('TKNC: '+str(tknc))
        f.write('\n')
        f.write('TKNP: '+str(tknp))
        f.write('\n')

        kmn, nb, sna, tknc, tknp=multi_testing_criteria(datasets, model, samples_path, de=True, attack=attack, just_adv=True, epoch=49)

        f.write('datasets: '+datasets+' model: '+attack+' adv_training '+model+' test_set: just '+attack+' adv example:')
        f.write('\n')
        f.write('KMN: '+str(kmn))
        f.write('\n')
        f.write('NB: '+str(nb))
        f.write('\n')
        f.write('SNA: '+str(sna))
        f.write('\n')
        f.write('TKNC: '+str(tknc))
        f.write('\n')
        f.write('TKNP: '+str(tknp))
        f.write('\n')
        '''

f.close()



