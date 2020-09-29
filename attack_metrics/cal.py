from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import numpy as np
import tensorflow as tf

sys.path.append("../")

from attack_metrics.acac import acac
from attack_metrics.actc import actc
from attack_metrics.ald import ald
from attack_metrics.ass import ass
from attack_metrics.mr import mr
from attack_metrics.nte import nte
from attack_metrics.psd import psd
from attack_metrics.rgb import rgb
from defense_metrics.ccv import ccv
from defense_metrics.cos import cos
from defense_metrics.cr import cr


def cal(datasets, attacks, models, orimodel_epoch, de_epoch):
    datasets = datasets
    attack = attacks
    model = models
    # models=['lenet1', 'lenet4','lenet5']

    tf.reset_default_graph()
    f = open('result.txt', 'w')

    samples_path = '../adv_result/' + datasets + '/' + attack + '/' + model + '/train_data'
    # result_rgb=rgb(datasets, model, samples_path,'5', epoch=49)
    print(orimodel_epoch)
    result_acac = acac(datasets, model, samples_path, epoch=orimodel_epoch)
    result_actc = actc(datasets, model, samples_path, epoch=orimodel_epoch)

    result_ald_0 = ald(datasets, model, samples_path, '0', epoch=orimodel_epoch)
    result_ald_1 = ald(datasets, model, samples_path, '1', epoch=orimodel_epoch)
    result_ald_2 = ald(datasets, model, samples_path, '2', epoch=orimodel_epoch)
    result_ald_inf = ald(datasets, model, samples_path, 'inf', epoch=orimodel_epoch)
    result_ass = ass(datasets, model, samples_path)
    if datasets == 'mnist':
        if attack == 'fgsm':
            va = 0.3
        elif attack == 'cw':
            va = 10
        else:
            va = 0.1
    elif datasets == 'cifar10':
        if attack == 'fgsm':
            va = 0.01
        elif attack == 'cw':
            va = 0.2
        else:
            va = 0.1
    result_mr = mr(datasets, model, attack, va, epoch=orimodel_epoch)
    result_nte = nte(datasets, model, samples_path, epoch=orimodel_epoch)
    result_psd = psd(datasets, model, samples_path, 3)
    result_rgb = rgb(datasets, model, samples_path, '5', epoch=orimodel_epoch)

    result_ccv = ccv(datasets, model, model, attack=attack, epoch=orimodel_epoch, de_epoch=de_epoch)
    result_cos = cos(datasets, model, model, attack=attack, epoch=orimodel_epoch, de_epoch=de_epoch)
    result_crr, result_csr, result_cav = cr(datasets, model, attack=attack, epoch=orimodel_epoch, de_epoch=de_epoch)

    f.write(datasets + '_' + model + '_' + attack + ' metric is: ')
    f.write('\n')
    f.write('mr: ' + str(result_mr))
    f.write('\n')
    f.write('acac: ' + str(result_acac))
    f.write('\n')

    f.write('actc: ' + str(result_actc))
    f.write('\n')

    f.write('ald_0: ' + str(result_ald_0))
    f.write('\n')
    f.write('ald_1: ' + str(result_ald_1))
    f.write('\n')
    f.write('ald_2: ' + str(result_ald_2))
    f.write('\n')
    f.write('ald_inf: ' + str(result_ald_inf))
    f.write('\n')
    f.write('ass: ' + str(result_ass))
    f.write('\n')
    f.write('psd: ' + str(result_psd))
    f.write('\n')
    f.write('nte: ' + str(result_nte))
    f.write('\n')
    f.write('rgb: ' + str(result_rgb))
    f.write('\n')
    f.write(datasets + '_' + model + '_' + attack + ' adv_training denfense metric is: ')
    f.write('\n')
    f.write('ccv: ' + str(result_ccv))
    f.write('\n')
    f.write('cos: ' + str(result_cos))
    f.write('\n')
    f.write('crr: ' + str(result_crr))
    f.write('\n')
    f.write('csr: ' + str(result_csr))
    f.write('\n')
    f.write('cav: ' + str(result_cav))
    f.write('\n')

    f.close()


if __name__ == '__main__':
    cal('mnist', 'fgsm', 'vgg11', 49, 49)



