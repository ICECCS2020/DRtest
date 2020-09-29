import sys
sys.path.append('../')

from nmutant_attack.fgsm import fgsm
from nmutant_attack.cw import cw
from nmutant_attack.jsma import jsma
from nmutant_attack.blackbox import blackbox
from nmutant_data.cifar10 import data_cifar10
import tensorflow as tf
from nmutant_data.data import get_shape
from nmutant_util.utils_imgproc import deprocess_image_1, preprocess_image_1, deprocess_image_1

datasets = 'cifar10'
num=10000
train_start = 0
train_end = 50000
test_start = 0
test_end = 10000

preprocess_image = preprocess_image_1
X_train, Y_train, X_test, Y_test = data_cifar10(train_start=train_start,
                                              train_end=train_end,
                                              test_start=test_start,
                                               test_end=test_end,
                                               preprocess=preprocess_image)
input_shape, nb_classes = get_shape(datasets)
sample = X_test[0:num]

models3=['lenet1','lenet4','lenet5']
models2=['vgg11', 'vgg13', 'vgg16', 'vgg19']
models1=['resnet18', 'resnet34', 'resnet50', 'resnet101',
	     'resnet152', 'googlenet12', 'googlenet16', 'googlenet22']
models=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
	     'resnet152', 'googlenet12', 'googlenet16', 'googlenet22']
#models=['lenet1', 'lenet4']
step_sizes=['0.01','0.02','0.03']
'''
f=open('bbox_mnist_output.txt', 'w')
for model in models3:
    for step in step_sizess:
        acc=0.0
        store_path = '../adv_result/'+datasets+'/bbox/'+model+'/'+ str(step)
        num_adv, total=blackbox(datasets, sample, model, 'sub', store_path, step, batch_size=128)
        acc=num_adv/total
        f.write(datasets+'_'+model+'_bbox accuracy for step size of '+str(step)+' is: '+ str(acc))
        f.write('\n')
f.close()
'''


f=open('fgsm_cifar_.txt', 'w')
for model in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet12', 'googlenet16', 'googlenet22']:
    for step in step_sizes:
        acc=0.0
        store_path = '../adv_result/'+datasets+'/fgsm/'+model+'/'+ step
        num_adv, total=fgsm(datasets, sample[0:10000], model, store_path, step, batch_size=64, epoch=9)
        acc=num_adv/total
        f.write(datasets+'_'+model+'_fgsm accuracy for step size of '+step+' is: '+ str(acc))
        f.write('\n')
f.close()



f=open('jsma_cifar.txt', 'w')
for model in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet12', 'googlenet16', 'googlenet22']:

    for gamma in [0.09, 0.1, 0.11]:
        for target in range(0, nb_classes):
            start=int(target*Y_test.shape[0]/nb_classes)
            end=int((target+1)*Y_test.shape[0]/nb_classes)
            #end=start+10
            acc=0.0
            store_path = '../adv_result/'+datasets+'/jsma/'+model+'/'+ str(gamma)+'_'+str(target)
            num_adv, total=jsma(datasets, sample, model, target, store_path, gamma, start, end, batch_size=128, epoch=9)
            acc=num_adv/total

            f.write(datasets+'_'+model+'_jsma accuracy for target class of '+str(target)+ 'and initial_constant: '+str(gamma)+' is: '+ str(acc))
            f.write('\n')
f.close()



f=open('cw_cifar_res.txt', 'w')
#for model in models1:
for model in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet12', 'googlenet16', 'googlenet22']:
    #for cw_ini_con in [8, 9, 10, 11, 12]:
    for cw_ini_con in [0.1, 0.2, 0.3]:
        for target in range(0, nb_classes):
            start=int(target*Y_test.shape[0]/nb_classes)
            end=int((target+1)*Y_test.shape[0]/nb_classes)
            #end=start+10
            acc=0.0
            store_path = '../adv_result/'+datasets+'/cw/'+model+'/'+ str(cw_ini_con)+'_'+str(target)
            num_adv, total=cw(datasets, sample, model, target, store_path, cw_ini_con, start, end, batch_size=64, epoch=9)
            acc=num_adv/total	
            f.write(datasets+'_'+model+'_cw accuracy for target class of '+str(target)+ 'and initial_constant: '+str(cw_ini_con)+' is: '+ str(acc))
            f.write('\n')

f.close() 




			
			
			
			

