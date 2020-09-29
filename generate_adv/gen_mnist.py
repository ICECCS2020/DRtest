import sys
sys.path.append('../')

from nmutant_attack.fgsm import fgsm
from nmutant_attack.cw import cw
from nmutant_attack.jsma import jsma
from nmutant_attack.blackbox import blackbox
from nmutant_data.mnist import data_mnist
import tensorflow as tf
from nmutant_data.data import get_shape
datasets = 'mnist'
num=10000
train_start = 0
train_end = 60000
test_start = 0
test_end = 10000
X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                              train_end=train_end,
                                              test_start=test_start,
                                               test_end=test_end)
input_shape, nb_classes = get_shape(datasets)
sample = X_test[0:num]

models3=['lenet1','lenet4','lenet5']
models2=['googlenet12', 'googlenet16', 'googlenet22']
models1=['resnet18', 'resnet34', 'resnet50', 'resnet101',
	     'resnet152', 'googlenet12', 'googlenet16', 'googlenet22']
models=['lenet1', 'lenet4', 'lenet5', 'sub', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
	     'resnet152', 'googlenet12', 'googlenet16', 'googlenet22']
#models=['lenet1', 'lenet4']

step_sizes=['0.2','0.3','0.4']
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


f=open('fgsm_mnist.txt', 'w')
for model in models:
    for step in step_sizes:
        acc=0.0
        store_path = '../adv_result/'+datasets+'/fgsm/'+model+'/'+ step
        num_adv, total=fgsm(datasets, sample[0:10000], model, store_path, step, batch_size=256, epoch=99)
        acc=1.0*num_adv/total
        f.write(datasets+'_'+model+'_fgsm accuracy for step size of '+step+' is: '+ str(acc))
        f.write('\n')
f.close()


f=open('jsma_mnist.txt', 'w')
for model in models:
    for gamma in [0.09, 0.1, 0.11]:
        for target in range(0, nb_classes):
            start=int(target*Y_test.shape[0]/nb_classes)
            end=int((target+1)*Y_test.shape[0]/nb_classes)
            #end=start+100
            acc=0.0
            store_path = '../adv_result/'+datasets+'/jsma/'+model+'/'+ str(gamma)+'_'+str(target)
            num_adv, total=jsma(datasets, sample, model, target, store_path, gamma, start, end, batch_size=128, epoch=99)
            acc=1.0*num_adv/total

            f.write(datasets+'_'+model+'_jsma accuracy for target class of '+str(target)+ 'and initial_constant: '+str(gamma)+' is: '+ str(acc))
            f.write('\n')
f.close()


f=open('cw_mnist.txt', 'w')
#for model in models1:
for model in models:
    for cw_ini_con in [9, 10, 11]:
    #for cw_ini_con in [10]:
        for target in range(0, nb_classes):
            start=int(target*Y_test.shape[0]/nb_classes)
            end=int((target+1)*Y_test.shape[0]/nb_classes)
            #end=start+100
            acc=0.0
            store_path = '../adv_result/'+datasets+'/cw/'+model+'/'+ str(cw_ini_con)+'_'+str(target)
            num_adv, total=cw(datasets, sample, model, target, store_path, cw_ini_con, start, end, batch_size=128, epoch=99)
            acc=1.0*num_adv/total
            f.write(datasets+'_'+model+'_cw accuracy for target class of '+str(target)+ 'and initial_constant: '+str(cw_ini_con)+' is: '+ str(acc))
            f.write('\n')
f.close() 




			
			
			
			

