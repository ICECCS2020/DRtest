import sys
sys.path.append('../')
from tensorflow.python.platform import flags

from nmutant_attack.fgsm import fgsm
from nmutant_attack.cw import cw
from nmutant_attack.jsma import jsma
from nmutant_attack.blackbox import blackbox
from nmutant_data.mnist import data_mnist
import tensorflow as tf
from nmutant_data.data import get_data, get_shape
from nmutant_util.utils_file import get_data_file
import math
import os
from scipy.misc import imsave, imread
from pure import pure
import numpy as np

FLAGS = flags.FLAGS
def choose_data(attack='fgsm', datasets='mnist', total_num=10000, model_name='lenet1'):
    tf.set_random_seed(1234)
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    X_train, Y_train, X_test, Y_test = get_data(datasets)
    train_path='../adv_result/'+datasets+'/'+attack+'/'+model_name
    store_path_train='../adv_result/'+datasets+'/'+attack+'/'+model_name+'/train_data'
    store_path_test='../adv_result/'+datasets+'/'+attack+'/'+model_name+'/test_data'
    if not os.path.isdir(store_path_train):
        os.makedirs(store_path_train)
    if not os.path.isdir(store_path_test):       
        os.makedirs(store_path_test)
    if datasets=='cifar10':
      if attack=='fgsm':
          step_size=[0.01, 0.02, 0.03]

          for s in range(0,len(step_size)):

              samples_path=train_path+'/'+str(step_size[s])
              [image_list, image_files, real_labels, predicted_labels] = get_data_file(samples_path)
              ind=[]
              for i in range(len(image_list)):
                  ind.append(image_files[i].split('_')[0])

              for i in range(0, int(math.ceil(X_test.shape[0]/6))):
                  if str(i) in ind:
                      i_index=ind.index(str(i))
                      image_files[i_index]=str(step_size[s])+'_'+image_files[i_index]

                      test_p =store_path_test+'/'+image_files[i_index]
                      np.save(test_p, image_list[i_index])
              for i in range(int(math.ceil(X_test.shape[0]/6)), X_test.shape[0]):
                  if str(i) in ind:
                      i_index=ind.index(str(i))
                      image_files[i_index]=str(step_size[s])+'_'+image_files[i_index]

                      train_p =store_path_train+'/'+image_files[i_index]
                      np.save(train_p, image_list[i_index])
          '''
          for i in range(0, math.ceil(len(all_images)/6)):
              test_p =store_path_test+'/'+all_image_files[i]
              np.save(test_p, all_images[i])
          for i in range(math.ceil(len(all_images)/6), len(all_images)):
              train_p =store_path_train+'/'+all_image_files[i]
              np.save(train_p, all_images[i])
          '''
      if attack=='cw':
          targets=[0,1,2,3,4,5,6,7,8,9]
          cw_ini_cons=[0.1, 0.2, 0.3]
          for t in range(0,len(targets)):
              for c in range(0,len(cw_ini_cons)):
                  
                  samples_path=train_path+'/'+str(cw_ini_cons[c])+'_'+str(targets[t])
                  [image_list, image_files, real_labels, predicted_labels] = get_data_file(samples_path)
                  ind=[]
                  for i in range(len(image_list)):
                      ind.append(image_files[i].split('_')[0])

                  for i in range(1000*t, 1000*t+int(math.ceil(1000/6))):
                      if str(i) in ind:
                          i_index=ind.index(str(i))
                          image_files[i_index]=str(cw_ini_cons[c])+'_'+image_files[i_index]

                          test_p =store_path_test+'/'+image_files[i_index]
                          np.save(test_p, image_list[i_index])

                  for i in range(1000*t+int(math.ceil(1000/6)), 1000*(t+1)):
                      if str(i) in ind:
                          i_index=ind.index(str(i))
                          image_files[i_index]=str(cw_ini_cons[c])+'_'+image_files[i_index]

                          train_p =store_path_train+'/'+image_files[i_index]
                          np.save(train_p, image_list[i_index])
                  '''
                  
                  for i in range(0, math.ceil(X_test.shape[0]/(len(targets)*len(cw_ini_cons)))):
                      train_p =store_path_train+'/'+str(cw_ini_cons[c])+'_'+str(targets[t])+'_'+image_files[i]
                      #imsave(train_p, image_list[i])
                      np.save(train_p, image_list[i])
                      j=i+len(image_list)-math.ceil(X_test.shape[0]/(len(targets)*len(cw_ini_cons)))
                      #print(j)
                      test_p =store_path_test+'/'+str(cw_ini_cons[c])+'_'+str(targets[t])+'_'+image_files[j]
                      #imsave(test_p, image_list[j])
                      np.save(test_p, image_list[j])
                  '''
                      
      if attack=='jsma':
          targets=[0,1,2,3,4,5,6,7,8,9]
          jsma_var=[0.09, 0.1, 0.11]
          for t in range(0,len(targets)):
              for c in range(0,len(jsma_var)):
                  
                  samples_path=train_path+'/'+str(jsma_var[c])+'_'+str(targets[t])
                  [image_list, image_files, real_labels, predicted_labels] = get_data_file(samples_path)
                  ind=[]
                  for i in range(len(image_list)):
                      ind.append(image_files[i].split('_')[0])

                  for i in range(1000*t, 1000*t+int(math.ceil(1000/6))):
                      if str(i) in ind:
                          i_index=ind.index(str(i))
                          image_files[i_index]=str(jsma_var[c])+'_'+image_files[i_index]

                          test_p =store_path_test+'/'+image_files[i_index]
                          np.save(test_p, image_list[i_index])

                  for i in range(1000*t+int(math.ceil(1000/6)), 1000*(t+1)):
                      if str(i) in ind:
                          i_index=ind.index(str(i))
                          image_files[i_index]=str(jsma_var[c])+'_'+image_files[i_index]

                          train_p =store_path_train+'/'+image_files[i_index]
                          np.save(train_p, image_list[i_index])
                  '''
                  for i in range(0, math.ceil(X_test.shape[0]/(len(targets)*len(jsma_var)))):
                      train_p =store_path_train+'/'+str(jsma_var[c])+'_'+str(targets[t])+'_'+image_files[i]
                      #imsave(train_p, image_list[i])
                      np.save(train_p, image_list[i])
                      j=i+len(image_list)-math.ceil(X_test.shape[0]/(len(targets)*len(jsma_var)))
                      #print(j)
                      test_p =store_path_test+'/'+str(jsma_var[c])+'_'+str(targets[t])+'_'+image_files[j]
                      #imsave(test_p, image_list[j])
                      np.save(test_p, image_list[j]) 
                   '''
    if datasets=='mnist':
      if attack=='fgsm':
          step_size=[0.2, 0.3, 0.4]

          for s in range(0,len(step_size)):

              samples_path=train_path+'/'+str(step_size[s])
              [image_list, image_files, real_labels, predicted_labels] = get_data_file(samples_path)
              ind=[]
              for i in range(len(image_list)):
                  ind.append(image_files[i].split('_')[0])

              for i in range(0, int(math.ceil(X_test.shape[0]/7))):
                  if str(i) in ind:
                      i_index=ind.index(str(i))
                      image_files[i_index]=str(step_size[s])+'_'+image_files[i_index]

                      test_p =store_path_test+'/'+image_files[i_index]
                      np.save(test_p, image_list[i_index])
              for i in range(int(math.ceil(X_test.shape[0]/7)), X_test.shape[0]):
                  if str(i) in ind:
                      i_index=ind.index(str(i))
                      image_files[i_index]=str(step_size[s])+'_'+image_files[i_index]

                      train_p =store_path_train+'/'+image_files[i_index]
                      np.save(train_p, image_list[i_index])

      if attack=='cw':
          targets=[0,1,2,3,4,5,6,7,8,9]
          cw_ini_cons=[9, 10, 11]
          for t in range(0,len(targets)):
              for c in range(0,len(cw_ini_cons)):
                  
                  samples_path=train_path+'/'+str(cw_ini_cons[c])+'_'+str(targets[t])
                  [image_list, image_files, real_labels, predicted_labels] = get_data_file(samples_path)
                  ind=[]
                  for i in range(len(image_list)):
                      ind.append(image_files[i].split('_')[0])

                  for i in range(1000*t, 1000*t+int(math.ceil(1000/7))):
                      if str(i) in ind:
                          i_index=ind.index(str(i))
                          image_files[i_index]=str(cw_ini_cons[c])+'_'+image_files[i_index]

                          test_p =store_path_test+'/'+image_files[i_index]
                          np.save(test_p, image_list[i_index])

                  for i in range(1000*t+int(math.ceil(1000/7)), 1000*(t+1)):
                      if str(i) in ind:
                          i_index=ind.index(str(i))
                          image_files[i_index]=str(cw_ini_cons[c])+'_'+image_files[i_index]

                          train_p =store_path_train+'/'+image_files[i_index]
                          np.save(train_p, image_list[i_index])

      if attack=='jsma':
          targets=[0,1,2,3,4,5,6,7,8,9]
          jsma_var=[0.09, 0.1, 0.11]
          for t in range(0,len(targets)):
              for c in range(0,len(jsma_var)):
                  
                  samples_path=train_path+'/'+str(jsma_var[c])+'_'+str(targets[t])
                  [image_list, image_files, real_labels, predicted_labels] = get_data_file(samples_path)
                  ind=[]
                  for i in range(len(image_list)):
                      ind.append(image_files[i].split('_')[0])

                  for i in range(1000*t, 1000*t+int(math.ceil(1000/7))):
                      if str(i) in ind:
                          i_index=ind.index(str(i))
                          image_files[i_index]=str(jsma_var[c])+'_'+image_files[i_index]

                          test_p =store_path_test+'/'+image_files[i_index]
                          np.save(test_p, image_list[i_index])

                  for i in range(1000*t+int(math.ceil(1000/7)), 1000*(t+1)):
                      if str(i) in ind:
                          i_index=ind.index(str(i))
                          image_files[i_index]=str(jsma_var[c])+'_'+image_files[i_index]

                          train_p =store_path_train+'/'+image_files[i_index]
                          np.save(train_p, image_list[i_index])

			
def main(argv=None):
    datasets='mnist'
    attacks=['fgsm', 'cw', 'jsma']
    model_names=['googlenet12','googlenet16','googlenet22']
    for attack in attacks:
        for model_name in model_names:
            choose_data(attack=attack, datasets=datasets, model_name=model_name)
            #pure(datasets=datasets, attack=attack, model_name=model_name)
    #choose_test(datasets = FLAGS.datasets,
    #                attack=FLAGS.attack,
    #                model_name=FLAGS.model_name)
	

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'cifar10', 'The target datasets.')
    flags.DEFINE_string('attack', 'fgsm', 'attack_method')#'../mt_result/mnist_jsma/adv_jsma'
    flags.DEFINE_string('model_name', 'lenet1', 'model_name')

    tf.app.run()


			
			
			
			

