import sys
from os.path import expanduser

import tensorflow as tf
from tensorflow.python.platform import flags
import numpy as np
from sklearn import metrics
from ggplot import *
import csv
sys.path.append("../")

FLAGS = flags.FLAGS
def extract_mutation_number(csv_path):
    kappas = []
    with open(csv_path, 'rb') as f:
    	reader = csv.reader(f)
    	result_list = list(reader)

    count = 0
    for result in result_list:
    	kappas.append(float(result[3])/500)#mutation number
    	count = count + 1 #samples
    	if count>=500:
    		break
    return kappas, count

def roc_auc(mutation_result_path):
	normal_result_path = mutation_result_path+'/ori_result.csv'
	adv_result_path = mutation_result_path+'/adv_result.csv'


	print('--- Extracting result from : ', normal_result_path)
	[kappas_nor, len1] = extract_mutation_number(normal_result_path)

	print('--- Extracting result from : ', adv_result_path)
	[kappas_adv, len2] = extract_mutation_number(adv_result_path)

	kappas = kappas_nor + kappas_adv
	labels = len1*[0] + len2*[1]

	ge = 0
	for i in range(len2):
		if kappas_nor>kappas_adv:
			ge = ge + 1
	print('--- Ratio of indicator success: ', 1-ge/len2)

	fpr, tpr, _ = metrics.roc_curve(labels, kappas)
	# print(fpr,tpr)

	auc = metrics.auc(fpr,tpr)
	print('--- AUC: ', auc)

def main(argv=None):
	roc_auc(FLAGS.path)

if __name__ == '__main__':
	flags.DEFINE_string('path', '../results/fgsm/mnist/1', 'The target path.')

	tf.app.run()








