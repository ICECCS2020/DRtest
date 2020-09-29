import os
import numpy as np
import sys
sys.path.append("../")

for model in ['lenet1', 'lenet4', 'lenet5']:
    for attack in ['fgsm', 'cw', 'jsma']:
        for mu_var in ['gf', 'nai', 'ns', 'ws']:
            os.system('CUDA_VISIBLE_DEVICES=0 python retrain_mu_mnist.py --datasets=mnist --attack=' + attack + ' --model_name=' + model + ' --mu_var=' + mu_var + ' --epochs=50')