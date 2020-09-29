from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

sys.path.append("../")

from nmutant_model.model_operation import model_load
from tensorflow.python.platform import flags
import re
import tensorflow as tf
import math
from collections import Counter

#FLAGS = flags.FLAGS

def local_lipschitz_constant(layer, sess):
    Lipschitz_layer=1.0
    if re.search("Conv2D", layer.name) is not None:
        np_kernel=layer.kernels.eval(session=sess)
        shapes=np_kernel.shape
        kernel_size=shapes[0]
        input_channels=shapes[2]
        output_channels=shapes[3]
        sum_conv_max=-1
        for output_ch in range(0,output_channels):
            sum=0
            for  k1 in range(0,kernel_size):
                for k2 in range(0,kernel_size):
                    for input_ch in range(0,input_channels):
                        sum+=abs(np_kernel[k1][k2][input_ch][output_ch])
            if sum>sum_conv_max:
                sum_conv_max=sum
        Lipschitz_layer=sum_conv_max
    if re.search("Linear", layer.name) is not None or re.search("logits", layer.name) is not None:
        np_W=layer.W.eval(session=sess)
        input_len=np_W.shape[0]
        output_len=np_W.shape[1]
        sum_linera_max=-1
        for j in range(output_len):
            sum=0
            for i in range(input_len):
                sum+=abs(np_W[i][j])
            if sum>sum_linera_max:
                sum_linera_max=sum
        Lipschitz_layer=sum_linera_max
    if re.search("ReLU", layer.name) is not None:
        Lipschitz_layer=1.0
    if re.search("BN", layer.name) is not None:
        gamma=layer.gamma.eval(session=sess)
        moving_variance=layer.moving_variance.eval(session=sess)
        max_BN=-1
        for i in range(gamma.shape[0]):
            Lipschitz_BN=abs(gamma[i]/math.pow(moving_variance[i]+layer.epsilon, 0.5))
            if Lipschitz_BN>max_BN:
                max_BN=Lipschitz_BN
        Lipschitz_layer=max_BN
    return Lipschitz_layer
    

def global_lipschitz_constant(datasets='cifar10', model_name='vgg11', de=False, attack='fgsm', epoch=49):
    Lipschitz_model=1.
    tf.reset_default_graph()
    sess, preds, x, y, model, feed_dict = model_load(datasets, model_name, de=de, attack=attack, epoch=epoch)
    layers=model.layers
    # Lipschitz_layer=0.
    if re.search("googlenet", model_name) is not None:
        inception_layers=[]
        other_layers=[]
        for layer in layers:
            search_inception_layer=re.search(".*_\d_\d", layer.name)
            if search_inception_layer is not None:
                inception_layers.append(layers.index(layer))
            else:
                other_layers.append(layers.index(layer))   
        inception_index=[]        
        for i in range(len(inception_layers)):
            if re.search("Conv2D\d_0_0", layers[inception_layers[i]].name) is not None:
                inception_index.append(i)
        for i in range(len(other_layers)):
            Lipschitz_model=Lipschitz_model*local_lipschitz_constant(layers[other_layers[i]], sess)
        for i in range(len(inception_index)):
            if i == len(inception_index)-1:
                inception_tail=len(inception_layers)
            else:
                inception_tail=inception_index[i+1]
            inception_num=[]
            for j in range(inception_index[i], inception_tail):
                string=layers[inception_layers[j]].name.split('_')
                inception_num.append(int(string[-2]))     
            result = Counter(inception_num)
            
            # before=0
            max_lip=-1
            for k in range(inception_num[-1]+1):
                before=0
                lipschitz_in=1.0
                for kk in range(0,k):
                    before+=result[kk]
                start=inception_index[i]+before
                for j in range(start, start+result[k]):
                    lipschitz_in=lipschitz_in*local_lipschitz_constant(layers[inception_layers[j]], sess)
                if lipschitz_in > max_lip:
                    max_lip=lipschitz_in
                    
            Lipschitz_model=Lipschitz_model*max_lip     
        sess.close()

    elif re.search("resnet", model_name) is not None:
        reslayers=[]
        other_layers=[]
        for layer in layers:
            search_bottlenecks=re.search("_", layer.name)
            if search_bottlenecks is None:
                other_layers.append(layers.index(layer))   
            else:
                reslayers.append(layers.index(layer))
                
        for i in range(len(other_layers)):
            Lipschitz_model=Lipschitz_model*local_lipschitz_constant(layers[other_layers[i]], sess)
            
        bottleneck_tail=[]        
        for i in range(len(reslayers)):
            if re.search(".*_\d_\d", layers[reslayers[i]].name) is None:
                bottleneck_tail.append(i)
                
        for i in range(len(bottleneck_tail)):
            if i == 0:
                bottleneck_start=0
            else:
                bottleneck_start=bottleneck_tail[i-1]+1
                
            is_con_short=[]
            for j in range(bottleneck_start, bottleneck_tail[i]):
                string=layers[reslayers[j]].name.split('_')
                is_con_short.append(int(string[-2]))     
            result = Counter(is_con_short)
            if len(result)==1:
                for j in range(bottleneck_start, bottleneck_tail[i]+1):
                    Lipschitz_model=Lipschitz_model*local_lipschitz_constant(layers[reslayers[j]], sess)
            else:
                lipschitz_layer_straight=1.0
                lipschitz_layer_short=1.0
                for j in range(bottleneck_start, bottleneck_start+result[0]):
                    lipschitz_layer_straight=lipschitz_layer_straight*local_lipschitz_constant(layers[reslayers[j]], sess)
                for j in range(bottleneck_start+result[0], bottleneck_tail[i]):
                    lipschitz_layer_short=lipschitz_layer_short*local_lipschitz_constant(layers[reslayers[j]], sess)
                lipschitz_shortcut=lipschitz_layer_straight+lipschitz_layer_short
                Lipschitz_model=Lipschitz_model*lipschitz_shortcut*local_lipschitz_constant(layers[reslayers[bottleneck_tail[i]]], sess)
                
        sess.close()

    else:
        for layer in layers:
            Lipschitz_model=Lipschitz_model*local_lipschitz_constant(layer, sess)
        sess.close()
    return Lipschitz_model

if __name__ == '__main__':
    f=open('result.txt', 'w')
    model_names=['vgg11']
    attacks=['fgsm', 'cw', 'jsma']
    for model_name in model_names:
        l=global_lipschitz_constant(datasets='mnist', model_name=model_name, epoch=49)

        print(l)
        f.write('Robustness value calculated use local Lipschitz constants of '+model_name+' is: '+str(l))
        f.write('\n')
        '''
        for attack in attacks:
            l=local_lipschitz(datasets='mnist', model_name=model_name, de=True, attack=attack, epoch=49)
            f.write('Robustness value calculated use local Lipschitz constants of retrained model: '+model_name+' with '+attack+' is: '+str(l))
            f.write('\n')
        '''
        





