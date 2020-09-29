from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

sys.path.append("../")

from nmutant_model.model_operation import model_load, model_eval
from nmutant_data.data import get_data, get_shape
from nmutant_util.configs import path
from model_mutation.gf import gf
from model_mutation.nai import nai
from model_mutation.ns import ns
from model_mutation.ws import ws

FLAGS = flags.FLAGS
mm_dict = {"ns":ns, "ws":ws, "gf":gf, "nai":nai}

def main(argv=None):
    m_m = mm_dict[FLAGS.mm]
    mu_dir = os.path.join(path.mu_model_path, FLAGS.mm, FLAGS.datasets + '_' + FLAGS.model)
    i = 0
    while not os.path.exists(mu_dir):
        print(i)
        m_m(datasets=FLAGS.datasets,
            model_name=FLAGS.model,
            ration=FLAGS.ration,
            threshold=FLAGS.threshold,
            epoch = FLAGS.epoch)
        i += 1

if __name__ == '__main__':
    flags.DEFINE_string('mm', 'ns', 'model mutation')
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('model', 'lenet5', 'The name of model.')
    flags.DEFINE_float('ration', 0.01, 'The ration of mutated neurons.')
    flags.DEFINE_float('threshold', 0.95, 'The threshold of accuacy compared with original.')
    flags.DEFINE_integer('epoch', 99, 'epoch')

    tf.app.run()

