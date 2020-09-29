from model_operation import model_load,sub_model_load, submodel_training
import tensorflow as tf
import sys
sys.path.append("../")

from nmutant_util.utils import to_categorical, set_log_level, TemporaryLogLevel
from nmutant_util.utils_tf import model_train, model_eval, batch_eval
from nmutant_data.data import get_data, get_shape

submodel_training('mnist', 'sub', 'lenet1', batch_size=256,learning_rate=0.001, data_aug=6, lmbda=0.1,nb_epochs=10, holdout = 150)

