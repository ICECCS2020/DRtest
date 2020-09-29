import shutil
from tensorflow.python.platform import flags
import tensorflow as tf

FLAGS = flags.FLAGS

def main(argv=None):
    for i in range(FLAGS.start, FLAGS.end):
        shutil.rmtree("../de_models/" + FLAGS.attack + '/cifar10_' + FLAGS.model + "/" + str(i))

if __name__ == '__main__':
    flags.DEFINE_integer('start', 100, '')
    flags.DEFINE_integer('end', 500, '')
    flags.DEFINE_string('datasets', 'cifar10', 'The target datasets.')
    flags.DEFINE_string('model', 'vgg11', 'The name of model.')
    flags.DEFINE_string('attack', 'fgsm', 'The name of attack.')
    # flags.DEFINE_string('store_path', '../mt_result/integration/cw/mnist', 'The path to store adversaries.')

    tf.app.run()