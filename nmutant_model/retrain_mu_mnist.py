from model_operation import model_training
import tensorflow as tf

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

# tf.reset_default_graph()
# print('rebuild')#rebuild
# datasets='mnist'
# models=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
# 	     'resnet152', 'googlenet12', 'googlenet16', 'googlenet22']
# attacks=['fgsm']
# mu_vars=['gf']
# for model in ['lenet5']:
#     for mu_var in mu_vars:
#         for attack in attacks:
#             tf.reset_default_graph()
#             samples_path='../adv_result/mu_'+datasets+'/'+mu_var+'/'+attack+'/'+model+'/train_data'
#             model_training(datasets=datasets, model_name = model, samples_path=samples_path,
#                              nb_epochs=50, batch_size=128, learning_rate=0.001, attack=attack, mu=True, mu_var=mu_var)


def main(argv=None):
    print('rebuild')  # rebuild

    tf.reset_default_graph()
    samples_path = '../adv_result/mu_' + FLAGS.datasets + '/' + FLAGS.mu_var + '/' + FLAGS.attack + '/' + FLAGS.model_name + '/train_data'
    model_training(datasets=FLAGS.datasets, model_name=FLAGS.model_name, samples_path=samples_path,
                   nb_epochs=FLAGS.epochs, batch_size=128, learning_rate=0.001, attack=FLAGS.attack, mu=True, mu_var=FLAGS.mu_var)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('attack', 'fgsm', 'attack_method')  # '../mt_result/mnist_jsma/adv_jsma'
    flags.DEFINE_string('model_name', 'lenet1', 'model_name')
    flags.DEFINE_string('mu_var', 'gf', '')
    flags.DEFINE_integer('epochs', 50, '')

    tf.app.run()