from collect_gradients import collect_gradients
import tensorflow as tf

#models=['lenet1', 'lenet4', 'lenet5']
attacks=['fgsm', 'cw', 'jsma']
dataset='mnist'

for model in ['vgg13']:
    
    tf.reset_default_graph()
    save_path_ori='./lipschitz_mat/target/'+dataset+'/'+model+'/oritest'
    collect_gradients(dataset=dataset, model_name=model, numimg=1, firstimg=5, target_type=0b01111, save=save_path_ori, batch_size=512, epoch=49, Niters=50)
    '''
    for attack in attacks:
        tf.reset_default_graph()
        #save_path_attack='./lipschitz_mat/untarget/'+dataset+'/'+model+'/'+attack
        save_path_attack='./lipschitz_mat/untarget/'+dataset+'/'+model+'/'+attack
        collect_gradients(dataset=dataset, model_name=model, numimg=50, firstimg=0, de=True, attack=attack, target_type=16, save=save_path_attack, batch_size=512, epoch=99, Niters=50)
    '''
    
'''
for model in ['vgg13']:
    tf.reset_default_graph()
    save_path_ori='./lipschitz_mat/untarget/'+dataset+'/'+model+'/ori'
    collect_gradients(dataset=dataset, model_name=model, numimg=100, firstimg=100, target_type=16, save=save_path_ori)
'''