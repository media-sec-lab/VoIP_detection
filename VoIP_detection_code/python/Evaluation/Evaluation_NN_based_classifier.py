# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:21:18 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 08:47:29 2019

@author: waiting
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
"""
Created on Mon Aug 28 19:57:15 2017

@author: huangyuankun
"""

import tensorflow as tf
#import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score
import numpy as np
#import h5py
#import math
import random
extracted_feature_path=('/VoIP_detection/VoIP_detection_code/python/classification/predict')
model_save_path=('/VoIP_detection/VoIP_detection_code/python/classification/DNN')

load_name=extracted_feature_path+'/train.mat'
load_name2=extracted_feature_path+'/val.mat'
load_name3=extracted_feature_path+'/mismatch.mat'
model_path=model_save_path+'/save_model/0.0002/save1_met.ckpt'


load_data=sio.loadmat(load_name)
load_data2=sio.loadmat(load_name2)
load_data3=sio.loadmat(load_name3)


mismatch_x=load_data3['test_x']
mismatch_y=load_data3['test_y']

nmismatch=mismatch_x.shape[0]
dim=mismatch_x.shape[1]
n_class=mismatch_y.shape[1] #output unit

learning_rate_v = [0.0002]
display_step = 1
display_ites=1
training_epochs = 50
#network parameters
batch_size1=1
batch_size2=25


val_acc_max=0.0
val_acc_max_epoch=0
val_tpr_max=0.0
val_tnr_max=0.0


mismatch_acc_max=0.0
mismatch_tpr_max=0.0
mismatch_tnr_max=0.0

with tf.name_scope('input'):
    x = tf.placeholder("float32",[None,dim],name='x_input')
    y = tf.placeholder("float32",[None, n_class],name='y_input')
    learning_rate=tf.placeholder("float32",name='learning_rate')
    is_training2=tf.placeholder(tf.int32)
    
    
def weight_variable(shape,variable_name):
    initial=tf.Variable(tf.truncated_normal(shape,mean=0,stddev=0.01),name=variable_name,dtype=tf.float32)
    tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(0.0001)(initial))
    return initial
    

def bias_variable(shape,in_name):
    #initial=tf.constant(0.1,shape=shape)
    return tf.Variable(tf.zeros(shape)+0.1,name=in_name)

    
    
def nn_layer3(input_tensor, input_dim, output_dim,in_name,act=None):
    w_name=in_name+'_weight'
    b_name=in_name+'_bias'
    weights=weight_variable([input_dim,output_dim],w_name)
    biases =bias_variable([output_dim],b_name)
    if act is None:
        preactivate=tf.matmul(input_tensor,weights)+biases
    else:
        preactivate=act(tf.matmul(input_tensor,weights)+biases)
    return preactivate


def CNN_1d(x, scope):   
    with tf.name_scope(scope):
        out_all_conv_final=nn_layer3(x,768*3,768,'all_1')
        pred_in_final=out_all_conv_final
        in_all_final=nn_layer3(pred_in_final,768,n_class,'all_connet1')
        out=tf.nn.softmax(in_all_final)  
        return out
    
    
x_input=x
print(x_input.shape)
pred=CNN_1d(x_input,scope='final_all_1D_CNN')

print('pred')
with tf.name_scope('cross_entropy'):
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred+1e-10),reduction_indices=[1]))
    #cost=cost1
tf.summary.scalar('cross_entropy',cost)

with tf.name_scope('train'):
    #optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cost)
    #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.name_scope('accuracy'):
    predictions = tf.argmax(pred, 1)
    actuals = tf.argmax(y, 1)
    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)
    with tf.name_scope('correct_prediction'):
        correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        tp_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals), tf.equal(predictions, ones_like_predictions)), "float"))
        tn_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals),tf.equal(predictions, zeros_like_predictions)),"float"))
        fp_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals), tf.equal(predictions, ones_like_predictions)), "float"))
        fn_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals), tf.equal(predictions, zeros_like_predictions)), "float"))
        tpr = (tp_op)/((tp_op) +(fn_op))
        tnr= (tn_op)/((tn_op)+(fp_op))
tf.summary.scalar('accuracy',accuracy)


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3) 
init = tf.initialize_all_variables()
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
merged=tf.summary.merge_all()
sess.run(init)
saver=tf.train.Saver()
vars=tf.trainable_variables()
params_load=[v for v in vars if (v.name.startswith('final_all_1D_CNN'))]
tf.train.Saver(params_load).restore(sess,model_path)

print("NN-based classifier prediction begin")
mismatch_acc=0.0
mismatch_tp=0
mismatch_tn=0
mismatch_fp=0
mismatch_fn=0
mismatch_tpr=0.0
mismatch_tnr=0.0
mismatch_FPR=0.0
mismatch_FNR=0.0
precision=0.0
recall=0.0
F1_score=0.0
mismatch_auc=0.0
mismatch_total_batch=int(nmismatch/batch_size2)
j_m=[i_m for i_m in range(mismatch_total_batch)]
rr=random.random
random.shuffle(j_m,random=rr)
mismatch_pro=[]
mismatch_label=[]
for i_m in j_m:
    mismatch_xs=mismatch_x[i_m*batch_size2:(i_m+1)*batch_size2]
    mismatch_ys =mismatch_y[i_m*batch_size2:(i_m+1)*batch_size2]
    mismatch_xs = mismatch_xs.reshape((batch_size2,dim))
    mismatch_acc+=sess.run(accuracy, feed_dict={x:mismatch_xs, y:mismatch_ys,is_training2:0})
    mismatch_tp+=sess.run(tp_op, feed_dict={x:mismatch_xs, y:mismatch_ys,is_training2:0})
    mismatch_tn+=sess.run(tn_op, feed_dict={x:mismatch_xs, y:mismatch_ys,is_training2:0})
    mismatch_fp+=sess.run(fp_op, feed_dict={x:mismatch_xs, y:mismatch_ys,is_training2:0})
    mismatch_fn+=sess.run(fn_op, feed_dict={x:mismatch_xs, y:mismatch_ys,is_training2:0})
    mismatch_y_l=sess.run(y, feed_dict={y:mismatch_ys})
    mismatch_y_p=sess.run(pred, feed_dict={x:mismatch_xs, y:mismatch_ys,is_training2:0})
    mismatch_y_l=mismatch_y_l.transpose(1,0)
    mismatch_y_p=mismatch_y_p.transpose(1,0)
    mismatch_pro.append(mismatch_y_p[1])
    mismatch_label.append(mismatch_y_l[1])
    
mismatch_pro2=np.array(mismatch_pro)
mismatch_label2=np.array(mismatch_label)
mismatch_pro2=mismatch_pro2.reshape([1,-1])
mismatch_label2=mismatch_label2.reshape([1,-1])
mismatch_auc=roc_auc_score(mismatch_label2[0],mismatch_pro2[0])
mismatch_acc=mismatch_acc/mismatch_total_batch


mismatch_tpr=mismatch_tp/(mismatch_tp+mismatch_fn)
mismatch_tnr=mismatch_tn/(mismatch_tn+mismatch_fp)
mismatch_FPR=mismatch_fp/(mismatch_fp+mismatch_tn)
mismatch_FNR=mismatch_fn/(mismatch_tp+mismatch_fn)
precision=mismatch_tp/(mismatch_tp+mismatch_fp)
recall=mismatch_tp/(mismatch_tp+mismatch_fn)
F1_score=2*precision*recall/(precision+recall)

print (" Mismatch accuracy: %.4f TPR: %.4f TNR: %.4f" % ((mismatch_acc,mismatch_tpr,mismatch_tnr)))
print (" Mismatch FPR: %.4f FNR: %.4f AUC: %.4f" % ((mismatch_FPR,mismatch_FNR,mismatch_auc)))
print (" Mismatch precision: %.4f recall: %.4f F1_score: %.4f" % ((precision,recall,F1_score)))
print (" NN-based classifier Finished .")
