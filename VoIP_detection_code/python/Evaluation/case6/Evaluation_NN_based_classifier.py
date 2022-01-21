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
extracted_feature_path=('/VoIP_detection/VoIP_detection_code/python/classification/case6/predict')
model_save_path=('/VoIP_detection/VoIP_detection_code/python/classification/case6/DNN')


load_name3=extracted_feature_path+'/mismatch.mat'
load_name4=extracted_feature_path+'/mismatch2.mat'
model_path=model_save_path+'/save_model/0.0002/save1_met.ckpt'



load_data3=sio.loadmat(load_name3)
load_data4=sio.loadmat(load_name4)

mismatch_x=load_data3['test_x']
mismatch_y=load_data3['test_y']

mismatch_x2=load_data4['test_x2']
mismatch_y2=load_data4['test_y2']
nmismatch2=mismatch_x2.shape[0]

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


print("The number of mismatchtesting sample is %d:"%nmismatch)
print("The number of input unit is %d"%dim)
print("The number of output unit is %d"%n_class)

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


def get_result(n_sample,batch_size_in,test_x_in,test_y_in,dim_in):
     test_loss=0.0
     test_tp=0
     test_tn=0
     test_fp=0
     test_fn=0
     test_acc=0.0
     test_pro=[]
     test_label=[]
     total_batch=int(n_sample/batch_size_in)
     j_t=[i_t for i_t in range(total_batch)]
     rr=random.random
     random.shuffle(j_t,random=rr)
     for i_t in j_t:
         test_xs=test_x_in[i_t*batch_size_in:(i_t+1)*batch_size_in]
         test_ys =test_y_in[i_t*batch_size_in:(i_t+1)*batch_size_in]
         test_xs = test_xs.reshape((batch_size_in,dim_in))
         test_acc+=sess.run(accuracy, feed_dict={x:test_xs, y:test_ys,is_training2:0})
         test_loss +=sess.run(cost,feed_dict={x:test_xs,y:test_ys,is_training2:0})
         test_tp+=sess.run(tp_op, feed_dict={x:test_xs, y:test_ys,is_training2:0})
         test_tn+=sess.run(tn_op, feed_dict={x:test_xs, y:test_ys,is_training2:0})
         test_fp+=sess.run(fp_op, feed_dict={x:test_xs, y:test_ys,is_training2:0})
         test_fn+=sess.run(fn_op, feed_dict={x:test_xs, y:test_ys,is_training2:0})
         test_y_l=sess.run(y, feed_dict={y:test_ys})
         test_y_p=sess.run(pred, feed_dict={x:test_xs, y:test_ys,is_training2:0})
         test_y_l=test_y_l.transpose(1,0)
         test_y_p=test_y_p.transpose(1,0)
         test_pro.append(test_y_p[1])
         test_label.append(test_y_l[1])
     return test_tp,test_tn,test_fp,test_fn,test_pro,test_label,total_batch,test_acc

print("Case6-1 NN-based classifier prediction begin")

mismatch1_tp,mismatch1_tn,mismatch1_fp,mismatch1_fn,mismatch1_pro,mismatch1_label,mismatch1_batch,mismatch1_acc=get_result(nmismatch,batch_size1,mismatch_x,mismatch_y,dim)
mismatch1_pro2=np.array(mismatch1_pro)
mismatch1_label2=np.array(mismatch1_label)
mismatch1_pro2=mismatch1_pro2.reshape([1,-1])
mismatch1_label2=mismatch1_label2.reshape([1,-1])
mismatch1_auc=roc_auc_score(mismatch1_label2[0],mismatch1_pro2[0])
mismatch1_acc=mismatch1_acc/mismatch1_batch

mismatch1_tpr=mismatch1_tp/(mismatch1_tp+mismatch1_fn)
mismatch1_tnr=mismatch1_tn/(mismatch1_tn+mismatch1_fp)
mismatch1_FPR=mismatch1_fp/(mismatch1_fp+mismatch1_tn)
mismatch1_FNR=mismatch1_fn/(mismatch1_tp+mismatch1_fn)
mismatch1_precision=mismatch1_tp/(mismatch1_tp+mismatch1_fp)
mismatch1_recall=mismatch1_tp/(mismatch1_tp+mismatch1_fn)
mismatch1_F1_score=2*mismatch1_precision*mismatch1_recall/(mismatch1_precision+mismatch1_recall)
print (" Case6-1 NN-based classifier final predicting:")
print (" Mismatch accuracy: %.4f TPR: %.4f TNR: %.4f" % ((mismatch1_acc,mismatch1_tpr,mismatch1_tnr)))
print (" Mismatch FPR: %.4f FNR: %.4f AUC: %.4f" % ((mismatch1_FPR,mismatch1_FNR,mismatch1_auc)))
print (" Mismatch precision: %.4f recall: %.4f F1_score: %.4f" % ((mismatch1_precision,mismatch1_recall,mismatch1_F1_score)))
print (" Case6-1 NN-based classifier final finish!")


print("Case6-1 NN-based classifier prediction begin")
mismatch2_tp,mismatch2_tn,mismatch2_fp,mismatch2_fn,mismatch2_pro,mismatch2_label,mismatch2_batch,mismatch2_acc=get_result(nmismatch2,batch_size1,mismatch_x2,mismatch_y2,dim)
mismatch2_pro2=np.array(mismatch2_pro)
mismatch2_label2=np.array(mismatch2_label)
mismatch2_pro2=mismatch2_pro2.reshape([1,-1])
mismatch2_label2=mismatch2_label2.reshape([1,-1])
mismatch2_auc=roc_auc_score(mismatch2_label2[0],mismatch2_pro2[0])
mismatch2_acc=mismatch2_acc/mismatch2_batch

mismatch2_tpr=mismatch2_tp/(mismatch2_tp+mismatch2_fn)
mismatch2_tnr=mismatch2_tn/(mismatch2_tn+mismatch2_fp)
mismatch2_FPR=mismatch2_fp/(mismatch2_fp+mismatch2_tn)
mismatch2_FNR=mismatch2_fn/(mismatch2_tp+mismatch2_fn)
mismatch2_precision=mismatch2_tp/(mismatch2_tp+mismatch2_fp)
mismatch2_recall=mismatch2_tp/(mismatch2_tp+mismatch2_fn)
mismatch2_F1_score=2*mismatch2_precision*mismatch2_recall/(mismatch2_precision+mismatch2_recall)
print (" Case6-2 NN-based classifier final predicting:")
print (" Mismatch accuracy: %.4f TPR: %.4f TNR: %.4f" % ((mismatch2_acc,mismatch2_tpr,mismatch2_tnr)))
print (" Mismatch FPR: %.4f FNR: %.4f AUC: %.4f" % ((mismatch2_FPR,mismatch2_FNR,mismatch2_auc)))
print (" Mismatch precision: %.4f recall: %.4f F1_score: %.4f" % ((mismatch2_precision,mismatch2_recall,mismatch2_F1_score)))
print (" Case6-2 NN-based classifier final finish!")
print ("Finished .")
