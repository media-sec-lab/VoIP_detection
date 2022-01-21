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
#import h5py
#import math
import random
extracted_feature_path=('/VoIP_detection/VoIP_detection_code/python/classification/case6/predict')
model_save_path=('/VoIP_detection/VoIP_detection_code/python/classification/case6/DNN')

load_name=extracted_feature_path+'/train.mat'
load_name2=extracted_feature_path+'/val.mat'
load_name3=extracted_feature_path+'/mismatch.mat'

val_acc_save_name=model_save_path+'/save_acc/0.0002/val/'
match_acc_save_name=model_save_path+'/save_acc/0.0002/test/'
mismatch_acc_save_name=model_save_path+'/save_acc/0.0002/mismatch/'
log_dir=model_save_path+'/save_model/0.0002'
model_path=model_save_path+'/save_model/0.0002/save1_met.ckpt'

load_data=sio.loadmat(load_name)
load_data2=sio.loadmat(load_name2)
load_data3=sio.loadmat(load_name3)

train_x=load_data['train_x']
train_y=load_data['train_y']
val_x=load_data2['val_x']
val_y=load_data2['val_y']
mismatch_x=load_data3['test_x']
mismatch_y=load_data3['test_y']
ntrain=train_x.shape[0]
nval=val_x.shape[0]
nmismatch=mismatch_x.shape[0]
dim=train_x.shape[1]
n_class=train_y.shape[1] #output unit

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
tf.summary.scalar('cross_entropy',cost)

with tf.name_scope('train'):
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
train_writer=tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')
sess.run(init)
saver=tf.train.Saver()

print("Start optimization")
for learning_rate_i in range(len(learning_rate_v)):
 epoch_match_acc=[]
 epoch_match_tpr=[]
 epoch_match_tnr=[]
 epoch_match_acc_save_name=match_acc_save_name+str(learning_rate_v[learning_rate_i])+'.mat'
 epoch_match_tpr_save_name=match_acc_save_name+str(learning_rate_v[learning_rate_i])+'_tpr.mat'
 epoch_match_tnr_save_name=match_acc_save_name+str(learning_rate_v[learning_rate_i])+'_tnr.mat'
 
                                                  
 epoch_val_acc=[]
 epoch_val_tpr=[]
 epoch_val_tnr=[]
 epoch_val_acc_save_name=val_acc_save_name+str(learning_rate_v[learning_rate_i])+'.mat'
 epoch_val_tpr_save_name=val_acc_save_name+str(learning_rate_v[learning_rate_i])+'_tpr.mat'
 epoch_val_tnr_save_name=val_acc_save_name+str(learning_rate_v[learning_rate_i])+'_tnr.mat'
 
 epoch_mismatch_acc=[]
 epoch_mismatch_tpr=[]
 epoch_mismatch_tnr=[]
 epoch_mismatch_acc_save_name=mismatch_acc_save_name+str(1)+'.mat'
 epoch_mismatch_tpr_save_name=mismatch_acc_save_name+str(1)+'_tpr.mat'
 epoch_mismatch_tnr_save_name=mismatch_acc_save_name+str(1)+'_tnr.mat'
 

 random.seed(2)
 count=0
 for epoch in range(training_epochs):
     
     loss=0.0
     total_batch=int(ntrain/batch_size2)
     j=[i for i in range(total_batch)]
     rr=random.random
     
     random.shuffle(j,random=rr)
     #shuffle(j)
     total_time = 0.0
     train_acc=0.0
     for i in j:
         batch_xs=train_x[i*batch_size2:(i+1)*batch_size2]
         batch_ys =train_y[i*batch_size2:(i+1)*batch_size2]
         batch_xs = batch_xs.reshape((batch_size2,dim))
         #is_training=True
         sess.run(optimizer,feed_dict={x:batch_xs,y:batch_ys,learning_rate:learning_rate_v[learning_rate_i],is_training2:1})
         loss +=sess.run(cost,feed_dict={x:batch_xs,y:batch_ys,learning_rate:learning_rate_v[learning_rate_i],is_training2:0,})
         train_acc+=sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys,learning_rate:learning_rate_v[learning_rate_i],is_training2:0})
         count=count+1 
     if epoch % display_ites ==0:           
             mismatch_acc=0.0
             mismatch_tp=0
             mismatch_tn=0
             mismatch_fp=0
             mismatch_fn=0
             mismatch_tpr=0.0
             mismatch_tnr=0.0
             mismatch_total_batch=int(nmismatch/batch_size2)
             j_m=[i_m for i_m in range(mismatch_total_batch)]
             rr=random.random
             random.shuffle(j_m,random=rr)
             for i_m in j_m:
                 mismatch_xs=mismatch_x[i_m*batch_size2:(i_m+1)*batch_size2]
                 mismatch_ys =mismatch_y[i_m*batch_size2:(i_m+1)*batch_size2]
                 mismatch_xs = mismatch_xs.reshape((batch_size2,dim))
                 mismatch_acc+=sess.run(accuracy, feed_dict={x:mismatch_xs, y:mismatch_ys,is_training2:0})
                 mismatch_tp+=sess.run(tp_op, feed_dict={x:mismatch_xs, y:mismatch_ys,is_training2:0})
                 mismatch_tn+=sess.run(tn_op, feed_dict={x:mismatch_xs, y:mismatch_ys,is_training2:0})
                 mismatch_fp+=sess.run(fp_op, feed_dict={x:mismatch_xs, y:mismatch_ys,is_training2:0})
                 mismatch_fn+=sess.run(fn_op, feed_dict={x:mismatch_xs, y:mismatch_ys,is_training2:0})
             mismatch_acc=mismatch_acc/mismatch_total_batch
             mismatch_tpr=mismatch_tp/(mismatch_tp+mismatch_fn)
             mismatch_tnr=mismatch_tn/(mismatch_tn+mismatch_fp)
             summary=sess.run(merged, feed_dict={x:mismatch_xs, y:mismatch_ys,is_training2:0})
             test_writer.add_summary(summary,epoch)
             epoch_mismatch_acc.append(mismatch_acc)
             epoch_mismatch_tpr.append(mismatch_tpr)
             epoch_mismatch_tnr.append(mismatch_tnr)
             print (" Mismatch accuracy: %.3f TPR: %.3f TNR: %.3f" % ((mismatch_acc,mismatch_tpr,mismatch_tnr)))
             if mismatch_acc > mismatch_acc_max:
                 mismatch_acc_max=mismatch_acc
                 mismatch_tpr_max=mismatch_tpr
                 mismatch_tnr_max=mismatch_tnr
                 mismatch_acc_max_epoch=epoch
                 mismatch_learning_rate_max=learning_rate_v[learning_rate_i]
             print("max_mismatch_accuracy: %.5f max_mismatch_tpr: %.5f max_mismatch_tnr: %.5f iterations: %03d learning_rate_max: %.5f" % (mismatch_acc_max,mismatch_tpr_max,mismatch_tnr_max, mismatch_acc_max_epoch, mismatch_learning_rate_max))
             
             
             val_loss=0.0
             val_acc=0.0
             val_tp=0
             val_tn=0
             val_fp=0
             val_fn=0
             val_tpr=0.0
             val_tnr=0.0
             val_total_batch=int(nval/batch_size2)
             j_v=[i_v for i_v in range(val_total_batch)]
             rr=random.random
             random.shuffle(j_v,random=rr)
             #shuffle(j)
             for i_v in j_v:
                 val_xs=val_x[i_v*batch_size2:(i_v+1)*batch_size2]
                 val_ys =val_y[i_v*batch_size2:(i_v+1)*batch_size2]
                 
                 val_xs = val_xs.reshape((batch_size2,dim))
                 val_acc+=sess.run(accuracy, feed_dict={x:val_xs, y:val_ys,is_training2:0})
                 val_tp+=sess.run(tp_op, feed_dict={x:val_xs, y:val_ys,is_training2:0})
                 val_tn+=sess.run(tn_op, feed_dict={x:val_xs, y:val_ys,is_training2:0})
                 val_fp+=sess.run(fp_op, feed_dict={x:val_xs, y:val_ys,is_training2:0})
                 val_fn+=sess.run(fn_op, feed_dict={x:val_xs, y:val_ys,is_training2:0})
                 val_loss +=sess.run(cost,feed_dict={x:val_xs,y:val_ys,is_training2:0})
             
             val_acc=val_acc/val_total_batch
             val_tpr=val_tp/(val_tp+val_fn)
             val_tnr=val_tn/(val_tn+val_fp)
             summary=sess.run(merged, feed_dict={x:val_xs, y:val_ys,is_training2:0})
             epoch_val_acc.append(val_acc)
             epoch_val_tpr.append(val_tpr)
             epoch_val_tnr.append(val_tnr)
             test_writer.add_summary(summary,epoch)
             print (" Val accuracy: %.3f TPR: %.3f TNR: %.3f cost: %.9f" % (val_acc,val_tpr,val_tnr,val_loss))
             if val_acc > val_acc_max:
                 val_acc_max=val_acc
                 val_tpr_max=val_tpr
                 val_tnr_max=val_tnr
                 val_acc_max_epoch=epoch
                 learning_rate_max=learning_rate_v[learning_rate_i]
             #    max_val_test=test_acc
             #    max_val_test_tpr=test_tpr
             #    max_val_test_tnr=test_tnr
                
                 max_val_mismatch=mismatch_acc
                 max_val_mismatch_tpr=mismatch_tpr
                 max_val_mismatch_tnr=mismatch_tnr
                 
                 save_path=saver.save(sess,model_path)
                 print("Save to path",save_path)
             print("max_val_accuracy: %.5f TPR: %.5f TNR: %.5f Iterations: %03d learning_rate_max: %.5f" % (val_acc_max,val_tpr_max,val_tnr_max, val_acc_max_epoch, learning_rate_max))
             #print("max_val_test_acc: %.5f  TPR: %.5f TNR: %.5f" % (max_val_test,max_val_test_tpr,max_val_test_tnr))
             print("max_mismatch_acc: %.5f  TPR: %.5f TNR: %.5f" % (max_val_mismatch,max_val_mismatch_tpr,max_val_mismatch_tnr))
             
           
     train_acc=train_acc/total_batch
     print ("Epoch: %03d/%03d cost: %.9f learning_rate: %.9f" % (epoch, training_epochs, loss,learning_rate_v[learning_rate_i]))
     summary,pre_y=sess.run([merged,pred], feed_dict={x:batch_xs, y:batch_ys,learning_rate:learning_rate_v[learning_rate_i],is_training2:0})
     train_writer.add_summary(summary,epoch)
     print (" Training accuracy: %.3f" % (train_acc))
     print ("high_pass f log_mel different software alicall Optimization Adam 0.0002  .")
     print("This epoch is done")
     print("This epoch is done")
         
 sio.savemat(epoch_mismatch_acc_save_name,{'mismatch_acc':epoch_mismatch_acc})
 sio.savemat(epoch_mismatch_tpr_save_name,{'mismatch_tpr':epoch_mismatch_tpr})
 sio.savemat(epoch_mismatch_tnr_save_name,{'mismatch_tnr':epoch_mismatch_tnr})
 sio.savemat(epoch_val_acc_save_name,{'val_acc':epoch_val_acc})
 sio.savemat(epoch_val_tpr_save_name,{'val_tpr':epoch_val_tpr})
 sio.savemat(epoch_val_tnr_save_name,{'val_tnr':epoch_val_tnr})
 train_writer.close()
 test_writer.close()

print ("Finished .")