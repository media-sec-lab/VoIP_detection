# -*- coding: utf-8 -*-
#!/usr/bin/env python2
"""
Created on Fri May 18 14:17:39 2018

@author: Administrator
inception
"""
import sys
sys.path.append('/home/huangyuankun/install_file')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"

import scipy.io as sio
import tensorflow as tf
import time
import random
import h5py
import numpy as np
tf.set_random_seed(3) 
#choose the mat
mat_load_path=('/VoIP_detection/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case0/mat')
model_save_path=('/VoIP_detection/VoIP_detection_code/python/Feature_extraction/swf_net')

load_name=mat_load_path+'/train_norm_per_sample_framewav.mat'
val_acc_save_name=model_save_path+'/save_acc/0.0002/val/'
match_acc_save_name=model_save_path+'/save_acc/0.0002/test/'
mismatch_acc_save_name=model_save_path+'/save_acc/0.0002/mismatch/'
log_dir=model_save_path+'/save_model/0.0002'
model_path=model_save_path+"/save_model/0.0002/save1_met.ckpt"


load_data=h5py.File(load_name)
mismatch_x=load_data['mismatch_x']
mismatch_y=load_data['mismatch_y']
train_x=load_data['train_x']
train_y=load_data['train_y']
val_x=load_data['val_x']
val_y=load_data['val_y']

train_x=train_x[0:,0:,0:]
train_y=train_y[0:,0:]
val_x=val_x[0:,0:,0:]
val_y=val_y[0:,0:]
mismatch_x=mismatch_x[0:,0:,0:]
mismatch_y=mismatch_y[0:,0:]

train_x=train_x.transpose((0,2,1))
train_y=train_y.transpose((1,0))
val_x=val_x.transpose((0,2,1))
val_y=val_y.transpose((1,0))
mismatch_x=mismatch_x.transpose((0,2,1))
mismatch_y=mismatch_y.transpose((1,0))


ntrain=train_x.shape[0]
fbin=train_x.shape[1] #input timesteps
#timesteps=1
nframes=train_x.shape[2]

n_class=train_y.shape[1]
n_val=val_x.shape[0]
nmismatch=mismatch_x.shape[0]


print "The number of training sample is %d:"%ntrain
print "The number of input unit is %d"%nframes
print "The number of output unit is %d"%n_class
print "The fbin is : %d"%fbin
#print "The number of testing sample is %d:"%ntest
print "The number of validation sample is %d:"%n_val
print "The number of mismatch sample is %d:"%nmismatch


learning_rate_v = [0.0002]
display_step = 1
display_ites=1
training_epochs = 150
n_hidden=300
lstm_layers = 3
#network parameters
n_input = nframes 
batch_size1=25


val_acc_max=0.0
val_acc_max_epoch=0
val_tpr_max=0.0
val_tnr_max=0.0


mismatch_acc_max=0.0
mismatch_tpr_max=0.0
mismatch_tnr_max=0.0
mismatch_acc_max_epoch=0

learning_rate_max=0
mismatch_learning_rate_max=0


max_val_mismatch=0.0
max_val_mismatch_tpr=0.0
max_val_mismatch_tnr=0.0



learning_rate=tf.placeholder("float32",name='learning_rate')
#tf Graph input



with tf.name_scope('input'):
    x = tf.placeholder("float32",[None,fbin,nframes],name='x_input')
    y = tf.placeholder("float32",[None,n_class],name='y_input')
#dropout=tf.placeholder(tf.float32)
batch_size=tf.placeholder(tf.int32)
keep_prob=tf.placeholder(tf.float32)
is_training2=tf.placeholder(tf.int32)

def xavier_init(fan_in, fan_out):
    low = -1 * np.sqrt(6.0 / (fan_in + fan_out))
    high = 1 * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
        
def weight_variable(shape,variable_name):
    initial=tf.Variable(tf.truncated_normal(shape,mean=0,stddev=0.01),name=variable_name,dtype=tf.float32)
    tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(0.0001)(initial))
    return initial

def bias_variable(shape,in_name):
    return tf.Variable(tf.zeros(shape)+0.1,name=in_name)


def batch_norm(inputs,is_training,variable_name,is_conv_out=True,decay=0.999):
    scale=tf.Variable(tf.ones([inputs.get_shape()[-1]]),name=variable_name+'_scale')
    beta=tf.Variable(tf.zeros(inputs.get_shape()[-1]),name=variable_name+'_beta')
    pop_mean=tf.Variable(tf.zeros([inputs.get_shape()[-1]]),name=variable_name+'_pop_mean',trainable=False)
    pop_var=tf.Variable(tf.ones([inputs.get_shape()[-1]]),name=variable_name+'_pop_var',trainable=False)
    if is_training==1:
        if is_conv_out:
            batch_mean,batch_var=tf.nn.moments(inputs,[0,1,2])
        else:
            batch_mean,batch_vat=tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, 0.001)        
    
def conv2d_with_batch_norm(x,filter_shape,stride_l,stride_w,in_name,act=None):
    filter_ =weight_variable(filter_shape,in_name)
    conv=tf.nn.conv2d(x,filter=filter_,strides=[1,stride_l,stride_w,1],padding='SAME')
    normed=batch_norm(conv,is_training2,in_name)  
    if act is None:
        return normed
    else:
        return act(normed)
    
def conv2d(x,filter_shape,stride_l,stride_w,in_name,padding_mode='SAME',layer_name="conv",act=None):
    with tf.name_scope(layer_name):
        out_channels=filter_shape[3]
        conv=tf.nn.conv2d(x,filter=weight_variable(filter_shape,variable_name=in_name),strides=[1,stride_l,stride_w,1],padding=padding_mode)
        #bias=tf.Variable(tf.zeros([out_channels]),name='bias')
        bias=bias_variable([out_channels])
        if act is None:
            return(tf.nn.bias_add(conv,bias))
        else:
            return act(tf.nn.bias_add(conv,bias))      

def max_pool(x,pleng1,pleng2,sleng1,sleng2,padding_mode='SAME'):
    return tf.nn.max_pool(x,ksize=[1,pleng1,pleng2,1],strides=[1,sleng1,sleng2,1],padding=padding_mode)
    


    
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

    
x_input=tf.reshape(x,[-1,fbin,nframes,1])
print(x_input.shape)
with tf.name_scope('New_Inception_fully_conv_for_frame_wav1_with_two_pred'):
        split_conv_x12=conv2d_with_batch_norm(x_input,[1,5,1,48],1,1,'frame_wav_conv1_2',act=tf.nn.relu)
        x2=split_conv_x12
        x2_pool=max_pool(x2,1,5,1,5)
        x2_normed=batch_norm(x2_pool,is_training2,'bn2')
        split_conv_x22=conv2d_with_batch_norm(x2_normed,[1,5,48,96],1,1,'frame_wav_conv2_2',act=tf.nn.relu)
        x3=split_conv_x22
        x3_pool=max_pool(x3,1,5,1,5)
        x3_normed=batch_norm(x3_pool,is_training2,'bn3')
        split_conv_x32=conv2d_with_batch_norm(x3_normed,[1,5,96,192],1,1,'frame_wav_conv3_2',act=tf.nn.relu)
        x4=split_conv_x32
        x4_pool=max_pool(x4,1,4,1,4)
        x4_normed=batch_norm(x4_pool,is_training2,'bn4')
        split_conv_x42=conv2d_with_batch_norm(x4_normed,[5,2,192,384],1,1,'frame_wav_conv4_2',act=tf.nn.relu)
        x5=split_conv_x42
        x5_1=x5
        x5_pool=max_pool(x5,4,2,4,2)
        x5_normed=batch_norm(x5_pool,is_training2,'bn5')
        split_conv_x52=conv2d_with_batch_norm(x5_normed,[5,1,384,768],1,1,'frame_wav_conv5_2',act=tf.nn.relu)
        x6=split_conv_x52
        x6_pool=max_pool(x6,2,1,2,1)
        x6_normed=batch_norm(x6_pool,is_training2,'bn6')
        split_conv_x62=conv2d_with_batch_norm(x6_normed,[5,1,768,768],1,1,'frame_wav_conv6_2',act=tf.nn.relu)
        x7=split_conv_x62
        h_pool6=tf.reduce_mean(x7,[1,2],name='avgpool',keep_dims=True)
        print(h_pool6.shape)
        
        shape=h_pool6.get_shape().as_list()
        i_shape=shape[1]*shape[2]*shape[3]
        h_pool6_flat=tf.reshape(h_pool6,[-1,i_shape])
       
        h_pool6_flat_out=nn_layer3(h_pool6_flat,768,n_class,'frame_wav_fully_connet1')

        pred=tf.nn.softmax(h_pool6_flat_out)
        
        h_pool6_2=tf.reduce_mean(x5_1,[1,2],name='avgpool',keep_dims=True)
        h_pool6_flat_2=tf.reshape(h_pool6_2,[-1,384])
        print(h_pool6_flat_2.shape)
        fc_6=nn_layer3(h_pool6_flat_2,384,192,'frame_wav_fully_connect2')
        
        fc_6_normed=tf.nn.relu(batch_norm(fc_6,is_training2,'bn9'))
        fc_6_nnout=nn_layer3(fc_6_normed,192,n_class,'frame_wav_fully_connet3')
        pred2=tf.nn.softmax(fc_6_nnout)

print(pred.shape)




print('pred')
with tf.name_scope('cross_entropy'):
    cost1 = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred+1e-10),reduction_indices=[1]))
    cost2 = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred2+1e-10),reduction_indices=[1]))
    cost=0.2*cost1+0.8*cost2

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

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.42) 
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
 epoch_mismatch_acc_save_name=mismatch_acc_save_name+str(learning_rate_v[learning_rate_i])+'.mat'
 epoch_mismatch_tpr_save_name=mismatch_acc_save_name+str(learning_rate_v[learning_rate_i])+'_tpr.mat'
 epoch_mismatch_tnr_save_name=mismatch_acc_save_name+str(learning_rate_v[learning_rate_i])+'_tnr.mat'
 random.seed(2)
 count=1
 for epoch in range(training_epochs):
     loss=0.0
     total_batch=int(ntrain/batch_size1)
     j=[i for i in range(total_batch)]
     rr=random.random
     
     random.shuffle(j,random=rr)
     #shuffle(j)
     total_time = 0.0
     start_time = time.time()
     train_acc=0.0
     for i in j:
         batch_xs=train_x[i*batch_size1:(i+1)*batch_size1]
         #batch_xs=max_min_normalization(batch_xs)
         batch_ys =train_y[i*batch_size1:(i+1)*batch_size1]
         batch_xs = batch_xs.reshape((batch_size1,fbin,nframes))
         #is_training=True
         sess.run(optimizer,feed_dict={x:batch_xs,y:batch_ys,learning_rate:learning_rate_v[learning_rate_i],keep_prob:0.5,is_training2:1,batch_size:batch_size1})
         loss +=sess.run(cost,feed_dict={x:batch_xs,y:batch_ys,learning_rate:learning_rate_v[learning_rate_i],keep_prob:1.0,is_training2:0,batch_size:batch_size1})
         train_acc+=sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys,learning_rate:learning_rate_v[learning_rate_i],keep_prob:1.0,is_training2:0,batch_size:batch_size1})
         if i!=j[-1]:
             batch_xs = batch_xs.reshape((batch_size1,fbin*nframes))
         count=count+1
     if epoch % display_ites ==0:  
             mismatch_acc=0.0
             mismatch_tp=0
             mismatch_tn=0
             mismatch_fp=0
             mismatch_fn=0
             mismatch_tpr=0.0
             mismatch_tnr=0.0
             mismatch_total_batch=int(nmismatch/batch_size1)
             j_m=[i_m for i_m in range(mismatch_total_batch)]
             rr=random.random
             random.shuffle(j_m,random=rr)
             #shuffle(j)
             for i_m in j_m:
                 mismatch_xs=mismatch_x[i_m*batch_size1:(i_m+1)*batch_size1]
                 #mismatch_xs=max_min_normalization(mismatch_xs)
                 mismatch_ys =mismatch_y[i_m*batch_size1:(i_m+1)*batch_size1]
                 mismatch_xs = mismatch_xs.reshape((batch_size1,fbin,nframes)) 
                 mismatch_acc+=sess.run(accuracy, feed_dict={x:mismatch_xs, y:mismatch_ys,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
                 mismatch_tp+=sess.run(tp_op, feed_dict={x:mismatch_xs, y:mismatch_ys,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
                 mismatch_tn+=sess.run(tn_op, feed_dict={x:mismatch_xs, y:mismatch_ys,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
                 mismatch_fp+=sess.run(fp_op, feed_dict={x:mismatch_xs, y:mismatch_ys,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
                 mismatch_fn+=sess.run(fn_op, feed_dict={x:mismatch_xs, y:mismatch_ys,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
                 if i_m!=j_m[-1]:
                     mismatch_xs = mismatch_xs.reshape((batch_size1,fbin*nframes)) 
             mismatch_acc=mismatch_acc/mismatch_total_batch
             mismatch_tpr=mismatch_tp/(mismatch_tp+mismatch_fn)
             mismatch_tnr=mismatch_tn/(mismatch_tn+mismatch_fp)
             summary=sess.run(merged, feed_dict={x:mismatch_xs, y:mismatch_ys,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
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
             val_total_batch=int(n_val/batch_size1)
             j_v=[i_v for i_v in range(val_total_batch)]
             rr=random.random
             random.shuffle(j_v,random=rr)
             #shuffle(j)
             for i_v in j_v:
                 val_xs=val_x[i_v*batch_size1:(i_v+1)*batch_size1]
                 #val_xs=max_min_normalization(val_xs)
                 val_ys =val_y[i_v*batch_size1:(i_v+1)*batch_size1]
                 val_xs = val_xs.reshape((batch_size1, fbin,nframes))
                 val_acc+=sess.run(accuracy, feed_dict={x:val_xs, y:val_ys,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
                 val_tp+=sess.run(tp_op, feed_dict={x:val_xs, y:val_ys,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
                 val_tn+=sess.run(tn_op, feed_dict={x:val_xs, y:val_ys,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
                 val_fp+=sess.run(fp_op, feed_dict={x:val_xs, y:val_ys,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
                 val_fn+=sess.run(fn_op, feed_dict={x:val_xs, y:val_ys,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
                 val_loss +=sess.run(cost,feed_dict={x:val_xs,y:val_ys,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
                 if i_v!=j_v[-1]:
                     val_xs = val_xs.reshape((batch_size1, fbin*nframes))
      
             val_acc=val_acc/val_total_batch
             val_tpr=val_tp/(val_tp+val_fn)
             val_tnr=val_tn/(val_tn+val_fp)
             summary=sess.run(merged, feed_dict={x:val_xs, y:val_ys,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
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
     summary,pre_y=sess.run([merged,pred], feed_dict={x:batch_xs, y:batch_ys,learning_rate:learning_rate_v[learning_rate_i],keep_prob:1.0,is_training2:0,batch_size:batch_size1})
     train_writer.add_summary(summary,epoch)
     print (" Training accuracy: %.3f" % (train_acc))
     print ("high_pass frame_wav different software alicall Inception test Optimization Adam 0.0002 .")
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
