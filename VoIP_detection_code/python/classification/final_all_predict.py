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
import h5py
import numpy as np
tf.set_random_seed(3)

 
mat_load_path=('/VoIP_detection/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case0/mat')
#the address to load trained swf-net model
swf_net_model=('/VoIP_detection/VoIP_detection_code/python/Feature_extraction/swf_net')
#the address to load trained ms-net model
ms_net_model=('/VoIP_detection/VoIP_detection_code/python/Feature_extraction/ms_net')
#the address to load trained ims-net model
ims_net_model=('/VoIP_detection/VoIP_detection_code/python/Feature_extraction/ims_net')
#the address to save the extracted feature
predict_path=('/VoIP_detection/VoIP_detection_code/python/classification/predict')

load_name=mat_load_path+'/train_norm_per_sample_logmel.mat'
load_name2=mat_load_path+'/train_norm_per_sample_ilogmel.mat'
load_name3=mat_load_path+'/train_norm_per_sample_framewav.mat'


load_data=h5py.File(load_name)
load_data2=h5py.File(load_name2)

test_x=load_data['mismatch_x']
test_y=load_data['mismatch_y']
train_x=load_data['train_x']
train_y=load_data['train_y']
val_x=load_data['val_x']
val_y=load_data['val_y']

train_x=train_x[0:,0:,0:]
train_y=train_y[0:,0:]
val_x=val_x[0:,0:,0:]
val_y=val_y[0:,0:]
test_x=test_x[0:,0:,0:]
test_y=test_y[0:,0:]

train_x=train_x.transpose((0,2,1))
train_y=train_y.transpose((1,0))
val_x=val_x.transpose((0,2,1))
val_y=val_y.transpose((1,0))
test_x=test_x.transpose((0,2,1))
test_y=test_y.transpose((1,0))


test_x_i=load_data2['mismatch_x']
train_x_i=load_data2['train_x']
val_x_i=load_data2['val_x']

train_x_i=train_x_i[0:,0:,0:]
val_x_i=val_x_i[0:,0:,0:]
test_x_i=test_x_i[0:,0:,0:]

train_x_i=train_x_i.transpose((0,2,1))
val_x_i=val_x_i.transpose((0,2,1))
test_x_i=test_x_i.transpose((0,2,1))

ntrain=train_x.shape[0]
fbin=train_x.shape[1] #input timesteps
#timesteps=1
nframes=train_x.shape[2]
#dim=train_x_i.shape[2] #input unit
n_class=train_y.shape[1]
n_val=val_x.shape[0]
n_test=test_x.shape[0]



load_data3=h5py.File(load_name3)
test_x_wav=load_data3['mismatch_x']
train_x_wav=load_data3['train_x']
val_x_wav=load_data3['val_x']

train_x_wav=train_x_wav[0:,0:,0:]
val_x_wav=val_x_wav[0:,0:,0:]
test_x_wav=test_x_wav[0:,0:,0:]
train_x_wav=train_x_wav.transpose((0,2,1))
val_x_wav=val_x_wav.transpose((0,2,1))
test_x_wav=test_x_wav.transpose((0,2,1))
dim=train_x_wav.shape[2] #input unt

print "The number of training sample is %d:"%ntrain
print "The number of input unit is %d"%nframes
print "The number of output unit is %d"%n_class
print "The fbin is : %d"%fbin
#print "The number of testing sample is %d:"%ntest
print "The number of validation sample is %d:"%n_val
print "The number of mismatch sample is %d:"%n_test

learning_rate_v = [0.0002]

display_step = 1
display_ites=1
training_epochs=50
n_hidden=300
lstm_layers = 3
#network parameters
n_input = nframes 
batch_size1=1


val_acc_max=0.0
val_acc_max_epoch=0
val_tpr_max=0.0
val_tnr_max=0.0

test_acc_max=0.0
test_tpr_max=0.0
test_tnr_max=0.0
test_acc_max_epoch=0

mismatch_acc_max=0.0
mismatch_tpr_max=0.0
mismatch_tnr_max=0.0
mismatch_acc_max_epoch=0

learning_rate_max=0
mismatch_learning_rate_max=0
max_val_test=0.0
max_val_test_tpr=0.0
max_val_test_tnr=0.0

max_val_mismatch=0.0
max_val_mismatch_tpr=0.0
max_val_mismatch_tnr=0.0


learning_rate=tf.placeholder("float32",name='learning_rate')
#tf Graph input
with tf.name_scope('input'):
    x = tf.placeholder("float32",[None,fbin,nframes],name='x_input')
    y = tf.placeholder("float32",[None,n_class],name='y_input')
    x_i = tf.placeholder("float32",[None,fbin,nframes],name='x_input_i')
    x_wav = tf.placeholder("float32",[None,nframes,dim],name='x_wav_input')

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
#[pred,pred2,pred3]=Inception_myself_fully_conv2(x_input,scope='New_3_3_with_bn_fully_conv2_with_three_pred_for_log_mel')
with tf.name_scope('New_3_3_with_bn_fully_conv2_with_three_pred_for_log_mel'):
        split_conv_x12=conv2d_with_batch_norm(x_input,[3,3,1,48],1,1,'log_mel_conv1_2',act=tf.nn.relu)
        x2=split_conv_x12
        x2_pool=max_pool(x2,2,2,2,2)
        print(x2_pool.shape)
        x2_normed=batch_norm(x2_pool,is_training2,'bn2') 
        split_conv_x22=conv2d_with_batch_norm(x2_normed,[3,3,48,96],1,1,'log_mel_conv2_2',act=tf.nn.relu)    
        x3=split_conv_x22
        x3_pool=max_pool(x3,2,2,2,2)
        print(x3_pool.shape)
        x3_normed=batch_norm(x3_pool,is_training2,'bn3')
        split_conv_x32=conv2d_with_batch_norm(x3_normed,[3,3,96,192],1,1,'log_mel_conv3_2',act=tf.nn.relu)
        x4=split_conv_x32
        x4_pool=max_pool(x4,2,2,2,2)
        print(x4_pool.shape)
        x4_normed=batch_norm(x4_pool,is_training2,'bn4')
        split_conv_x42=conv2d_with_batch_norm(x4_normed,[3,3,192,384],1,1,'log_mel_conv4_2',act=tf.nn.relu)
        x5=split_conv_x42
        x5_pool=max_pool(x5,2,1,2,1)
        print(x5_pool.shape)
        x5_normed=batch_norm(x5_pool,is_training2,'bn5')
        split_conv_x52=conv2d_with_batch_norm(x5_normed,[3,3,384,768],1,1,'log_mel_conv5_2',act=tf.nn.relu)
        x6=split_conv_x52
        x6_pool=max_pool(x6,3,1,3,1)
        print(x6_pool.shape)
        x6_normed=batch_norm(x6_pool,is_training2,'bn6')
        split_conv_x62=conv2d_with_batch_norm(x6_normed,[1,3,768,768],1,1,'log_mel_conv6_2',act=tf.nn.relu)
        x7=split_conv_x62
        h_pool6=tf.reduce_mean(x7,[1,2],name='avgpool',keep_dims=True)
        shape=h_pool6.get_shape().as_list()
        shape1=shape[1]*shape[2]*shape[3]
        h_pool6_flat=tf.reshape(h_pool6,[-1,shape1])
        h_pool6_nnout=nn_layer3(h_pool6_flat,768,n_class,'log_mel_fully_connect1')
        out=tf.nn.softmax(h_pool6_nnout)
        x4_avg_pool=tf.reduce_mean(x4,[1,2],name='avgpool_x4',keep_dims=True)
        h_pool4_flat=tf.reshape(x4_avg_pool,[-1,192])
        h_pool5_flat=nn_layer3(h_pool4_flat,192,96,'log_mel_fully_connect2')
        h_pool5_flat_normed=tf.nn.relu(batch_norm(h_pool5_flat,is_training2,'bn8'))
        h_pool5_nnout=nn_layer3(h_pool5_flat_normed,96,n_class,'log_mel_fully_connect3')

        out2=tf.nn.softmax(h_pool5_nnout)
         


x_input_i=tf.reshape(x_i,[-1,fbin,nframes,1])
print(x_input_i.shape)
with tf.name_scope('New_3_3_with_bn_fully_conv2_with_three_pred_for_ilog_mel'):
        i_split_conv_x12=conv2d_with_batch_norm(x_input_i,[3,3,1,48],1,1,'log_mel_conv1_2',act=tf.nn.relu)
        i_x2=i_split_conv_x12
        i_x2_pool=max_pool(i_x2,2,2,2,2)
        print(i_x2_pool.shape)
        i_x2_normed=batch_norm(i_x2_pool,is_training2,'bn2') 
        
        i_split_conv_x22=conv2d_with_batch_norm(i_x2_normed,[3,3,48,96],1,1,'log_mel_conv2_2',act=tf.nn.relu)
        i_x3=i_split_conv_x22
        i_x3_pool=max_pool(i_x3,2,2,2,2)
        print(i_x3_pool.shape)
        i_x3_normed=batch_norm(i_x3_pool,is_training2,'bn3')
        
        i_split_conv_x32=conv2d_with_batch_norm(i_x3_normed,[3,3,96,192],1,1,'log_mel_conv3_2',act=tf.nn.relu)
        i_x4=i_split_conv_x32
        i_x4_pool=max_pool(i_x4,2,2,2,2)
        print(i_x4_pool.shape)
        i_x4_normed=batch_norm(i_x4_pool,is_training2,'bn4')

        i_split_conv_x42=conv2d_with_batch_norm(i_x4_normed,[3,3,192,384],1,1,'log_mel_conv4_2',act=tf.nn.relu)
        i_x5=i_split_conv_x42
        i_x5_pool=max_pool(i_x5,2,1,2,1)
        print(i_x5_pool.shape)
        i_x5_normed=batch_norm(i_x5_pool,is_training2,'bn5')
        
        i_split_conv_x52=conv2d_with_batch_norm(i_x5_normed,[3,3,384,768],1,1,'log_mel_conv5_2',act=tf.nn.relu)
        i_x6=i_split_conv_x52
        i_x6_pool=max_pool(i_x6,3,1,3,1)
        print(i_x6_pool.shape)
        i_x6_normed=batch_norm(i_x6_pool,is_training2,'bn6')
        
        i_split_conv_x62=conv2d_with_batch_norm(i_x6_normed,[1,3,768,768],1,1,'log_mel_conv6_2',act=tf.nn.relu)
        i_x7=i_split_conv_x62 
        i_h_pool6=tf.reduce_mean(i_x7,[1,2],name='avgpool',keep_dims=True)

        i_shape=i_h_pool6.get_shape().as_list()
        i_i_shape=i_shape[1]*i_shape[2]*i_shape[3]
        i_h_pool6_flat=tf.reshape(i_h_pool6,[-1,i_i_shape])
        i_h_pool6_nnout=nn_layer3(i_h_pool6_flat,768,n_class,'log_mel_fully_connect1')
        #i_h_pool6_flat_normed=tf.nn.relu(batch_norm(i_h_pool6_nnout,is_training2,'bn7'))
        i_out=tf.nn.softmax(i_h_pool6_nnout)
        
        
        i_x4_avg_pool=tf.reduce_mean(i_x4,[1,2],name='avgpool_x4',keep_dims=True)
        i_h_pool4_flat=tf.reshape(i_x4_avg_pool,[-1,192])
        i_h_pool5_flat=nn_layer3(i_h_pool4_flat,192,96,'log_mel_fully_connect2')
        i_h_pool5_flat_normed=tf.nn.relu(batch_norm(i_h_pool5_flat,is_training2,'bn8'))
        i_h_pool5_nnout=nn_layer3(i_h_pool5_flat_normed,96,n_class,'log_mel_fully_connect3')
        #i_h_pool5_softmax_in=tf.nn.relu(batch_norm(i_h_pool5_nnout,is_training2,'bn9'))
        i_out2=tf.nn.softmax(i_h_pool5_nnout)
           


x_input_wav=tf.reshape(x_wav,[-1,nframes,dim,1])
print(x_input_wav.shape)
with tf.name_scope('New_Inception_fully_conv_for_frame_wav1_with_two_pred'):
        split_conv_x12_wav=conv2d_with_batch_norm(x_input_wav,[1,5,1,48],1,1,'frame_wav_conv1_2',act=tf.nn.relu)
        x2_wav=split_conv_x12_wav
        x2_pool_wav=max_pool(x2_wav,1,5,1,5)
        x2_normed_wav=batch_norm(x2_pool_wav,is_training2,'bn2')
        
        split_conv_x22_wav=conv2d_with_batch_norm(x2_normed_wav,[1,5,48,96],1,1,'frame_wav_conv2_2',act=tf.nn.relu)
        x3_wav=split_conv_x22_wav
        x3_pool_wav=max_pool(x3_wav,1,5,1,5)
        x3_normed_wav=batch_norm(x3_pool_wav,is_training2,'bn3')
        split_conv_x32_wav=conv2d_with_batch_norm(x3_normed_wav,[1,5,96,192],1,1,'frame_wav_conv3_2',act=tf.nn.relu)
        x4_wav=split_conv_x32_wav

        x4_pool_wav=max_pool(x4_wav,1,4,1,4)
        x4_normed_wav=batch_norm(x4_pool_wav,is_training2,'bn4')
        split_conv_x42_wav=conv2d_with_batch_norm(x4_normed_wav,[5,2,192,384],1,1,'frame_wav_conv4_2',act=tf.nn.relu)
        x5_wav=split_conv_x42_wav
        x5_1_wav=x5_wav
        x5_pool_wav=max_pool(x5_wav,4,2,4,2)
        x5_normed_wav=batch_norm(x5_pool_wav,is_training2,'bn5')
        
        split_conv_x52_wav=conv2d_with_batch_norm(x5_normed_wav,[5,1,384,768],1,1,'frame_wav_conv5_2',act=tf.nn.relu)
        x6_wav=split_conv_x52_wav
        x6_pool_wav=max_pool(x6_wav,2,1,2,1)
        x6_normed_wav=batch_norm(x6_pool_wav,is_training2,'bn6')
        
        split_conv_x62_wav=conv2d_with_batch_norm(x6_normed_wav,[5,1,768,768],1,1,'frame_wav_conv6_2',act=tf.nn.relu)
        x7_wav=split_conv_x62_wav

        h_pool6_wav=tf.reduce_mean(x7_wav,[1,2],name='avgpool',keep_dims=True)
        print(h_pool6_wav.shape)

        shape_wav=h_pool6_wav.get_shape().as_list()
        i_shape_wav=shape_wav[1]*shape_wav[2]*shape_wav[3]
        h_pool6_flat_wav=tf.reshape(h_pool6_wav,[-1,i_shape_wav])
        h_pool6_flat_out_wav=nn_layer3(h_pool6_flat_wav,768,n_class,'frame_wav_fully_connet1')
        pred_wav=tf.nn.softmax(h_pool6_flat_out_wav)
        h_pool6_2_wav=tf.reduce_mean(x5_1_wav,[1,2],name='avgpool',keep_dims=True)
        h_pool6_flat_2_wav=tf.reshape(h_pool6_2_wav,[-1,384])
        print(h_pool6_flat_2_wav.shape)
        fc_6_wav=nn_layer3(h_pool6_flat_2_wav,384,192,'frame_wav_fully_connect2')
        fc_6_normed_wav=tf.nn.relu(batch_norm(fc_6_wav,is_training2,'bn9'))
        fc_6_nnout_wav=nn_layer3(fc_6_normed_wav,192,n_class,'frame_wav_fully_connet3')
        pred2_wav=tf.nn.softmax(fc_6_nnout_wav)

print(pred_wav.shape)

with tf.name_scope('Three_all'):
    out_all_final=tf.concat([h_pool6_wav,h_pool6,i_h_pool6],axis=3)
    out_all_conv_final=conv2d_with_batch_norm(out_all_final,[1,1,768*3,768],1,1,'all_1',act=tf.nn.relu)
    pred_in_final=tf.reshape(out_all_conv_final,[-1,768])
    in_all_final=nn_layer3(pred_in_final,768,n_class,'all_connet1')
    pred_final=tf.nn.softmax(in_all_final)
    print(pred_final.shape)
    


vars=tf.trainable_variables()
params_mel=[v for v in vars if (v.name.startswith('New_3_3_with_bn_fully_conv2_with_three_pred_for_log_mel'))]
params_wav=[v for v in vars if (v.name.startswith('New_Inception_fully_conv_for_frame_wav1_with_two_pred'))]
params_imel=[v for v in vars if (v.name.startswith('New_3_3_with_bn_fully_conv2_with_three_pred_for_ilog_mel'))]
#params_imel_mel=[v for v in vars if (v.name.startswith('log_mel_and_ilog_mel'))]
params_train=[v for v in vars if (v.name.startswith('Three_all'))]

print('pred')
with tf.name_scope('cross_entropy'):
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred_final+1e-10),reduction_indices=[1]))
tf.summary.scalar('cross_entropy',cost)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost,var_list=params_train)

with tf.name_scope('accuracy'):
    #pred_final=0.1*pred2+0.8*pred+0.1*pred3
    predictions = tf.argmax(pred_final, 1)
    actuals = tf.argmax(y, 1)
    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)
    with tf.name_scope('correct_prediction'):
        correct_pred = tf.equal(tf.argmax(pred_final,1),tf.argmax(y,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        tp_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals), tf.equal(predictions, ones_like_predictions)), "float"))
        tn_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals),tf.equal(predictions, zeros_like_predictions)),"float"))
        fp_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, zeros_like_actuals), tf.equal(predictions, ones_like_predictions)), "float"))
        fn_op = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals), tf.equal(predictions, zeros_like_predictions)), "float"))
        tpr = (tp_op)/((tp_op) +(fn_op))
        tnr= (tn_op)/((tn_op)+(fp_op))
tf.summary.scalar('accuracy',accuracy)


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9) 
init = tf.initialize_all_variables()
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
merged=tf.summary.merge_all()

sess.run(init)
saver=tf.train.Saver()
swf_model_path=swf_net_model+'/save_model/0.0002/save1_met.ckpt'
ims_net_path=ims_net_model+'/save_model/0.0002/save1_met.ckpt'
ms_net_path=ms_net_model+'/save_model/0.0002/save1_met.ckpt'
tf.train.Saver(params_mel).restore(sess,ms_net_path)
tf.train.Saver(params_imel).restore(sess,ims_net_path)
tf.train.Saver(params_wav).restore(sess,swf_model_path)


total_batch=int(ntrain/batch_size1)
j=[i for i in range(total_batch)]
train_data=np.zeros((total_batch,768*3))
train_label=np.zeros((total_batch,2))
for i in j:      
    batch_xs=train_x[i*batch_size1:(i+1)*batch_size1]
    batch_ys =train_y[i*batch_size1:(i+1)*batch_size1]
    batch_xs = batch_xs.reshape((batch_size1,fbin,nframes))
    batch_xs_i=train_x_i[i*batch_size1:(i+1)*batch_size1]
    batch_xs_i=batch_xs_i.reshape((batch_size1,fbin,nframes))
    batch_xs_wav=train_x_wav[i*batch_size1:(i+1)*batch_size1]
    batch_xs_wav=batch_xs_wav.reshape((batch_size1,nframes,dim))
    train_output=sess.run(out_all_final,feed_dict={x:batch_xs,y:batch_ys,x_i:batch_xs_i,x_wav:batch_xs_wav,keep_prob:0.5,is_training2:1,batch_size:batch_size1})
    train_output2=sess.run(y,feed_dict={y:batch_ys})
    train_data[i]=train_output
    train_label[i]=batch_ys
    if i == 0:
        print('train：')
        print(train_output)
    if i==1:
        print('train：')
        print(train_output)
        
  

test_total_batch=int(n_test/batch_size1)
j_t=[i_t for i_t in range(test_total_batch)]
test_data=np.zeros((test_total_batch,768*3))
test_label=np.zeros((test_total_batch,2))
for i_t in j_t:
    test_xs=test_x[i_t*batch_size1:(i_t+1)*batch_size1]
    test_ys =test_y[i_t*batch_size1:(i_t+1)*batch_size1]
    test_xs = test_xs.reshape((batch_size1,fbin,nframes))
    test_xs_i=test_x_i[i_t*batch_size1:(i_t+1)*batch_size1]
    test_xs_i=test_xs_i.reshape((batch_size1,fbin,nframes))
    test_xs_wav=test_x_wav[i_t*batch_size1:(i_t+1)*batch_size1]
    test_xs_wav=test_xs_wav.reshape((batch_size1,nframes,dim))
    test_output=sess.run(out_all_final,feed_dict={x:test_xs, y:test_ys,x_i:test_xs_i,x_wav:test_xs_wav,keep_prob:1.0,is_training2:0,batch_size:batch_size1})   
    test_data[i_t]=test_output 
    test_label[i_t]=test_ys
    if i_t == 0:
        print('test：')
        print(test_output)
    if i_t==1:
        print('test：')
        print(test_output)

val_total_batch=int(n_val/batch_size1)
j_v=[i_v for i_v in range(val_total_batch)]
val_data=np.zeros((val_total_batch,768*3))
val_label=np.zeros((val_total_batch,2))
for i_v in j_v:
    val_xs=val_x[i_v*batch_size1:(i_v+1)*batch_size1]
    val_ys =val_y[i_v*batch_size1:(i_v+1)*batch_size1]
    val_xs = val_xs.reshape((batch_size1, fbin,nframes))
    val_xs_i=val_x_i[i_v*batch_size1:(i_v+1)*batch_size1]
    val_xs_i = val_xs_i.reshape((batch_size1, fbin,nframes))
    val_xs_wav=val_x_wav[i_v*batch_size1:(i_v+1)*batch_size1]
    val_xs_wav = val_xs_wav.reshape((batch_size1,nframes,dim))  
    val_output=sess.run(out_all_final, feed_dict={x:val_xs, y:val_ys,x_i:val_xs_i,x_wav:val_xs_wav,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
    val_data[i_v]=val_output
    val_label[i_v]=val_ys
    if i_v == 0:
        print('val：')
        print(val_output)
    if i_v==1:
        print('val：')
        print(val_output)
    
train_save_name=predict_path+'/train.mat'
sio.savemat(train_save_name,{'train_x':train_data,'train_y':train_label})

val_save_name=predict_path+'/val.mat'
sio.savemat(val_save_name,{'val_x':val_data,'val_y':val_label})

test_save_name=predict_path+'/mismatch.mat'
sio.savemat(test_save_name,{'test_x':test_data,'test_y':test_label})
         


print ("Finished .")
