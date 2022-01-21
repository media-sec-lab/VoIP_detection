# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:07:22 2019

@author: Administrator
"""

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
from sklearn.metrics import roc_auc_score
import tensorflow as tf
#from random import shuffle
import random
import h5py
import numpy as np
tf.set_random_seed(3) 
mat_load_path=('/VoIP_detection/VoIP_detection_data/VPCID_wav/vad/spilt_clips/case0/mat')
model_save_path=('/VoIP_detection/VoIP_detection_code/python/Feature_extraction/ms_net')

load_name=mat_load_path+'/train_norm_per_sample_logmel.mat'
model_path=model_save_path+"/save_model/0.0002/save1_met.ckpt"



load_data=h5py.File(load_name)

test_x=load_data['mismatch_x']
test_y=load_data['mismatch_y']


test_x=test_x[0:,0:,0:]
test_y=test_y[0:,0:]



test_x=test_x.transpose((0,2,1))
test_y=test_y.transpose((1,0))



fbin=test_x.shape[1] #input timesteps
#timesteps=1
nframes=test_x.shape[2]

n_class=test_y.shape[1]
ntest=test_x.shape[0]




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

test_acc_max=0.0
test_tpr_max=0.0
test_tnr_max=0.0
test_acc_max_epoch=0



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
    #initial=tf.constant(0.1,shape=shape)
    return tf.Variable(tf.zeros(shape)+0.1,name=in_name)

def variable_summaries(var,name):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar(name+'/mean',mean)
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar(name+'/stddev',stddev)
        tf.summary.scalar(name+'/max',tf.reduce_max(var))
        tf.summary.scalar(name+'/min',tf.reduce_min(var))
        tf.summary.histogram(name+'/weights',var) 
    

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

    
    
def Inception_myself_fully_conv2(x, scope):   
    with tf.name_scope(scope):
        split_conv_x12=conv2d_with_batch_norm(x,[3,3,1,48],1,1,'log_mel_conv1_2',act=tf.nn.relu)
     
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
        i_shape=shape[1]*shape[2]*shape[3]
        h_pool6_flat=tf.reshape(h_pool6,[-1,i_shape])
        h_pool6_nnout=nn_layer3(h_pool6_flat,768,n_class,'log_mel_fully_connect1')
        out=tf.nn.softmax(h_pool6_nnout)
        
        
        x4_avg_pool=tf.reduce_mean(x4,[1,2],name='avgpool_x4',keep_dims=True)
        h_pool4_flat=tf.reshape(x4_avg_pool,[-1,192])     
        h_pool5_flat=nn_layer3(h_pool4_flat,192,96,'log_mel_fully_connect2')
        h_pool5_flat_normed=tf.nn.relu(batch_norm(h_pool5_flat,is_training2,'bn8'))
        h_pool5_nnout=nn_layer3(h_pool5_flat_normed,96,n_class,'log_mel_fully_connect3')
        out2=tf.nn.softmax(h_pool5_nnout)

        return out,out2   

    
x_input=tf.reshape(x,[-1,fbin,nframes,1])
print(x_input.shape)
[pred,pred2]=Inception_myself_fully_conv2(x_input,scope='New_3_3_with_bn_fully_conv2_with_three_pred_for_log_mel')
print(pred.shape)
print(pred2.shape)




print('pred')
with tf.name_scope('cross_entropy'):
    #cost1 = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred+1e-10),reduction_indices=[1]))
    #tf.add_to_collection("losses",cost1)
    #cost=tf.add_n(tf.get_collection("losses"))
    cost1 = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred+1e-10),reduction_indices=[1]))
    cost2 = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred2+1e-10),reduction_indices=[1]))
    #cost3 = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred3+1e-10),reduction_indices=[1]))
    cost=0.8*cost1+0.2*cost2
tf.summary.scalar('cross_entropy',cost)

with tf.name_scope('train'):
    #optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate,0.9).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
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


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.23) 
init = tf.initialize_all_variables()
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
merged=tf.summary.merge_all()
sess.run(init)
saver=tf.train.Saver()
vars=tf.trainable_variables()
params_load=[v for v in vars if (v.name.startswith('New_3_3_with_bn_fully_conv2_with_three_pred_for_log_mel'))]
tf.train.Saver(params_load).restore(sess,model_path)


print("MS-net Start prediction") 
test_loss=0.0
test_tp=0
test_tn=0
test_fp=0
test_fn=0
test_acc=0.0
test_tpr=0.0
test_tnr=0.0
test_FPR=0.0
test_FNR=0.0
precision=0.0
recall=0.0
F1_score=0.0
test_auc=0.0
test_total_batch=int(ntest/batch_size1)
j_t=[i_t for i_t in range(test_total_batch)]
rr=random.random
random.shuffle(j_t,random=rr)
test_pro=[]
test_label=[]
for i_t in j_t:
    test_xs=test_x[i_t*batch_size1:(i_t+1)*batch_size1]
    test_ys =test_y[i_t*batch_size1:(i_t+1)*batch_size1]
    test_xs = test_xs.reshape((batch_size1, fbin,nframes))
    test_acc+=sess.run(accuracy, feed_dict={x:test_xs, y:test_ys,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
    
    #test_tpr+=sess.run(tpr, feed_dict={x:test_xs, y:test_ys,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
    #test_tnr+=sess.run(tnr, feed_dict={x:test_xs, y:test_ys,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
    test_loss +=sess.run(cost,feed_dict={x:test_xs,y:test_ys,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
    test_tp+=sess.run(tp_op, feed_dict={x:test_xs, y:test_ys,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
    test_tn+=sess.run(tn_op, feed_dict={x:test_xs, y:test_ys,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
    test_fp+=sess.run(fp_op, feed_dict={x:test_xs, y:test_ys,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
    test_fn+=sess.run(fn_op, feed_dict={x:test_xs, y:test_ys,keep_prob:1.0,is_training2:0,batch_size:batch_size1})
    
    test_y_l=sess.run(y, feed_dict={y:test_ys})
    test_y_p=sess.run(pred, feed_dict={x:test_xs, y:test_ys,is_training2:0})
    test_y_l=test_y_l.transpose(1,0)
    test_y_p=test_y_p.transpose(1,0)
    test_pro.append(test_y_p[1])
    test_label.append(test_y_l[1])

test_pro2=np.array(test_pro)
test_label2=np.array(test_label)
test_pro2=test_pro2.reshape([1,-1])
test_label2=test_label2.reshape([1,-1])
test_auc=roc_auc_score(test_label2[0],test_pro2[0])
test_acc=test_acc/test_total_batch

test_tpr=test_tp/(test_tp+test_fn)
test_tnr=test_tn/(test_tn+test_fp)
test_FPR=test_fp/(test_fp+test_tn)
test_FNR=test_fn/(test_tp+test_fn)
precision=test_tp/(test_tp+test_fp)
recall=test_tp/(test_tp+test_fn)
F1_score=2*precision*recall/(precision+recall)

print (" Mismatch accuracy: %.4f TPR: %.4f TNR: %.4f" % ((test_acc,test_tpr,test_tnr)))
print (" Mismatch FPR: %.4f FNR: %.4f AUC: %.4f" % ((test_FPR,test_FNR,test_auc)))
print (" Mismatch precision: %.4f recall: %.4f F1_score: %.4f" % ((precision,recall,F1_score)))
print (" MS-net Finished .")



print ("Finished .")
