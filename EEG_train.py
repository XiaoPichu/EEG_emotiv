#!/usr/bin/python3 python3
# -*- coding: utf-8 -*-
"""
Created on Sun July 10 16:41 2018

@author: QWQ

训练流程
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import time
from matplotlib import pyplot as plt
import pandas as pd
from EEG_network import NETWORK
import tensorflow as tf
from datetime import datetime
import logging
import sys
import glob    
from random import shuffle,random

def initLogging(logFilename='terminal.log'):
  """Init for logging
  """
  logging.basicConfig(
                    level= logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)
    
initLogging()

def model_loss(logits1, labels): #### batch_size,classes;batch_size,classes
    #### sparse_softmax_cross_entropy_with_logits中 lables接受直接的数字标签
    #### 如[1], [2], [3], [4] （类型只能为int32，int64）
    #### softmax_cross_entropy_with_logits中 labels接受one-hot标签
    #### 如[1,0,0,0], [0,1,0,0],[0,0,1,0], [0,0,0,1] （类型为int32， int64）
    with tf.variable_scope('caculate_loss') :
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits1, name='cross_entropy')
        #labels = np.argmax(labels,axis=-1)
        #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits1, name='corss_entropy1')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')  #### 得到均值
        tf.summary.scalar('cross_loss', cross_entropy_mean) #### 添加标量统计结果
    return cross_entropy_mean

def model_training(variables1, loss1): 
    def make_optimizer(loss, variables, name='Adam'):
        """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
            and a linearly decaying rate that goes to zero over the next 100k steps
        """
        global_step = tf.Variable(0, trainable=False)
        learning_rate = 1e-4
        beta1 = 0.5
        learning_step = tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name).minimize(loss, global_step=global_step, var_list=variables)
        return learning_step  
    optimizer = make_optimizer(loss1, variables1, name='Adam_1')
    ema = tf.train.ExponentialMovingAverage(decay=0.95)
    assert  len([var for var in tf.trainable_variables()])==len(variables1)
    update_losses = ema.apply([loss1])

    return tf.group(update_losses, optimizer)

def model_predict(logits):
    with tf.variable_scope('predict') :    
        predicts = tf.argmax(logits, axis=-1, name='predict')
        return predicts

def one_hot(y_):
    """
    Function to encode output labels from number indexes.
    E.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    """
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

def getdata_random(path):
    csvs = glob.glob(path+'*.txt')
    X_train = []
    y_train = []
    X_valid = []
    y_valid = []
    for csv in csvs:
        with open(csv, 'r') as f:
            file = f.readlines()
            N_LINES = len(file)
            n_start = (INPUT_DIM-1)//2
            n_end = N_LINES - n_start
            index=list(range(n_start,n_end,1))
            shuffle(index)         
            train_ratio = int(0.7*len(index))
            for i in range(len(index)):
                if i < train_ratio:
                    if random() > 0.5:
                        continue
                    lines_content = np.zeros((NUM_EEG,TIME_STEPS,INPUT_DIM),dtype=np.float32)
                    for j in range(INPUT_DIM):                
                        tmpcontent = file[index[i]-(n_start-j)].strip('\n').split(',')
                        content = list(map(float,tmpcontent[:25]))
                        content = np.array(content,dtype=np.float32)
                        content = np.reshape(content,(NUM_EEG,TIME_STEPS))
                        lines_content[:,:,j] = content			
                    X_train.append(lines_content)
                    y_train.append([j for j in range(CLASSES) if labels[j] in csv]) #### label从0开始
                else:
                    lines_content = np.zeros((NUM_EEG,TIME_STEPS,INPUT_DIM),dtype=np.float32)
                    for j in range(INPUT_DIM):                
                        tmpcontent = file[index[i]-(n_start-j)].strip('\n').split(',')
                        content = list(map(float,tmpcontent[:25]))
                        content = np.array(content,dtype=np.float32)
                        content = np.reshape(content,(NUM_EEG,TIME_STEPS))
                        lines_content[:,:,j] = content			
                    X_valid.append(lines_content)
                    y_valid.append([j for j in range(CLASSES) if labels[j] in csv]) #### label从0开始
    
    X_train = np.array(X_train,dtype=np.float32)# n,NUM_EEG,TIME_STEPS,INPUT_DIM
    X_valid = np.array(X_valid,dtype=np.float32)
    y_train = np.array(y_train,dtype=np.int32)
    y_valid = np.array(y_valid,dtype=np.int32)   
    y_train = one_hot(y_train)
    y_valid = one_hot(y_valid)
    return y_train.shape[0],[X_train,y_train],y_valid.shape[0],[X_valid,y_valid]

def dataprocess(path):
    csvs = glob.glob(path+'*.txt')
    X_signals = []
    y_signals = []
    for csv in csvs:           
        if csv !=csvs[2]:
            continue
        with open(csv, 'r') as f:         
            file = f.readlines()
            N_LINES = len(file)
            theta = locals()
            for i in range(TIME_STEPS):
                theta['channel_%d'%i] = []
            for line in range(N_LINES):
                tmpcontent = file[line].strip('\n').split(',')
                content = list(map(float,tmpcontent[:25]))
                for i in range(len(content)):
                    theta['channel_%d'%(i//NUM_EEG)].append(content[i//NUM_EEG])	
        plt.figure()
        for i in range(TIME_STEPS):                   
            plt.plot(list(range(len(theta['channel_%d'%i]))),theta['channel_%d'%i],'r-o',label="channel_%d'%i",linewidth=1)
            plt.show()
            plt.close()

DATA_PATH = "results/"
DATAFLAG = ["train/","valid/"]

labels = ['neutral','happy','sad']   
channel = [3,7,9,12,16]    #### time_steps
inputvector = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]    #### input_dim
EEG = ['theta','alpha','low_bata','high_bata','gamma']

BATCH_SIZE = 20
EPOCHES = 25
checkpoint_epoches = 5
checkpoint_dir = 'ckpt'
checkpoint_file = os.path.join(checkpoint_dir, 'model.ckpt')

TIME_STEPS = len(channel)
INPUT_DIM = len(inputvector)
CLASSES = len(labels)
NUM_EEG = len(EEG)
n_train = 0
n_valid = 0

graph_dir='summary'

def run_train():       
    with tf.Graph().as_default():      
        #### 得到 EEGs_valid, labels_valid,EEGs_train, labels_train		
        path = "/home/xd/qh/EEG_FINAL/data/"
        assert path=="/home/xd/qh/EEG_FINAL/data/"
        n_train,[X_train,y_train],n_valid,[X_valid,y_valid] = getdata_random(path) 
        
        #### 网络结构 
        inputs_pl = tf.placeholder(tf.float32,shape=    (BATCH_SIZE,NUM_EEG,TIME_STEPS,INPUT_DIM),name = "x_")
        label_pl = tf.placeholder(tf.int32, shape=(BATCH_SIZE,CLASSES),name="y_")      
        network =  NETWORK(BATCH_SIZE, TIME_STEPS, INPUT_DIM, CLASSES)
        logits, variables = network(inputs_pl)
        predicts = model_predict(logits)  
        correct_pred = tf.equal(predicts,tf.argmax(label_pl, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))   #### 求出平均值
        losses = model_loss(logits, label_pl)
        train_op = model_training(variables, losses)
        
        #### 建立会话
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        summary = tf.summary.merge_all() 
        summary_writer = tf.summary.FileWriter(graph_dir, sess.graph)  
        saver = tf.train.Saver(max_to_keep=50,keep_checkpoint_every_n_hours=1) #### 最多保留50个模型每1h至少一个		
		#### 开始训练   
        train_index = list(range(n_train))
        for i in range(EPOCHES):  
            print('*****************')
            print('train')  
            shuffle(train_index)
            batch_steps = 1
            train_accuracy_outs = [] 
            train_loss_outs = []
            for start, end in zip(range(0,n_train, BATCH_SIZE),range(BATCH_SIZE, n_train + 1, BATCH_SIZE)):
                accuracy_train, _updateloss, loss_value, summary_str, predicts_value = \
                                            sess.run([accuracy,train_op,losses,summary,predicts], 
                                                      feed_dict={inputs_pl: X_train[train_index[start:end]],
                                                                 label_pl: y_train[train_index[start:end]]})
                train_accuracy_outs.append(accuracy_train)
                train_loss_outs.append(loss_value)
                summary_writer.add_summary(summary_str, batch_steps+i*BATCH_SIZE)
                summary_writer.flush()
                batch_steps += 1
            if (i + 1) % checkpoint_epoches == 0:
                saver.save(sess, checkpoint_file, global_step=i+1)             
                #saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            # 每一代验证
            print('valid')           
            batch_steps = 0
            valid_accuracy_outs = []
            valid_loss_outs = []
            for start, end in zip(range(0,n_valid, BATCH_SIZE),range(BATCH_SIZE, n_valid + 1, BATCH_SIZE)):
                accuracy_out, loss_out = sess.run([accuracy, losses], 
                                                   feed_dict={inputs_pl: X_valid[start:end],
                                                              label_pl: y_valid[start:end]})
                valid_accuracy_outs.append(accuracy_out)
                valid_loss_outs.append(loss_out)
                batch_steps += 1                    
            logging.info("\n training iter: {},".format(i+1) +
                  " train accuracy : {}".format(sum(train_accuracy_outs)/len(train_accuracy_outs)) +
                  " train loss : {}".format(sum(train_loss_outs)/len(train_loss_outs))+
                  " valid accuracy : {},".format(sum(valid_accuracy_outs)/len(valid_accuracy_outs)) +
                  " valid loss : {}".format(sum(valid_loss_outs)/len(valid_loss_outs)))
        summary_writer.close()
        sess.close()

if __name__ == '__main__':
    run_train()
