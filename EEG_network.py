#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun July 10 16:41 2018

@author: QWQ

EEG_LSTM
请不要用中文命名目录，中文目录中看不到任何图形
"""

import tensorflow as tf
import numpy as np

class NETWORK(): 
    def __init__(self, batch_size = 32, time_steps = 14, input_dim = 3, classes = 3):    
        self.time_steps = time_steps
        self.input_dim = input_dim
        self.classes = classes
        self.batch_size = batch_size
        self.n_hidden = 32
    def __call__(self,inputs):
        EEG = ['theta','alpha','low_bata','high_bata','gamma']
        flatten = []
        for i in range(len(EEG)):
            with tf.variable_scope(EEG[i]):    
                flatten.append(self.LSTM(inputs[:,i]))  ##10(BATCH_SIZE) 5(TIME_STEPS) 5(input_dim)
                # features = tf.nn.dropout(flatten, 0.5)
                # logits = tf.layers.dense(features, self.classes, name='logits')
                # variables = [var for var in tf.trainable_variables() if var.name.startswith(EEG[i])]
        final_flatten = tf.concat(flatten,axis=-1,name='concatenate')
        final_features = tf.nn.dropout(final_flatten, 0.5)
        final_logits = tf.layers.dense(final_features, self.classes, name='final_logits')
        final_variables = [var for var in tf.trainable_variables()]
        return final_logits, final_variables

    def LSTM(self,input):
        """Function returns a TensorFlow RNN with two stacked LSTM cells

        Two LSTM cells are stacked which adds deepness to the neural network.
        Note, some code of this notebook is inspired from an slightly different
        RNN architecture used on another dataset, some of the credits goes to
        "aymericdamien".

        Args:
            input:     ndarray feature matrix, shape: [batch_size, time_steps, input_dim]
        Return:
            _Y:     matrix  output shape [batch_size,n_classes]
        """
        with tf.name_scope('LSTM_input'):
            # input shape: (batch_size, TIME_STEPS, input_dim)
            _X = tf.transpose(input, [1, 0, 2])  # permute n_steps and batch_size
            # Reshape to prepare input to hidden activation
            _X = tf.reshape(_X, [-1,self.input_dim ])
            # new shape: (TIME_STEPS*batch_size, input_dim)

            # Linear activation
            _X = tf.nn.relu(tf.matmul(_X, tf.Variable(tf.random_normal([self.input_dim, self.n_hidden]))) +  tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)))
            #(TIME_STEPS*batch_size,N_HIDDEN)
            
            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            _X = tf.split(_X,self.time_steps, 0)
            # new shape: n_steps * (batch_size, n_hidden)
        with tf.name_scope('LSTM1'):
            # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
            lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('LSTM2'):
            lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('LSTMs'):
            lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
        with tf.name_scope('LSTM_output'):
            # Get LSTM cell output
            outputs, _ = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)
            lstm_last_output = outputs[-1]
        with tf.name_scope('flatten'):
            _Y = tf.contrib.layers.flatten(tf.matmul(lstm_last_output, tf.Variable(tf.random_normal([self.n_hidden, self.classes]))) + tf.Variable(tf.random_normal([self.classes])))
            # Get last time step's output feature for a "many to one" style classifier,
            # as in the image describing RNNs at the top of this page
        return _Y
    
if __name__ == '__main__':
    with tf.Graph().as_default():      
        labels = ['neutral','happy','sad']   
        channel = [3,7,9,12,16]    #### time_steps
        inputvector = [-2,-1,0,1,2]    #### input_dim
        EEG = ['theta','alpha','low_bata','high_bata','gamma']
        BATCH_SIZE = 10
        TIME_STEPS = len(channel)
        INPUT_DIM = len(inputvector)
        CLASSES = len(labels)
        NUM_EEG = len(EEG)
        inputs_pl = tf.placeholder(tf.float32,shape=(BATCH_SIZE,NUM_EEG,TIME_STEPS,INPUT_DIM))
        label_pl = tf.placeholder(tf.int32, shape=(BATCH_SIZE,CLASSES))      
        network =  NETWORK(BATCH_SIZE, TIME_STEPS, INPUT_DIM, CLASSES)
        logits, variables = network(inputs_pl)
        
        sess = tf.Session()
        #sess.run(tf.initialize_all_variables())initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
        sess.run(tf.global_variables_initializer())
        summary = tf.summary.merge_all() 
        summary_writer = tf.summary.FileWriter('summary', sess.graph)  
        '''
        for i in range(1000):
            sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
            if(i%50==0): #每50次写一次日志
                result = sess.run(summary,feed_dict={xs:x_data,ys:y_data}) #计算需要写入的日志数据
                summary_writer.add_summary(result,i) #将日志数据写入文件
        print("EEG_LSTM NetWork")
        ''' 
