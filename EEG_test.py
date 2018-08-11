#!/usr/bin/python3 python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from EEG_network import NETWORK
import EEG_train 
import numpy as np

"""
Created on Sun July 10 16:41 2018

@author: QWQ

测试文件
.meta文件保存了当前图结构
.index文件保存了当前参数名
.data文件保存了当前参数值
"""

#由于实际上我们参数保存的都是Variable变量的值，所以其他的参数值（例如batch_size）等，我们在restore时可能希望修改，但是图结构在train时一般就已经确定了，所以我们可以使用tf.Graph().as_default()新建一个默认图（建议使用上下文环境），利用这个新图修改和变量无关的参值大小，从而达到目的。
def test(n): 
    print("test")   
    n_train,[X_train,y_train],n_valid,[X_valid,y_valid] = EEG_train.getdata_random("/home/xd/qh/EEG_FINAL/data/") 
    ckpt = tf.train.get_checkpoint_state('./ckpt/')
    print(ckpt.model_checkpoint_path)
    print(tf.train.latest_checkpoint('./ckpt/'))
    if n==0:#### 不改batch_size
        with tf.Graph().as_default() as graph:
            sess = tf.Session(graph=graph)
            saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
            saver.restore(sess,ckpt.model_checkpoint_path)
            x_= graph.get_tensor_by_name('x_:0')
            predicts = graph.get_tensor_by_name('predict/predict:0')
            for start, end in zip(range(0,n_valid, EEG_train.BATCH_SIZE),range(EEG_train.BATCH_SIZE, n_valid + 1, EEG_train.BATCH_SIZE)):
                Y = X_valid[start:end].astype(np.float32)
                predict = sess.run(predicts,feed_dict={x_: Y})
                print(EEG_train.labels[predict[0]])
            sess.close()
    if n==1:#### 改batch_size
        with tf.Graph().as_default() as graph:
            inputs_pl = tf.placeholder(tf.float32,shape=(1,EEG_train.NUM_EEG,EEG_train.TIME_STEPS,EEG_train.INPUT_DIM))
            network =  NETWORK(1, EEG_train.TIME_STEPS, EEG_train.INPUT_DIM, EEG_train.CLASSES)
            logits, variables = network(inputs_pl)
            predicts = EEG_train.model_predict(logits) 
            sess = tf.Session(graph=graph)
            saver = tf.train.Saver()
            saver.restore(sess,ckpt.model_checkpoint_path)
            for start, end in zip(range(0,n_valid, 1),range(1, n_valid + 1, 1)):
                predict = sess.run(predicts,feed_dict={inputs_pl: X_valid[start:end]})
                print(EEG_train.labels[predict[0]])
            sess.close()
            
if __name__ == '__main__':                
    test(0)
