#-*- coding=utf-8 -*-
import sys
import os
import platform
import time
import ctypes

from array import *
from ctypes import *
from __builtin__ import exit
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from EEG_network import NETWORK
from EEG_train import model_predict
if sys.platform.startswith('win32'):
    import msvcrt
elif sys.platform.startswith('linux'):
    import atexit
    from select import select

from ctypes import *

try:
    if sys.platform.startswith('win32'):
        libEDK = cdll.LoadLibrary("edk.dll")
    elif sys.platform.startswith('linux'):
        srcDir = os.getcwd()
	if platform.machine().startswith('arm'):
            libPath = srcDir + "/../../bin/armhf/libedk.so"
	else:
            libPath = srcDir + "/../../bin/linux64/libedk.so"
        libEDK = CDLL(libPath)
    else:
        raise Exception('System not supported.')
except Exception as e:
    print 'Error: cannot load EDK lib:', e
    exit()

IEE_EmoEngineEventCreate = libEDK.IEE_EmoEngineEventCreate
IEE_EmoEngineEventCreate.restype = c_void_p
eEvent = IEE_EmoEngineEventCreate()

IEE_EmoEngineEventGetEmoState = libEDK.IEE_EmoEngineEventGetEmoState
IEE_EmoEngineEventGetEmoState.argtypes = [c_void_p, c_void_p]
IEE_EmoEngineEventGetEmoState.restype = c_int

IEE_EmoStateCreate = libEDK.IEE_EmoStateCreate
IEE_EmoStateCreate.restype = c_void_p
eState = IEE_EmoStateCreate()

userID = c_uint(0)
user   = pointer(userID)
ready  = 0
state  = c_int(0)

alphaValue     = c_double(0)
low_betaValue  = c_double(0)
high_betaValue = c_double(0)
gammaValue     = c_double(0)
thetaValue     = c_double(0)

alpha     = pointer(alphaValue)
low_beta  = pointer(low_betaValue)
high_beta = pointer(high_betaValue)
gamma     = pointer(gammaValue)
theta     = pointer(thetaValue)

channelList = array('I',[3, 7, 9, 12, 16])   # IED_AF3, IED_AF4, IED_T7, IED_T8, IED_Pz

labels = ['neutral','happy','sad']   
channel = [3,7,9,12,16]    #### time_steps
inputvector = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]    #### input_dim
EEG = ['theta','alpha','low_bata','high_bata','gamma']

TIME_STEPS = len(channel)
INPUT_DIM = len(inputvector)
CLASSES = len(labels)
NUM_EEG = len(EEG)

print("test")   
n_train,[X_train,y_train],n_valid,[X_valid,y_valid] = getdata_random("/home/xd/qh/EEG_FINAL/data/") 
ckpt = tf.train.get_checkpoint_state('./ckpt/')
print(ckpt.model_checkpoint_path)
print(tf.train.latest_checkpoint('./ckpt/'))

graph = tf.Graph().as_default()
inputs_pl = tf.placeholder(tf.float32,shape=(1,NUM_EEG,TIME_STEPS,INPUT_DIM))
network =  NETWORK(1, TIME_STEPS, INPUT_DIM, CLASSES)
logits, variables = network(inputs_pl)
predicts = model_predict(logits) 
sess = tf.Session(graph=graph)
saver = tf.train.Saver()
saver.restore(sess,ckpt.model_checkpoint_path)

# -------------------------------------------------------------------------
print "==================================================================="
print "Example to get the average band power for a specific channel from" \
" the latest epoch."
print "==================================================================="

# -------------------------------------------------------------------------
if libEDK.IEE_EngineConnect("Emotiv Systems-5") != 0:
        print "Emotiv Engine start up failed."
        exit();

print "Theta, Alpha, Low_beta, High_beta, Gamma \n"
while (1):
    state = libEDK.IEE_EngineGetNextEvent(eEvent)
    if state == 0:
        eventType = libEDK.IEE_EmoEngineEventGetType(eEvent)
        libEDK.IEE_EmoEngineEventGetUserId(eEvent, user)
        if eventType == 16:  # libEDK.IEE_Event_enum.IEE_UserAdded
            ready = 1
            libEDK.IEE_FFTSetWindowingType(userID, 1);  # 1: libEDK.IEE_WindowingTypes_enum.IEE_HAMMING
            print "User added"
        if ready == 1:
            thetalist = []
            alphalist = []
            low_batalist = []
            high_betalist = []
            gammalist = []
            for i in channelList:
                result = c_int(0)
                result = libEDK.IEE_GetAverageBandPowers(userID, i, theta, alpha, low_beta, high_beta, gamma)
                if result == 0:    #EDK_OK
                    if thetaValue.value!=None: 
                        thetaValue.value = min(thetaValue.value,20)
                        alphaValue.value = min(alphaValue.value,30)
                        low_betaValue.value = min(low_betaValue.value,20)
                        high_betaValue.value = min(high_betaValue.value,20)
                        gammaValue.value = min(gammaValue.value,25)
                        thetalist.append(thetaValue.value)
                        alphalist.append(alphaValue.value)
                        low_batalist.append(low_betaValue.value)
                        high_betalist.append(high_betaValue.value)
                        gammalist.append(gammaValue.value)
                    else:
                        thetalist = []
                        alphalist = []
                        low_batalist = []
                        high_betalist = []
                        gammalist = []
                    if len(thetalist)>=INPUT_DIM:
                        for start, end in zip(range(0,n_valid, 1),range(1, n_valid + 1, 1)):
                            predict = sess.run(predicts,feed_dict={inputs_pl: X_valid[start:end]})
                            print(labels[predict[0]])
                            thetalist = []
                            alphalist = []
                            low_batalist = []
                            high_betalist = []
                            gammalist = []

    elif state != 0x0600:
        print "Internal error in Emotiv Engine ! "
# -------------------------------------------------------------------------
sess.close()
graph.close()
libEDK.IEE_EngineDisconnect()
libEDK.IEE_EmoStateFree(eState)
libEDK.IEE_EmoEngineEventFree(eEvent)
