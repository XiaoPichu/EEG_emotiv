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
#channelList = array('I',[i+3 for i in xrange(14)])
# IED_AF3, IED_F7, IED_F3, IED_FC5, IED_T7, IED_P7, IED_Pz, IED_O2, IED_P8, IED_T8, IED_FC6, IED_F4, IED_F8, IED_AF4

# -------------------------------------------------------------------------
print "==================================================================="
print "Example to get the average band power for a specific channel from" \
" the latest epoch."
print "==================================================================="

# -------------------------------------------------------------------------
if libEDK.IEE_EngineConnect("Emotiv Systems-5") != 0:
        print "Emotiv Engine start up failed."
        exit();

label_names = ['neutral','happy','sad']
label = label_names[2]

print "Theta, Alpha, Low_beta, High_beta, Gamma \n"
with open('./results/byd/'+label+'.txt', 'a') as f:
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
#                        thetalist.append(thetaValue.value)
#                        alphalist.append(alphaValue.value)
#                        low_batalist.append(low_betaValue.value)
#                        high_betalist.append(high_betaValue.value)
#                        gammalist.append(gammaValue.value)
                        f.write(str(thetaValue.value)+',')
                        f.write(str(alphaValue.value)+',')
                        f.write(str(low_betaValue.value)+',')
                        f.write(str(high_betaValue.value)+',')
                        f.write(str(gammaValue.value)+',')
                f.write('\n')

        elif state != 0x0600:
            print "Internal error in Emotiv Engine ! "
# -------------------------------------------------------------------------
libEDK.IEE_EngineDisconnect()
libEDK.IEE_EmoStateFree(eState)
libEDK.IEE_EmoEngineEventFree(eEvent)
