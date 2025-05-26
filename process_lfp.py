"---得到LFP信号---"
from scipy import io
import os
import numpy as np
import h5py
from scipy import signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import math
import pickle
import scipy.io as sio


#单天处理数据，文件夹
data_file='data/LFP/'
data_folder_out="data/LFP/"

files_days = ['天数日期'] # files_days = ['indy_20170124_01']


for day in files_days:
    print("正在处理"+day+".mat")
    file1_path=data_file+day+".mat"
    data1=h5py.File(file1_path)
    rawdata=data1['rawdata'][:]
    t=data1['t'][:]

    rawdata=rawdata.astype(np.float32)
    print(t.shape[0])
    #CAR
    rawaverage=np.average(rawdata,axis=1)
    rawdata=rawdata-rawaverage[:,np.newaxis]
    #CAR完成

    #4阶巴特沃斯高通滤波：300Hz
    Fs=1/(t[0,2]-t[0,1])
    ws=300
    wn=ws/(0.5*Fs)
    b,a=signal.butter(4,wn,'highpass')
    num_chan=rawdata.shape[1]
    for i in range(num_chan):
        rawdata[:,i]=signal.filtfilt(b,a,rawdata[:,i])
    #完成

    #full-wave rectified(taking the absolute value)
    rawdata=abs(rawdata)
    #完成

    #1阶巴特沃斯低通滤波：12Hz
    ws=12
    wn=ws/(0.5*Fs)
    b, a = signal.butter(1, wn, 'lowpass')
    for i in range(num_chan):
        rawdata[:,i]=signal.filtfilt(b,a,rawdata[:,i])
    #完成


    #降采样至1Khz
    num_step=round(Fs/1000)
    num_step=int(num_step)
    num_t=t.shape[1]
    index1=range(0,num_t,num_step)
    new_rawdata=rawdata[index1,:]
    new_t=t[0,index1]



    with open(data_folder_out+day+'.pickle','wb') as f:
        pickle.dump([new_rawdata,new_t],f)




    print(day+".mat  处理完毕!")

