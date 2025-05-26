# 从LFP中提取LMP特征
import pickle
import h5py
import numpy as np


def find_index(end_t,lfpt,search_start):
    for i in range(search_start,lfpt.shape[0]):
        if lfpt[i]>=end_t:
            return i



def get_bin_LFP(lfpdata,lfpt,cursor_pos,pos_t):
    LFP_bin_data=[]
    LFP_bin_pos=[]
    start_pos=0
    end_pos=17
    search_start=0
    while end_pos<=(cursor_pos.shape[1]-1):
        end_t=pos_t[0,end_pos-1]
        end_lfp=find_index(end_t,lfpt,search_start)
        if end_lfp<256:
            start_pos=start_pos+16
            end_pos=start_pos+17
            search_start=end_lfp
            continue
        if end_lfp+1>lfpt.shape[0]:
            break
        lfp_mat=lfpdata[end_lfp-255:end_lfp+1,:]
        LFP_bin_data=LFP_bin_data+[lfp_mat]
        pos_mat=np.mean(cursor_pos[:,start_pos:end_pos],axis=1)
        LFP_bin_pos=LFP_bin_pos+[pos_mat]
        start_pos=start_pos+16
        end_pos=start_pos+17
        search_start=end_lfp
    return np.array(LFP_bin_data),np.array(LFP_bin_pos)

def getlmp(lfpdata,lfpt,cursor_pos_filter,pos_t):
# lfpdata (n*96)  lfpt (n,)1维  cursor_pos_filter (2,125000)  pos_t (1,125000)二维
    lmp=[]
    lmppos=[]
    numpoint=cursor_pos_filter.shape[1]
    search_start=0
    for i in range(numpoint):
        now_pos_t=pos_t[0,i]
        end_lfp=find_index(now_pos_t,lfpt,search_start)
        if end_lfp<256:
            search_start=end_lfp
            continue
        mat1=lfpdata[end_lfp-256:end_lfp,:]
        mean_mat1=np.average(mat1,axis=0)
        lmp=lmp+[mean_mat1]
        lmppos=lmppos+[cursor_pos_filter[:,i]]
        search_start=end_lfp

    return np.array(lmp),np.array(lmppos)


def meanfilter(lfpdata,lfpt):
    lmp=[]
    lmpt=[]
    numpoint=lfpdata.shape[0]
    startlmp=0
    endlmp=256
    startt=endlmp-256
    while(endlmp<=numpoint):
        mat1=lfpdata[startlmp:endlmp,:]
        mat2=lfpt[startt:endlmp]
        mat1_mean=np.mean(mat1,axis=0)
        mat2_mean=np.mean(mat2)
        lmp=lmp+[mat1_mean]
        lmpt=lmpt+[mat2_mean]
        startlmp=startlmp+4
        endlmp=startlmp+256
        startt=endlmp-256
    lmp=np.array(lmp)
    # lmp=lmp.reshape((-1,96)) #2维：第二维是96
    lmpt=np.array(lmpt)  #1维
    return lmp,lmpt

def align_lmp_cursor(lmp1,lmpt1,cursor_pos_filter,pos_t):
    pos_t=np.squeeze(pos_t)#将pos_t变为1维，消除多余的维度
    if (lmpt1[0]>pos_t[0])or(lmpt1[-1]<pos_t[-1]):
        print("error! time is not alignmented")
        return [],[]
    length1=len(lmpt1)#一般lmpt要长一些
    length2=len(pos_t)
    for i in range(1,length1):
        if (pos_t[0]<=lmpt1[i])and(pos_t[0]>=lmpt1[i-1]):
            print(i)
            break
    if (pos_t[0]-lmpt1[i-1])<(lmpt1[i]-pos_t[0]):
        start_index=i-1
    else:
        start_index=i
    lmp2=lmp1[start_index:start_index+length2,:]

    return lmp2,cursor_pos_filter


folder_LFP="data/LFP/"
folder_spike="data/spike/"

result_folder="data/LMP/"


files_days = ['天数日期'] # files_days = ['indy_20170124_01']

for day in files_days:

    file = day + ".pickle"
    print("正在处理"+file)
    filename_nosuffix=file[:-7]
    with open(folder_LFP+file,'rb') as f:
        lfpdata,lfpt=pickle.load(f,encoding='latin1')# lfpt:1维 --,lfpdata:--*96
        num_chan=lfpdata.shape[1]
        if num_chan!=96:
            print('error：通道数为')
            print(num_chan)
    data = h5py.File(folder_spike+filename_nosuffix+'.mat')
    cursor_pos = data['cursor_pos'][:]  # h5py读mat数据存在转置现象：详见：https://www.cnblogs.com/hechangchun/p/14305968.html
    pos_t = data['t'][:]# cursor_pos:(2,125000)    pos_t:(1,125000)注意他是二维的
    cursor_pos_filter=cursor_pos

    [lmp,lmppos] = getlmp(lfpdata,lfpt,cursor_pos_filter,pos_t)


    with open(result_folder + filename_nosuffix + '.pickle', 'wb') as f:
        pickle.dump([lmp,lmppos], f)

    print(day + ".pickle  处理完毕!")
