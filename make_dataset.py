
# import packages
import argparse
import numpy as np
import h5py
from bmi.features import extract
import time as timer
from scipy.io import savemat

def main(args):
    run_start = timer.time()
    
    print(f"Loading spike and kinematic data from file: {args.input_filepath}")
    with h5py.File(args.input_filepath, 'r') as f:
        task_data = f['task_data'][()]  # kinematic data
        task_time = f['task_time'][()]  # time associated with the kinematic data
        sua_train = f['sua_trains'][()] # sorted spike times (single unit activity)
        mua_train = f['mua_trains'][()] # unsorted spike times (threshold crossing/multi unit activity)

    print(task_data.shape, task_time.shape, mua_train.shape, sua_train.shape)

    num_sua = len(sua_train)
    num_mua = len(mua_train)

    delta_time = 0.004 # sampling interval in second
    nperseg = int(args.wdw_time / delta_time) + 1
    noverlap = int(args.ol_time / delta_time) + 1



    X_sua = []
    X_mua = []
    for i in range(num_sua):
        #print(f"Extracting SUA/sorted spike features from unit no: {i}")
        if args.method == 'binning':
            sua_rate, y_task = extract(sua_train[i], task_time, nperseg, noverlap, task=task_data, method=args.method)
        elif args.method == 'gaussian':
            std_time = args.std_time
            std = int(std_time/delta_time)
            sua_rate, y_task = extract(sua_train[i], task_time, nperseg, noverlap, task=task_data, method=args.method, window=args.method, std=std)
        elif args.method == 'baks':
            sua_rate, y_task = extract(sua_train[i], task_time, nperseg, noverlap, task=task_data, method=args.method, a=args.alpha)
        X_sua.append(sua_rate)

    run_times = []
    for i in range(num_mua):
        #print(f"Extracting MUA/threshold crossing features from channel no: {i}")
        extract_start = timer.time()
        if args.method == 'binning':
            mua_rate = extract(mua_train[i], task_time, nperseg, noverlap, task=None, method=args.method)
        elif args.method == 'gaussian':
            std_time = args.std_time
            std = int(std_time/delta_time)
            mua_rate = extract(mua_train[i], task_time, nperseg, noverlap, task=None, method=args.method, window=args.method, std=std)
        elif args.method == 'baks':
            mua_rate = extract(mua_train[i], task_time, nperseg, noverlap, task=None, method=args.method, a=args.alpha)
        extract_end = timer.time()
        run_time = (extract_end - extract_start) / mua_rate.shape[0]
        run_times.append(run_time)
        X_mua.append(mua_rate)
    run_times = np.asarray(run_times)
    print(f"Average (std) run time for spike rate estimation: {run_times.mean()*1e6} ({run_times.std()*1e6}) µs")
    # convert to array
    X_sua = np.asarray(X_sua).T
    X_mua = np.asarray(X_mua).T

    print(f"Storing dataset into file : {args.output_filepath}")
    with h5py.File(args.output_filepath, 'w') as f:
        f['X_sua'] = X_sua
        f['X_mua'] = X_mua
        f['y_task'] = y_task

    # # 存.mat文件
    matlab_vars = {}
    matlab_vars[f'task_time'] = task_time
    matlab_vars[f'task_data'] = X_sua
    matlab_vars[f'target_pos'] = y_task
    savemat('data/spike/spike2_indy_20160630_01.mat', matlab_vars)


    run_end = timer.time()
    print(f"Finished whole processes within {(run_end-run_start)/60:.2f} minutes")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath',   type=str,   default="data\processed\indy_20160630_01.h5", help='Path to the spike dan kinematic data')

    parser.add_argument('--output_filepath',  type=str,   default="data\spike\indy_20160630_01_baks.h5", help='Path to the created dataset')
    parser.add_argument('--method',           type=str,   default='gaussian',  help='Spike rate estimation method')
    parser.add_argument('--wdw_time',         type=float, default=0.240,      help='Segment window size (s)')
    parser.add_argument('--ol_time',          type=float, default=0.120,      help='Overlap window size (s)')
    parser.add_argument('--std_time',         type=float, default=0.060,      help='Bandwidth of Gaussian window (s)')
    parser.add_argument('--alpha',            type=float, default=4.,         help='Shape parameter of BAKS')
    
    args = parser.parse_args()
    main(args)