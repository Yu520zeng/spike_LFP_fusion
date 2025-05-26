##  对单只spikes参数调优
import os
import pickle
from my_model_MLP import CNN_lstm_v2_lfp
import numpy as np

import xlrd
from xlutils.copy import copy

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

files_days = ['天数日期'] # files_days = ['indy_20170124_01']



for day in files_days:
    raw_filename = day+".mat"
    raw_filepath = "data/spike_1/"+raw_filename
    print(f"data into file: {raw_filepath}")


    """提取尖峰和运动学数据     HDF5数据格式存储。"""
    processed_dirname = os.path.join('data', 'processed')
    os.makedirs(processed_dirname, exist_ok=True)
    # specify filename to store the processed data
    processed_filename = raw_filename.split('.')[0]+'.h5'
    processed_filepath = os.path.join(processed_dirname, processed_filename)
    # process and store the raw data
    run = f"python process_data.py --input_filepath {raw_filepath} --output_filepath {processed_filepath}"
    # subprocess.run(run, shell=True, check=True)  # 使用subprocess模块执行shell命令并检查是否成功执行
    os.system(run)



    """制作用于训练和测试算法的数据集   HDF5数据格式存储。"""
    # directory to store the dataset
    dataset_dirname = os.path.join('data', 'dataset')
    # create folder if it doesn't exist
    os.makedirs(dataset_dirname, exist_ok=True)
    methods = ['baks']
    for method in methods:
        dataset_filename = f"{raw_filename.split('.')[0]}_{method}.h5"
        dataset_filepath = os.path.join(dataset_dirname, dataset_filename)
        print(processed_filepath)
        print(dataset_filepath)
        run = f"python make_dataset.py --input_filepath {processed_filepath} --output_filepath {dataset_filepath} --method {method}"
        # subprocess.run(run, shell=True, check=True)  # 使用subprocess模块执行shell命令并检查是否成功执行
        os.system(run)



    # directory to store the result
    result_dirname = 'params'
    # create folder if it doesn't exist
    os.makedirs(result_dirname, exist_ok=True)
    feature = 'mua'
    method = 'baks'
    decoders = ['qrnn']
    n_trials = 2  # Number of trials for optimisation
    timeout = 300  # Stop study after the given number of seconds
    n_startup_trials = 1  # Number of trials in the beginning for which pruning is disabled
    for decoder in decoders:
        result_filename = f"{raw_filename.split('.')[0]}_{feature}_{method}_{decoder}.pkl"
        result_filepath = os.path.join(result_dirname, result_filename)
        run = f"python opt_dl_decoder.py --input_filepath {dataset_filepath} --output_filepath {result_filepath} --decoder {decoder} --n_trials {n_trials} --timeout {timeout} --n_startup_trials {n_startup_trials}"
        os.system(run)





