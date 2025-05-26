import datetime
import os
import argparse
import json
import pickle
import numpy as np
from pandas import Categorical
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from my_model_MLP import CNN_lstm_v2_lfp  # 假设这是你的模型类


# 保留原始工具函数（数据处理相关）
def get_posvelacc_mat(bin_pos, dt): ...


def get_R2(y_test, y_test_pred): ...


def get_bin_lmp(lmp, lmppos): ...


def write_excel_xls_append(path, value): ...


def load_preprocess_data(file_path):
    """封装数据加载与预处理逻辑"""
    with open(file_path, 'rb') as f:
        lmp, lmppos = pickle.load(f, encoding='latin1')

    # 数据分箱处理
    multi_bin_neural_data, multi_bin_pos = get_bin_lmp(lmp, lmppos)
    pos_vel_acc_bin = get_posvelacc_mat(multi_bin_pos, 0.064)
    bin_vel = pos_vel_acc_bin[:, 2:4]  # 提取速度分量

    # 数据集划分
    num_examples = multi_bin_neural_data.shape[0]
    training_set = np.arange(int(0.8 * num_examples))  # 80%训练
    testing_set = np.arange(int(0.9 * num_examples) + 11, num_examples)  # 10%测试

    # 数据标准化
    x_train = multi_bin_neural_data[training_set]
    x_test = multi_bin_neural_data[testing_set]

    x_train_mean = np.nanmean(x_train, axis=0)
    x_train_std = np.nanstd(x_train, axis=0) + 1e-5  # 防止除零
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    # 输出零中心化
    y_train = bin_vel[training_set] - np.mean(bin_vel[training_set], axis=0)
    y_test = bin_vel[testing_set] - np.mean(bin_vel[training_set], axis=0)

    return (x_train, y_train), (x_test, y_test)


# 定义超参数搜索空间（根据模型调整）
param_space = [
        Integer(50, 200, step=25,  name='units'),
        Real(0.1, 0.5, step=0.1, name='dropout'),
        Real(1e-4, 0.1, name='learning_rate'),
        Integer(32, 128, step=0.1, name='batch_size'),
        Integer(1, 3, name='n_layers'),
        Integer(1, 10, name='timesteps'),
        Integer(1, 100, name='epochs'),
        Categorical(['Adam', 'RMSprop'], name='optimizer')
    ]


@use_named_args(param_space)
def objective(units, dropout, batch_size, num_epochs, learning_rate):
    """贝叶斯优化目标函数（返回RMSE）"""
    (x_train, y_train), (x_test, y_test) = load_preprocess_data(folder + begin_day)

    # 初始化并训练模型
    model = CNN_lstm_v2_lfp(
        units=units,
        dropout=dropout,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=0  # 训练过程静默
    )
    model.fit(x_train, y_train, x_val=x_train[:50], y_val=y_train[:50])  # 简单验证集

    # 评估模型
    y_pred = model.predict(x_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    return rmse  # 优化器将最小化RMSE


def main(args):
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_filepath), exist_ok=True)

    # 执行贝叶斯优化
    print(f"开始贝叶斯超参数搜索（目标文件: {args.output_filepath}）")
    result = gp_minimize(
        objective,
        param_space,
        n_calls=300,  # 总搜索次数（可调整）
        n_initial_points=8,  # 初始随机采样点数
        random_state=args.seed,
        acq_func="gp_hedge"  # 采集函数（平衡探索与利用）
    )

    # 保存最优参数到JSON
    best_params = {dim.name: result.x[i] for i, dim in enumerate(param_space)}
    with open(args.output_filepath, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"\n最优参数已保存至: {args.output_filepath}")
    print(f"最优RMSE: {result.fun:.4f}")
    print("最优参数详情:", best_params)


if __name__ == '__main__':
    # 全局参数配置
    folder = "data/LMP_1/"
    files = [f for f in os.listdir(folder) if f.endswith(".pickle")]
    training_range = [0, 0.8]
    valid_range = [0.8, 0.9]
    testing_range = [0.9, 1]

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='贝叶斯超参数优化')
    parser.add_argument('--day', type=str, default=files[0].replace(".pickle", ""),
                        help='实验日期（对应pickle文件名）')
    parser.add_argument('--output_filepath', type=str,
                        default="results/decoder/f_{day}_qrnn.json",
                        help='最优参数输出路径（支持{day}占位符）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    # 动态替换路径中的{day}占位符
    args.output_filepath = args.output_filepath.format(day=args.day)

    # 遍历文件执行优化（示例：仅处理第一个文件）
    for begin_day in files[:1]:
        print(f"\n===== 处理文件: {begin_day} =====")
        main(args)