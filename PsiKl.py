# -*- coding:utf-8 -*-
"""
@author: lisiqi
@time: 2019/1/7 17:07
@目的：查看特征的分布变化
所用指标：PSI和KL散度
K-L散度，是一种量化两种概率分布P和Q之间差异的方式，又叫相对熵.
"""
import pandas as pd
from sklearn import preprocessing
import math
import json
import requests
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.wivalueh', 1000)
# pd.set_option('display.max_columns', 1000)
# pd.set_option('expand_frame_repr', False)


# 针对建模样本，统计每个特征的最大最小值、分箱情况
def offline_stat(df_train, var_list):
    # print(var_list)
    # 统计每个特征的最大最小值
    df_stat_minmax_train = pd.DataFrame(columns=['var_name', 'var_min_train', 'var_max_train'])
    df_stat_minmax_train['var_name'] = var_list
    # 统计每一个特征每一箱的分箱占比
    df_stat_binning_train = pd.DataFrame(columns=['var_name', 'var_binning'])
    df_stat_binning_train['var_name'] = var_list
    for var in var_list:
        seris_var = df_train[var][df_train[var].values != -3]  # 去掉-3的情况,没调用
        print("train set: ", var, seris_var.shape)
        seris_var = seris_var.map(lambda x: np.nan if (x == -1 or x == -2) else x)  # 将-1，-2置为空值
        # 将缺失值去掉,再进行归一化
        # seris_var = seris_var.fillna(seris_var.mean())
        seris_var_sub = seris_var.loc[seris_var.notnull()]
        print("train var without nan :", seris_var_sub.shape)
        # 对特征值进行归一化，方便分箱,并记录最大最小值
        min_max_scaler = preprocessing.MinMaxScaler()
        df_train_trans = min_max_scaler.fit_transform(seris_var_sub.values.reshape(-1, 1))
        # np.savetxt(r"E:\work\2 mission\1 acard\acard_dae\result\df_train_trans_" + var+ ".csv", df_train_trans)
        df_stat_minmax_train.loc[df_stat_minmax_train["var_name"] == var, 'var_min_train'] = min_max_scaler.data_min_[0]
        df_stat_minmax_train.loc[df_stat_minmax_train["var_name"] == var, 'var_max_train'] = min_max_scaler.data_max_[0]

        # 每个特征均分11箱，计算每箱样本占比，其中最后一箱放特征值为空的样本
        count_train = len(seris_var)  # 该特征总样本数，包含特征值为空的样本
        var_binning_dict = dict(list(zip(range(11), [0] * 11)))
        for value in df_train_trans:
            key = int(value * 10)
            key = 9 if key > 9 else key
            var_binning_dict[key] += 1
        for key, value in var_binning_dict.items():
            var_binning_dict[key] = round(var_binning_dict[key] * 1.0 / count_train, 6)
        # 特征值为空的样本，放在第11箱
        var_binning_dict[10] = round(seris_var.isnull().sum() * 1.0 / count_train, 6)
        df_stat_binning_train.loc[df_stat_binning_train['var_name'] == var, 'var_binning'] = json.dumps(var_binning_dict)  # 以json的形式保存分箱信息

    # 保存每一个特征的最大最小值
    df_stat_minmax_train.to_csv("result/df_stat_minmax_train.csv", index=None, float_format="%.6f")
    # 保存每一个特征的分箱
    df_stat_binning_train.to_csv("result/df_stat_binning_train.csv", index=None, float_format="%.6f")

    # 统计结果转化为JSON
    df_stat = pd.merge(df_stat_minmax_train, df_stat_binning_train, on='var_name', how='left')
    df_stat = pd.merge(df_stat, df_conf, on='var_name', how='left')
    df_stat.to_csv(r"result\df_stat.csv", index=None)
    df_stat['name'] = 'pass'
    df_stat['is_null'] = 'true'
    df_stat['status'] = 1
    df_stat['data_type'] = 'float'
    df_stat['var_weight'] = df_stat['var_weight'].astype(int)
    df_stat['var_id'] = df_stat['var_name']
    df_stat['regex'] = df_stat.apply(lambda x: "["+str(x['var_min_train'])+","+str(x['var_max_train'])+"]", axis=1)
    df_stat['value_distri'] = df_stat['var_binning'].apply(lambda x: json.loads(x))
    columns = ['var_id', 'name', 'type', 'data_type', 'desc', 'status', 'var_weight', 'is_null', 'regex', 'value_distri']
    # 将上面所有的信息都放在一个json串里面
    df_stat['json'] = df_stat[columns].apply(lambda x: json.loads(x.to_json()), axis=1)

    # model_name和线上跑的model名字对应
    # type可以填"primary" "sub_tongdun" "sub_jiguang" "sub_geo"等
    # status为1表示当前模型正在线上调用
    res = {"model_var_info": [{"model_info":
                                       {"model_name": "trans_ir_zx_dae_v1",
                                        "type": "primary",
                                        "version": 1,
                                        "desc": "征信大额贷前利率转化率模型主要特征监控",
                                        "status": 1,
                                        "var_info": list(df_stat['json'])}}]}

    # print(json.dumps(res))
    with open("result/df_stat_json.txt", 'w') as json_file:
        json.dump(res, json_file, encoding='utf-8')

# 针对test集样本，按照train集的分箱方式，统计分箱情况
def online_stat(df_stat_minmax_train, df_stat_binning_train, df_test, var_list):
    # 统计test集中，每一个特征每一箱的个数占比
    df_stat_binning_test = pd.DataFrame(columns=['var_name', 'var_binning'])
    df_stat_binning_test['var_name'] = var_list
    for var in var_list:
        # 加载训练集中，该特征的最大最小值
        var_min_train = float(df_stat_minmax_train[df_stat_minmax_train['var_name'] == var]['var_min_train'].values[0])
        var_max_train = float(df_stat_minmax_train[df_stat_minmax_train['var_name'] == var]['var_max_train'].values[0])
        # 处理test集数据
        seris_var = df_test[var][df_test[var].values != -3]  # 去掉-3的情况,没调用
        print("test set: ", var, seris_var.shape)
        seris_var = seris_var.map(lambda x: np.nan if (x == -1 or x == -2) else x)  # 将-1，-2置为空值
        # 将缺失值去掉,再进行归一化
        seris_var_sub = seris_var.loc[seris_var.notnull()]

        # 按照训练集的分箱方式进行分箱：同分箱数、同分箱区间,其中最后一箱放特征值为空的样本
        count_test = len(seris_var)  # 该特征总样本数，包含特征值为空的样本
        bins_train = len(eval(df_stat_binning_train["var_binning"][1]).keys())
        var_binning_dict = dict(list(zip(range(bins_train), [0] * bins_train)))
        for value in seris_var_sub:
            key = int(10.0 * (value - var_min_train) / (var_max_train - var_min_train))
            if key < 0:
                key = 0
            elif key > 9:
                key = 9
            var_binning_dict[key] += 1
        for key, value in var_binning_dict.items():
            var_binning_dict[key] = round(var_binning_dict[key] * 1.0 / count_test, 6)
        # 特征值为空的样本，放在第11箱
        var_binning_dict[10] = round(seris_var.isnull().sum() * 1.0 / count_test, 6)
        df_stat_binning_test.loc[df_stat_binning_test["var_name"] == var, 'var_binning'] = json.dumps(var_binning_dict) # 以json的形式保存分箱信息
        df_stat_binning_test.to_csv("result/df_stat_binning_test.csv", index=None, float_format="%.6f")


def get_psi_kl(df_stat_binning_train, df_stat_binning_test, var_list):
    df_var_psi = pd.DataFrame(columns={"var_name", "psi", "kl"})
    df_var_psi["var_name"] = var_list
    for var in var_list:
        # 加载train集和test集的分箱信息
        stat_binning_train = eval(df_stat_binning_train.loc[df_stat_binning_train['var_name'] == var, 'var_binning'].values[0])
        stat_binning_train = [stat_binning_train[k] for k in sorted(stat_binning_train.keys())]
        stat_binning_test = eval(df_stat_binning_test.loc[df_stat_binning_test['var_name'] == var, 'var_binning'].values[0])
        stat_binning_test = [stat_binning_test[k] for k in sorted(stat_binning_test.keys())]

        # 计算psi
        psi = 0
        for rate in list(zip(stat_binning_train, stat_binning_test)):
            psi += (rate[0] - rate[1]) * math.log((rate[0]+0.000001) / (rate[1]+0.000001))
        df_var_psi.loc[df_var_psi["var_name"] == var, "psi"] = psi
        # 计算kl散度
        kl = 0
        for i in range(len(stat_binning_train)):
            kl += stat_binning_test[i] * math.log((stat_binning_test[i]/(stat_binning_train[i]+0.000001))+0.000001)
        df_var_psi.loc[df_var_psi["var_name"] == var, "kl"] = kl

    df_var_psi = df_var_psi[["var_name", "psi", "kl"]]
    df_var_psi.sort_values(by="psi").to_csv(r"result/var_psi_kl.csv", index=None)

if __name__ == '__main__':
    import numpy as np
    df_conf = pd.read_csv("f_conf/acard_zx_dae_v1_feat_monitor.conf", dtype=str)
    var_list = df_conf['var_name'].tolist()
    # var_list = ['01004003','01001003']
    df_train = pd.read_csv(r".\source\20181212\final_main_0814_1007_1.csv")[var_list]
    print(df_train.shape)
    print("对train集进行分箱，并统计每箱内的人数占比，并记录下分箱区间。")
    offline_stat(df_train, var_list)

    # example
    df_stat_minmax_train = pd.read_csv(r"result/df_stat_minmax_train.csv", dtype=str)
    df_stat_binning_train = pd.read_csv(r"result/df_stat_binning_train.csv", dtype=str)
    df_test = df_train.sample(n=1000)
    print("针对test集样本，按照train集的分箱方式，统计分箱情况")
    online_stat(df_stat_minmax_train, df_stat_binning_train, df_test, var_list)
    df_stat_binning_test = pd.read_csv(r"result/df_stat_binning_test.csv", dtype=str)

    # 计算psi和kl
    get_psi_kl(df_stat_binning_train, df_stat_binning_test, var_list)
