# -*- coding:utf-8 -*-
"""
@author: lisiqi
@time: 2019/01/19 18:01
@目的：针对二分类任务的特征分析  
1）概率密度图：查看特征的分布情况
2）odds图：查看特征对2类样本的区分能力
3）比较2个集合的特征异同情况
"""
import pandas as pd
import os
import numpy as np
import codecs
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.dpi'] = 150
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


# 得到特征的code和describe。
def get_features_list(var_conf_path):
    list_name = []
    list_desc = []
    with codecs.open(var_conf_path, 'r', encoding="utf-8") as fr:
        for l in fr.readlines():
            if len(l) > 4 and l[0] != '#':
                ll = l.strip().split('\t')
                list_name.append(ll[0])
                list_desc.append(ll[2])
    return list_name, list_desc


# 画图
def plot_distribution(var_name, x, y_train, y_test, save_path):
    plt.figure(figsize=(15, 10))
    plt.tick_params(labelsize=8)  # tick_params可设置坐标轴刻度值属性
    plt.plot(x, y_train, label='train', color="lightskyblue")
    plt.bar(x, y_train, width=0.5, bottom=0, facecolor='lightskyblue', alpha=0.5)
    plt.bar(x, y_test, width=0.5, bottom=0, facecolor='orange', alpha=0.5)
    plt.plot(x, y_test, label='test', color="orange")
    plt.title('Distribution of %s' % var_name)
    plt.xlabel("%s" % var_name)
    plt.ylabel("density")
    plt.legend()
    path = os.path.join(save_path, var_name + '_density.png')
    print("保存图片，路径为%s" % path)
    plt.savefig(path)
    plt.close()


# 计算每一个特征的缺失率
def get_var_miss_rate(df_train, df_test):
    var_list = df_train.columns.to_list()  # 两个集合的特征要一致
    df_var_miss = pd.DataFrame({"var_name": var_list})
    df_var_miss["var_train_miss_rate"] = df_train.apply(lambda x: x.isnull().sum() * 1.0 / x.shape[0], axis=1)
    df_var_miss["var_test_miss_rate"] = df_test.apply(lambda x: x.isnull().sum() * 1.0 / x.shape[0], axis=1)
    print('get_var_miss_rate is done.')


# 对每一个特征进行等宽分箱，返回带有分箱信息的df
def get_var_bins(threshold, var_name, df_train, df_test, bins):
    # 为了使得两个集合具有相同的分箱区间，画图的时候方便：将两个集合的特征值放在一起，再进行分箱
    df_train_copy = df_train.copy()
    df_train_copy["is_train"] = 1  # 添加一个标志位，用来区分train集和test集
    df_test_copy = df_test.copy()
    df_test_copy["is_train"] = 0
    df_all = pd.concat([df_train_copy, df_test_copy])

    value_num = len(set(df_all[var_name]))
    if value_num >= threshold:  # 如果特征值的个数>=threshold个，当做连续变量来处理，分成bins箱
        # 等宽分箱
        df_all["bins"] = pd.cut(df_all[var_name], bins)
    else:  # 如果特征值的个数<threshold个,当做离散特征来处理，每一个值当做一箱
        # 按照该特征的每一个离散值进行分箱
        df_all["bins"] = df_all[var_name]
    return df_all


# 画特征概率密度图
def get_var_density(threshold, df_train, df_test, bins, save_path):
    """
    特征概率密度图：对每一个特征进行等宽分箱，统计每一箱内的人数占比
    目的：比较2个集合中每一个特征的分布情况，查看特征的分布是否一致
    """
    var_list = list(df_train.columns)
    for i in var_list:
        print(("特征 %s" % i).center(80, '*'))
		
	# 每一个特征，去掉空值的情况
        df_train = df_train[df_train[i].notnull()]
        df_test = df_test[df_test[i].notnull()]
		
        # 对每一个特征进行分箱，并根据分箱区间进行分组
        df_all_bins = get_var_bins(threshold, i, df_train, df_test, bins)
        df_all_groupby = df_all_bins.groupby("bins")
		
        # x轴：每一箱
        x_list = [str(x) for x in df_all_groupby.count().index]
        # y轴：统计train集和test集，该特征在每箱内的人数占比
        y_list_train = df_all_groupby.sum()["is_train"] * 1.0 / df_train[i].shape[0]
        y_list_test = (df_all_groupby.count()["is_train"] * 1.0 - df_all_groupby.sum()["is_train"] * 1.0) \
                       / df_test[i].shape[0]

        # 画图
        plot_distribution(i, x_list, y_list_train, y_list_test, save_path)


# 画特征odds图
def get_var_odds(threshold, df_train, df_test, bins, save_path):
    """
    odds图：对每一个特征进行等宽分箱，统计每一箱内的odds：label=1的样本个数/该分箱内总样本数
    能看出：1）该特征对两类样本是否有区分能力；2）特征对样本的区分能力在两个集合上是否一样
    """
    var_list = list(df_train.columns)
    for i in var_list:
        print(("特征 %s" % i).center(80, '*'))
		
	# 每一个特征，去掉空值的情况
        df_train = df_train[df_train[i].notnull()]
        df_test = df_test[df_test[i].notnull()]
		
        # 对每一个特征进行分箱，返回带有分箱信息的df
        df_all_bins = get_var_bins(threshold, i, df_train, df_test, bins)
		
	# 根据bins进行分组,统计每一个bin中样本的总个数
        # 先将两个集合拆出来，各自统计.此时index就是bins
        df_train_count = df_all_bins[df_all_bins["is_train"] == 1].groupby("bins").count()
        df_test_count = df_all_bins[df_all_bins["is_train"] == 0].groupby("bins").count()
        # 再将两个df以index为关联键merge在一起。先拆分后又组合，这样做是为了让两个集合具有相同的bins，方便画图
        df_all_count = pd.merge(df_train_count, df_test_count, right_index=True, left_index=True, how="outer")
        df_all_count.fillna(0, inplace=True)
		
	# 根据bins进行分组,统计每一个bin中label=1样本的个数
        # 先将两个集合拆出来，各自统计.此时index就是bins
        df_train_sum = df_all_bins[df_all_bins["is_train"] == 1].groupby("bins").sum()
        df_test_sum = df_all_bins[df_all_bins["is_train"] == 0].groupby("bins").sum()
        # 再将两个df以index为关联键merge在一起
        df_all_sum = pd.merge(df_train_sum, df_test_sum, right_index=True, left_index=True, how="outer")
        df_all_sum.fillna(0, inplace=True)

        # x轴：每一箱
        x_list = [str(x) for x in df_all_count.index]
        # y轴：统计train集和test集每箱内odds：label=1的样本个数/该分箱内总样本数
        y_odds_train = df_all_sum["label_x"] * 1.0 / df_all_count["label_x"]
        y_odds_test = df_all_sum["label_y"] * 1.0 / df_all_count["label_y"] 

        # 画图
        plot_distribution(i, x_list, y_odds_train, y_odds_test, save_path)


if __name__ == "__main__":
    df_all = pd.read_csv(r'E:\work\2 mission\acard\acard_dae\source\final_main_0814_1007_1.csv', na_values=[-3, -2, -1])
    # 图片保存地址
    density_save_path = r'E:\work\2 mission\transform_model\transform_model_zx_dae\result\plot_var_density'
    odds_save_path = r'E:\work\2 mission\transform_model\transform_model_zx_dae\result\plot_var_odds'
    # conf文件是一个特征的配置文件，有3列，分别为：特征code，特征英文名，特征中文描述
    var_conf_path = r"E:\work\2 mission\acard\acard_dae\f_conf\test.conf"
	
    df_all["label"] = df_all.apply(lambda x: 1 if x["overdue_day"] > 7 else 0, axis=1)
    var_list, _ = get_features_list(var_conf_path)
    df_all = df_all[var_list + ["user_id","loan_id", "label", "overdue_day", "created"]]
    # 按照created时间，分成2个集合
    df_train = df_all[df_all["created"] < 20180920]
    df_test = df_all[df_all["created"] >= 20180920]

    bins = 20
    threshold = 30
	
    # 画概率密度图：比较2个集合中每一个特征的分布情况，查看特征的分布是否一样
    # get_var_density(threshold, df_train[var_list], df_test[var_list], bins, density_save_path)
	
    # 画odds图：查看在2个集合上特征的区分能力是否一样
    get_var_odds(threshold, df_train, df_test, bins, odds_save_path)
