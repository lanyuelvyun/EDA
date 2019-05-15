# -*- coding:utf-8 -*-
"""
@author: lisiqi
@time: 2019/5/10 17:20
"""
# -*- coding:utf-8 -*-
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


class VariableDistribution(object):
    def __init__(self, var_name, value_list1, label_list1, value_list2, label_list2, var_threshold, bins, binning_mode, save_path):
        """
        @目的：针对2分类任务的变量分析
        1）分布图：比较2个集合中，同一变量的值分布情况，查看变量的分布是否一致
        2）odds图：比较2个集合中，同一变量对2类样本的区分能力，查看变量区分能力是否一致
        :param var_name: variable name
        :param value_list1: the value_list of the variable
        :param label_list1: sample label_list, one-to-one correspondence with value_list1. like[1,1,0,0,1,0,0,0,0]
        :param value_list2: the value_list of the variable
        :param label_list2: sample label_list, one-to-one correspondence with value_list2
        :param var_threshold: the number of unique values of this variable
        :param bins: number of bins
        :param binning_mode: ="qcut",equal frequency bin; ="cut",equal width bin
        :param save_path: the path where the distribution histgram is saved
        :return:
        """
        self.__var_name = var_name
        self.__value_list1 = value_list1
        self.__label_list1 = label_list1
        self.__value_list2 = value_list2
        self.__label_list2 = label_list2
        self.__var_threshold = var_threshold
        self.__bins = bins
        self.__binning_mode = binning_mode
        self.__save_path = save_path


    def calc_bad_good_rate(self):
        """
        计算好坏/好样本在总样本量中的比例
        """
        total_num1 = len(self.__label_list1)
        bad_num1 = sum(self.__label_list1)
        good_num1 = total_num1 - bad_num1
        bad_rate1 = bad_num1 * 1.0 / (total_num1 + 1e-20)
        good_rate1 = good_num1 * 1.0 / (total_num1 + 1e-20)
        print("bad_rate1 = %s, good_rate1 = %s" % (bad_rate1, good_rate1))
        # set 2
        total_num2 = len(self.__label_list2)
        bad_num2 = sum(self.__label_list2)
        good_num2 = total_num2 - bad_num2
        bad_rate2 = bad_num2 * 1.0 / (total_num2 + 1e-20)
        good_rate2 = good_num2 * 1.0 / (total_num2+ 1e-20)
        print("bad_rate2 = %s, good_rate2 = %s" % (bad_rate2, good_rate2))
        # return bad_rate, good_rate

    def get_var_distribution(self):
        """
        变量分布图：对该变量进行分箱，统计每一箱内的值个数占比
        odds图：对该变量进行分箱，计算每一个分箱内label=1的样本占该分箱内总样本的比例，查看变量对2类样本的区分能力
        """
        print("get var distribution and odds".center(80, '*'))
        print(("variable %s" % self.__var_name).center(80, '*'))

        # divide the variable with bins, and group it according to the bins
        df_all_bins = self.__divide_var_with_bins()
        df_all_groupby = df_all_bins.groupby("bins")

        # calculate distribution
        # x：each bin
        x_list = [str(x) for x in df_all_groupby.count().index]
        # y：distribution = (the number of values in each bin)/(total numbel of values in each bin)
        y_list1 = df_all_groupby.sum()["flag"] * 1.0 / len(self.__value_list1)
        y_list2 = (df_all_groupby.count()["flag"] * 1.0 - df_all_groupby.sum()["flag"] * 1.0) / len(self.__value_list2)
        # y：odds = (number of samples with label=1 in each bin)/(total number of samples in this bin)
        result = df_all_groupby.apply(lambda x: (x[(x["flag"] == 1) & (x["label"] == 1)].shape[0], x[x["flag"] == 1].shape[0]))
        y_list3 = result.map(lambda x: x[0]) * 1.0 / (result.map(lambda x: x[1]) + 1e-20)
        result = df_all_groupby.apply(lambda x: (x[(x["flag"] == 0) & (x["label"] == 1)].shape[0], x[x["flag"] == 0].shape[0]))
        y_list4 = result.map(lambda x: x[0]) * 1.0 / (result.map(lambda x: x[1]) + 1e-20)
        # plot
        self.__plot_distribution(x_list, y_list1, y_list2, y_list3, y_list4)

    def __divide_var_with_bins(self):
        # 为了使得两个集合具有相同的分箱区间，画图的时候方便：将两个集合的变量值放在一起，再进行分箱
        value_list_all = self.__value_list1 + self.__value_list2
        label_list_all = self.__label_list1 + self.__label_list2
        # add a flag to distinguish two list: 1 is value_list1, 0 is value_list2
        flag = list(np.ones(len(self.__value_list1))) + list(np.zeros(len(self.__value_list2)))
        df_var_all = pd.DataFrame({self.__var_name: value_list_all, "label": label_list_all, "flag": flag})

        # the number of unique values for this variable
        value_num = len(set(df_var_all[self.__var_name]))
        if value_num >= self.__var_threshold:  # 如果变量值的个数>=threshold个，当做连续变量来处理
            if self.__binning_mode == 'qcut':   # 等频分箱
                df_var_all["bins"] = pd.qcut(df_var_all[self.__var_name], self.__bins, duplicates="drop")
            elif self.__binning_mode == 'cut':  # 等宽分箱，左闭右开
                df_var_all["bins"] = pd.cut(df_var_all[self.__var_name], self.__bins, duplicates="drop", right=False, include_lowest=True)
        else:  # 如果变量值的个数<threshold个,当做离散变量来处理，每一个值当做一箱
            df_var_all["bins"] = df_var_all[self.__var_name]
        df_var_all = df_var_all.sort_values(by="bins")
        print("bins: ", df_var_all["bins"].unique())
        return df_var_all

    def __plot_distribution(self, x_list, y_list1, y_list2, y_list3, y_list4):
        plt.figure(figsize=(10, 8))
        plt.tick_params(labelsize=8)  # tick_params可设置坐标轴刻度值属性
        plt.bar(x_list, y_list1, label='set_1_distribution', width=0.5, bottom=0, facecolor='lightskyblue', alpha=0.5)
        plt.bar(x_list, y_list2, label='set_2_distribution', width=0.5, bottom=0, facecolor='orange', alpha=0.5)
        plt.plot(x_list, y_list3, label='set_1_odds', color="lightskyblue")
        plt.plot(x_list, y_list4, label='set_2_odds', color="orange")
        plt.xticks(x_list, color='black', rotation=60)  # 横坐标旋转60度
        plt.title('%s analysis' % self.__var_name)
        plt.xlabel("%s" % self.__var_name)
        plt.ylabel("distribution/odds")
        plt.legend()
        path = os.path.join(self.__save_path, self.__var_name+'.png')
        print("save_path is %s" % path)
        plt.savefig(path)
        plt.close()


if __name__ == '__main__':
    # 2 sets，which have same columns
    df1 = pd.read_csv('set1.csv')
    df1 = pd.read_csv('set2.csv')
    save_path = r"result"
 
    # feat_list
    var_list = df1.columns.tolist()

    # parameter
    var_threshold = 50  # 当unique(特征值)的个数<50的时候，该变量当做离散变量处理
    bins = 30  # 分成30箱
    binning_mode = "cut"  # 分箱模式："cut"等宽，"qcut"等频
    for var in var_list:
        # 去掉空值
        df_sub1 = df1[df1[var].notnull()]
        df_sub2 = df2[df2[var].notnull()]
        variabledistribution = VariableDistribution(var_name=var,value_list1=df_sub1[var].tolist(), label_list1=df_sub1["label"].tolist(),
                             value_list2=df_sub2[var].tolist(), label_list2=df_sub2["label"].tolist(),
                             var_threshold=var_threshold, bins=bins, binning_mode="cut", save_path=save_path)
        variabledistribution.get_var_distribution()
        variabledistribution.calc_bad_good_rate()

