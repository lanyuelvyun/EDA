# -*- coding:utf-8 -*-
"""
@author: lisiqi
@time: 2019/03/01 18:01
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


class VariableDistribution(object):
    def __init__(self, var_name, value_list1, label_list1, value_list2, label_list2, var_threshold, bins, binning_mode, save_path):
        """
        @目的：针对2分类任务的变量分析
        1）分布图：比较2个集合中，同一变量的值分布情况，查看变量的分布是否一致
        2）odds图：比较2个集合中，同一变量对2类样本的区分能力，查看变量区分能力是否一致
        :param var_name: variable name
        :param value_list1: the value_list of the variable
        :param label_list1: sample label_list, one-to-one correspondence with value_list1
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
        # self.get_var_distribution()
        self.get_var_odds()

    def get_var_distribution(self):
        """
        变量分布图：对该变量进行分箱，统计每一箱内的值个数占比
        """
        print("get var distribution".center(80, '*'))
        print(("variable %s" % self.__var_name).center(80, '*'))
        # remove null value
        # self.__value_list1 = [x for x in self.__value_list1 if x >= 0]
        # self.__value_list2 = [x for x in self.__value_list2 if x >= 0]

        # divide the variable with bins, and group it according to the bins
        df_all_bins = self.__divide_var_with_bins()
        print("bins: ", df_all_bins["bins"].unique())
        df_all_groupby = df_all_bins.groupby("bins")

        # x：each bin
        x_list1 = [str(x) for x in df_all_groupby.count().index]
        x_list2 = [str(x) for x in df_all_groupby.count().index]
        # y：distribution = the number of values in each bin/total numbel of values
        y_list1 = df_all_groupby.sum()["flag"] * 1.0 / len(self.__value_list1)
        y_list2 = (df_all_groupby.count()["flag"] * 1.0 - df_all_groupby.sum()["flag"] * 1.0) \
                      / len(self.__value_list2)

        # plot distrbution histgram
        self.__plot_distribution("distribution", x_list1, x_list2, y_list1, y_list2)

    def get_var_odds(self):
        """
        odds分布图：对该变量进行分箱，统计每一箱内label=1的样本个数与该箱内总样本个数之比
        """
        print("get var odds".center(80, '*'))
        print(("variable %s" % self.__var_name).center(80, '*'))
        # remove null value
        # self.__value_list1 = [x for x in self.__value_list1 if x >= 0]
        # self.__value_list2 = [x for x in self.__value_list2 if x >= 0]

        # divide the variable with bins
        df_all_bins = self.__divide_var_with_bins()
        print("bins: ", df_all_bins["bins"].unique())
        # separate the value_list1 and value_list2 into two df
        df_sub1_bins = df_all_bins.loc[df_all_bins["flag"] == 1]
        df_sub2_bins = df_all_bins.loc[df_all_bins["flag"] == 0]

        # x：each bin
        x_list1 = [str(x) for x in df_sub1_bins.groupby("bins").count().index]
        x_list2 = [str(x) for x in df_sub2_bins.groupby("bins").count().index]
        # y：odds = number of samples with label=1 in each bin/total number of samples in this bin
        y_list1 = df_sub1_bins.groupby("bins").sum()["label"]*1.0 / df_sub1_bins.groupby("bins").count()["label"]
        y_list2 = df_sub2_bins.groupby("bins").sum()["label"]*1.0 / df_sub2_bins.groupby("bins").count()["label"]

        # plot odds histgram
        self.__plot_distribution("odds", x_list1, x_list2, y_list1, y_list2)

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
            elif self.__binning_mode == 'cut':  # 等宽分箱
                df_var_all["bins"] = pd.cut(df_var_all[self.__var_name], self.__bins, duplicates="drop")
        else:  # 如果变量值的个数<threshold个,当做离散变量来处理，每一个值当做一箱
            df_var_all["bins"] = df_var_all[self.__var_name]
        df_var_all = df_var_all.sort_values(by="bins")
        # print(df_var_all["bins"].dtype)
        return df_var_all

    def __plot_distribution(self, png_name, x_list1, x_list2, y_list1, y_list2):
        plt.figure(figsize=(15, 10))
        plt.tick_params(labelsize=8)  # tick_params可设置坐标轴刻度值属性
        plt.plot(x_list1, y_list1, label='set_1', color="lightskyblue")
        plt.bar(x_list1, y_list1, width=0.5, bottom=0, facecolor='lightskyblue', alpha=0.5)
        plt.plot(x_list2, y_list2, label='set_2', color="orange")
        plt.bar(x_list2, y_list2, width=0.5, bottom=0, facecolor='orange', alpha=0.5)
        plt.title('%s of %s' % (png_name,self.__var_name))
        plt.xlabel("%s" % self.__var_name)
        plt.ylabel(png_name)
        plt.legend()
        path = os.path.join(self.__save_path, self.__var_name+"_"+png_name+'.png')
        print("save_path is %s" % path)
        plt.savefig(path)
        plt.close()
