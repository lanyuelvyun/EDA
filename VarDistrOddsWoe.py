# -*- coding:utf-8 -*-
"""
@author: lisiqi
@time: 2019/5/10 17:20
"""
# -*- coding:utf-8 -*-
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决坐标轴负数的负号显示问题
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.dpi'] = 150
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


class VarDistrOddsWoe(object):
    def __init__(self, var_name, value_list1, label_list1, value_list2, label_list2, var_threshold, bins, divide_mode, save_path):
        """
        @目的：针对2分类任务的变量分析
        1）分布图：比较2个集合中，同一变量的值分布情况，查看变量的分布是否一致
        2）odds图：比较2个集合中，同一变量对2类样本的区分能力，查看变量区分能力是否一致
        3) woe图：
        :param var_name: variable name
        :param value_list1: the value_list of the variable
        :param label_list1: sample label_list, one-to-one correspondence with value_list1. like[1,1,0,0,1,0,0,0,0]
        :param value_list2: the value_list of the variable
        :param label_list2: sample label_list, one-to-one correspondence with value_list2
        :param var_threshold: the number of unique values of this variable
        :param bins: number of bins
        :param divide_mode: ="qcut",equal frequency bin; ="cut",equal width bin
        :param save_path: the path where the distribution histgram is saved
        :return:
        """
        self.__var_name = var_name
        self.__value_list1 = value_list1
        self.__label_list1 = label_list1
        self.__value_list2 = value_list2
        self.__label_list2 = label_list2
        self.__cnt_threshold = var_threshold
        self.__bins = bins
        self.__divide_mode = divide_mode
        self.__save_path = save_path
        self.__get_result()

    def __get_result(self):
        df_distribution = self.get_var_distribution()
        df_odds = self.get_var_odds()
        df_woe_iv = self.get_var_woe_iv()
        df_result = df_distribution.merge(df_odds, left_index=True, right_index=True, how="outer")
        df_result = df_result.merge(df_woe_iv, left_index=True, right_index=True, how="outer")
        df_result.to_csv(self.__save_path + '/%s_distribution_odds_woe.csv' % self.__var_name)
        self.__plot(df_result)

    def get_var_distribution(self):
        """
        变量频数分布图：对该变量进行分箱，统计每一箱内的频数占比
        """
        print("get var distribution".center(80, '*'))
        print(("variable %s" % self.__var_name).center(20, '-'))
        # divide the variable, and group it according to the bins
        df_with_bins = self.__divide_var()
        df_groupby = df_with_bins.groupby("bins")
        result1 = df_groupby.sum()["flag"] * 1.0 / len(self.__value_list1)
        result2 = (df_groupby.count()["flag"] * 1.0 - df_groupby.sum()["flag"] * 1.0) / len(self.__value_list2)
        df_distribution = pd.concat([result1, result2], axis=1, keys=["distribution_set1", "distribution_set2"])
        return df_distribution

    def get_var_odds(self):
        """
        变量odds图：对该变量进行分箱，统计每一箱内的label=1的样本个数/label=0的样本个数
        """
        print("get var odds".center(80, '*'))
        print(("variable %s" % self.__var_name).center(80, '-'))
        df_with_bins = self.__divide_var()
        df_groupby = df_with_bins.groupby("bins")
        # set1
        result = df_groupby.apply(lambda x: (x[(x["flag"] == 1) & (x["label"] == 1)].shape[0], x[(x["flag"] == 1) & (x["label"] == 0)].shape[0]))
        df_odds1 = result.map(lambda x: x[0]) * 1.0 / (result.map(lambda x: x[1]) + 1e-20)  # 该特征在set1中的分布
        # set2
        result = df_groupby.apply(lambda x: (x[(x["flag"] == 0) & (x["label"] == 1)].shape[0], x[(x["flag"] == 0) & (x["label"] == 0)].shape[0]))
        df_odds2 = result.map(lambda x: x[0]) * 1.0 / (result.map(lambda x: x[1]) + 1e-20)  # 该特征在set2中的分布
        df_odds = pd.concat([df_odds1, df_odds2], axis=1, keys=["odds_set1", "odds_set2"])  # pd.concat沿着axis=1对series进行合并时，keys会成为合并之后df的列名
        return df_odds

    # 计算woe\IV值
    def get_var_woe_iv(self):
        print("get var woe_iv".center(80, '*'))
        print(("variable %s" % self.__var_name).center(80, '-'))
        df_with_bins = self.__divide_var()
        df_groupby = df_with_bins.groupby("bins")
        # set1
        total_bad_cnt = df_with_bins[df_with_bins["flag"] == 1]["label"].sum()
        total_good_cnt = df_with_bins[df_with_bins["flag"] == 1].shape[0] - total_bad_cnt
        df_woe_iv = df_groupby.apply(lambda x: self._cal_woe_tmp(x[x["flag"] == 1], total_bad_cnt, total_good_cnt)).reset_index(level=1, drop=True)
        df_woe_iv["woe"] = np.log(df_woe_iv["bad_rate"] * 1.0 / (df_woe_iv["good_rate"] + 10e-20))
        df_woe_iv["iv"] = (df_woe_iv["bad_rate"] - df_woe_iv["good_rate"]) * df_woe_iv["woe"]
        df_woe_iv.columns = [x + '_set1' for x in df_woe_iv.columns]
        # set2
        total_bad_cnt = df_with_bins[df_with_bins["flag"] == 0]["label"].sum()
        total_good_cnt = df_with_bins[df_with_bins["flag"] == 0].shape[0] - total_bad_cnt
        result = df_groupby.apply(lambda x: self._cal_woe_tmp(x[x["flag"] == 0], total_bad_cnt, total_good_cnt)).reset_index(level=1, drop=True)
        result["woe"] = np.log(result["bad_rate"] * 1.0 / (result["good_rate"] + 10e-20))
        result["iv"] = (result["bad_rate"] - result["good_rate"]) * result["woe"]
        result.columns = [x + '_set2' for x in result.columns]
        df_woe_iv = df_woe_iv.merge(result, left_index=True, right_index=True, how="outer")
        return df_woe_iv

    @staticmethod
    def _cal_woe_tmp(df, total_bad_cnt, total_good_cnt):
        # 计算IV值的时候，分组计算，存储中间值
        return pd.DataFrame.from_dict(
            {
                "bad_cnt": df["label"].sum(),
                "total_bad_cnt": total_bad_cnt,
                "bad_rate": df["label"].sum() * 1.0 / total_bad_cnt,  # 该分组内坏样本/总坏样本
                "good_cnt": df[df["label"] == 0].shape[0],
                "total_good_cnt": total_good_cnt,
                "good_rate": df[df["label"] == 0].shape[0] * 1.0 / total_good_cnt  # 该分组内好样本/总好样本
            }, orient='index').T

    def __divide_var(self):
        # 为了使得两个集合具有相同的分箱区间，画图的时候方便：将两个集合的变量值放在一起，再进行分箱
        value_list = self.__value_list1 + self.__value_list2
        label_list = self.__label_list1 + self.__label_list2
        # add a flag to distinguish two list: 1 is value_list1, 0 is value_list2
        flag = list(np.ones(len(self.__value_list1))) + list(np.zeros(len(self.__value_list2)))
        df_var = pd.DataFrame({self.__var_name: value_list, "label": label_list, "flag": flag})

        # the number of unique values for this variable
        value_cnt = len(set(df_var[self.__var_name]))
        if value_cnt >= self.__cnt_threshold:  # 如果变量值的个数>=threshold个，当做连续变量来处理
            if self.__divide_mode == 'qcut':   # 等频分箱
                df_var["bins"] = pd.qcut(df_var[self.__var_name], self.__bins, duplicates="drop")
            elif self.__divide_mode == 'cut':  # 等宽分箱，左闭右开
                df_var["bins"] = pd.cut(df_var[self.__var_name], self.__bins, duplicates="drop", right=False, include_lowest=True)
        else:  # 如果变量值的个数<threshold个,当做离散变量来处理，每一个值当做一箱
            df_var["bins"] = df_var[self.__var_name]
        df_var = df_var.sort_values(by="bins")
        print("bins: ", df_var["bins"].unique())
        return df_var

    def __plot(self, df_result):
        plt.figure(figsize=(20, 8))
        plt.tick_params(labelsize=8)  # tick_params可设置坐标轴刻度值属性
        X = [str(x) for x in df_result.index]
        # feat distribution
        plt.subplot(131)
        plt.bar(X, df_result["distribution_set1"], label='distribution_set1', width=0.5, bottom=0, facecolor='blue', alpha=0.5)
        plt.bar(X, df_result["distribution_set2"], label='distribution_set2', width=0.5, bottom=0, facecolor='red', alpha=0.5)
        plt.xticks(X, color='black', rotation=45)  # 横坐标旋转60度
        plt.title('%s distribution' % self.__var_name)
        plt.xlabel("%s" % self.__var_name)
        plt.legend()  # 用来显示图例
        plt.ylabel("distribution")
        # feat odds
        plt.subplot(132)
        plt.plot(X, df_result["odds_set1"], label='odds_set1', color="blue")
        plt.plot(X, df_result["odds_set2"], label='odds_set2', color="red")
        plt.xticks(X, color='black', rotation=45)  # 横坐标旋转60度
        plt.title('%s odds=bad/good' % self.__var_name)
        plt.xlabel("%s" % self.__var_name)
        plt.ylabel("odds")
        plt.legend()  # 用来显示图例
        # woe
        plt.subplot(133)
        plt.plot(X, df_result["woe_set1"], label='woe_set1', color="blue")
        plt.bar(X, df_result["woe_set1"], width=0.5, bottom=0, facecolor='blue', alpha=0.5)
        plt.plot(X, df_result["woe_set2"], label='woe_set2', color="red")
        plt.bar(X, df_result["woe_set2"], width=0.5, bottom=0, facecolor='red', alpha=0.5)
        plt.xticks(X, color='black', rotation=45)  # 横坐标旋转60度
        plt.title('%s woe' % self.__var_name)
        plt.xlabel("%s" % self.__var_name)
        plt.ylabel("woe")
        plt.legend()  # 用来显示图例
        path = os.path.join(self.__save_path, self.__var_name+'.png')
        print("save_path is %s" % path)
        plt.savefig(path)
        plt.close()


if __name__ == '__main__':
    # test
    save_path = r"./result"
    df1 = pd.read_csv(r"./source/data1.csv")
    df2 = pd.read_csv(r"./source/data2.csv")
    
    # feat_list
    var_list = df1.columns
    
    var_threshold = 30  # 当unique(特征值)的个数<50的时候，该变量当做离散变量处理
    bins = 10  # 分成30箱
    divide_mode = "qcut"  # 分箱模式："cut"等宽，"qcut"等频
    for var in var_list:
        # 去掉空值
        df_sub1 = df1[df1[var].notnull()]
        df_sub2 = df2[df2[var].notnull()]
        variabledistribution = VarDistrOddsWoe(var_name=var, value_list1=df_sub1[var].tolist(),
                                           label_list1=df_sub1["label"].tolist(), value_list2=df_sub2[var].tolist(),
                                           label_list2=df_sub2["label"].tolist(), var_threshold=var_threshold, bins=bins,
                                           divide_mode=divide_mode, save_path=save_path)
