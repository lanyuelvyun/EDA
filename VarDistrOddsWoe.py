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
        2）odds图/woe图：比较2个集合中，同一变量对2类样本的区分能力，查看变量区分能力是否一致
        3) 计算特征在2个集合中的IV和max(abs(woe))
        :param var_name: 特征字段名
        :param value_list1: 集合1中，该特征的特征值
        :param label_list1: 集合1中，样本标签,与value_list1一一对应，举例[1,1,0,0,1,0,0,0,0]
        :param value_list2: 集合2中，该特征的特征值
        :param label_list2: 集合2中，样本标签,与value_list2一一对应，举例[1,1,0,0,1,0,0,0,0]
        :param var_threshold: 该特征唯一值的个数
        :param bins: 分箱数
        :param divide_mode: 分箱模式：="qcut",等频分箱; ="cut",等宽分箱
        :param save_path: 结果保存路径
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

    def get_result(self):
        df_distr_bins_cut, df_distr_bins_qcut = self.get_var_distr()
        df_odds_bins_cut, df_odds_bins_qcut = self.get_var_odds()
        df_woe_bins_cut, df_woe_bins_qcut = self.get_var_woe()
        # 合并
        df_result_cut = df_distr_bins_cut.merge(df_odds_bins_cut, left_index=True, right_index=True, how="outer")
        df_result_cut = df_result_cut.merge(df_woe_bins_cut, left_index=True, right_index=True, how="outer")
        df_result_cut.to_csv(self.__save_path + '/%s_distr_odds_woe_bins_cut.csv' % self.__var_name)
        df_result_qcut = df_distr_bins_qcut.merge(df_odds_bins_qcut, left_index=True, right_index=True, how="outer")
        df_result_qcut = df_result_qcut.merge(df_woe_bins_qcut, left_index=True, right_index=True, how="outer")
        df_result_qcut.to_csv(self.__save_path + '/%s_distr_odds_woe_bins_qcut.csv' % self.__var_name)      
        # 画图
        self.__plot(df_result_cut, df_result_qcut)
        return df_result_cut, df_result_qcut
        

    def __divide_var(self):
        """
        对变量进行分箱：
        1、为了使得两个集合具有相同的分箱区间，画图的时候方便：将两个集合的该变量值放在一起，再进行分箱；
        2、区分连续变量和离散变量：如果唯一值的个数>=self.__cnt_threshold，认为是连续变量，否则离散变量；
        3、连续变量：等宽+等频。离散：每个值单独分一箱
        """
        # 等频分箱
        def Rank_qcut(var_list, k):
            quantile = np.array([float(i)*1.0/k for i in list(range(1,k+1))])
            funBounder = lambda x: (quantile>=x).argmax()
            return var_list.rank(pct=True).apply(funBounder)
        
        value_list = self.__value_list1 + self.__value_list2
        label_list = self.__label_list1 + self.__label_list2
        # 添加一个flag，用于区分两个集合: 1 is value_list1, 2 is value_list2
        flag = list(np.ones(len(self.__value_list1))) + list(np.ones(len(self.__value_list2))*2)
        df_var = pd.DataFrame({self.__var_name: value_list, "label": label_list, "flag": flag})
        
        value_cnt = len(df_var[self.__var_name].unique())
        print("unique_value_cnt = ", value_cnt)
        if value_cnt >= self.__cnt_threshold:
            # 等频分箱
            #pd.qcut分箱，边界有重复，虽然在较高版本中有duplicates参数可以解决这个问题
            #但是如果为了删除重复值设置 duplicates=‘drop'，则易出现于分片个数少于指定个数的问题
            #df_var["bins_qcut"] = pd.qcut(df_var[self.__var_name], q=self.__bins) 
            df_var["bins_qcut"] = Rank_qcut(df_var[self.__var_name], k=self.__bins) # 自己编写的等频分箱，注意空值分到了0箱里面
            # 等宽分箱，左闭右开
            df_var["bins_cut"] = pd.cut(df_var[self.__var_name], bins=self.__bins, right=False, include_lowest=True)
        else: 
            df_var["bins_qcut"] = df_var[self.__var_name]
            df_var["bins_cut"] = df_var[self.__var_name]
        # df_var = df_var.sort_values(by="bins")
        print("bins_qcut: ", df_var["bins_qcut"].unique())
        print("bins_cut: ", df_var["bins_cut"].unique())
        return df_var

    def get_var_distr(self):
        """
        变量频数分布图：对该变量进行分箱（等频+等宽），统计每一箱内的频数占比
        """
        print("get var distr".center(40, '_'))
        df_with_bins = self.__divide_var() # 先分箱（等频+等宽）
        for mode in ["bins_cut", "bins_qcut"]:
            result = df_with_bins.groupby(mode).apply(lambda x: (
                x[x["flag"]==1].shape[0] * 1.0 / (len(self.__value_list1) + 1e-20),
                x[x["flag"]==2].shape[0] * 1.0 / (len(self.__value_list2) + 1e-20)))
            result1 = result.map(lambda x: x[0])
            result2 = result.map(lambda x: x[1])
            if mode == 'bins_cut':
                df_distr_bins_cut = pd.concat([result1, result2], axis=1, keys=["distr_set1", "distr_set2"])
            else:
                df_distr_bins_qcut = pd.concat([result1, result2], axis=1, keys=["distr_set1", "distr_set2"])
        return df_distr_bins_cut, df_distr_bins_qcut

    def get_var_odds(self):
        """
        变量odds图：对该变量进行分箱（等频+等宽），统计每一箱内的label=1的样本个数/label=0的样本个数
        """
        print("get var odds".center(40, '_'))
        df_with_bins = self.__divide_var() # 先分箱（等频+等宽）
        for mode in ["bins_cut", "bins_qcut"]:
            result = df_with_bins.groupby(mode).apply(lambda x: (
                x[(x["flag"] == 1) & (x["label"] == 1)].shape[0], 
                x[(x["flag"] == 1) & (x["label"] == 0)].shape[0],
                x[(x["flag"] == 2) & (x["label"] == 1)].shape[0], 
                x[(x["flag"] == 2) & (x["label"] == 0)].shape[0]))
            df_odds1 = result.map(lambda x: x[0]) * 1.0 / (result.map(lambda x: x[1]) + 1e-20)
            df_odds2 = result.map(lambda x: x[2]) * 1.0 / (result.map(lambda x: x[3]) + 1e-20)
            # 按列合并：pd.concat沿着axis=1对series进行合并时，keys会成为合并之后df的列名
            if mode == 'bins_cut':
                df_odds_bins_cut = pd.concat([df_odds1, df_odds2], axis=1, keys=["odds_set1", "odds_set2"])
            else:
                df_odds_bins_qcut = pd.concat([df_odds1, df_odds2], axis=1, keys=["odds_set1", "odds_set2"])
        return df_odds_bins_cut, df_odds_bins_qcut

    def get_var_woe(self):
        """
        变量每一个分箱的woe/iv值：对该变量进行分箱（等频+等宽），计算每一箱内的woe/iv
        """
        print("get var woe".center(40, '_'))
        df_with_bins = self.__divide_var() # 先分箱（等频+等宽）
        for mode in ["bins_cut", "bins_qcut"]:
            for flag in [1,2]: # 对于两个集合
                total_p_cnt = df_with_bins[(df_with_bins["flag"] == flag)&(df_with_bins["label"]==1)].shape[0]
                total_n_cnt = df_with_bins[(df_with_bins["flag"] == flag)&(df_with_bins["label"]==0)].shape[0]
                df_result = df_with_bins.groupby(mode).apply(lambda x: self._cal_woe_tmp(x[x["flag"] == flag], total_p_cnt, total_n_cnt)).reset_index(level=1, drop=True)
                df_result["woe"] = np.log(df_result["p_rate"] * 1.0 / (df_result["n_rate"] + 1e-20))
                df_result["iv"] = (df_result["p_rate"] - df_result["n_rate"]) * df_result["woe"]
                df_result.columns = [x+'_set'+str(flag) for x in df_result.columns]
                if flag == 1:
                    df_woe_set1 = df_result
                else:
                    df_woe_set2 = df_result
            if mode == 'bins_cut':
                df_woe_bins_cut = df_woe_set1.merge(df_woe_set2,left_index=True,right_index=True, how="outer")
            else:
                df_woe_bins_qcut = df_woe_set1.merge(df_woe_set2,left_index=True, right_index=True, how="outer")
        return df_woe_bins_cut,df_woe_bins_qcut

    @staticmethod
    def _cal_woe_tmp(df, total_p_cnt, total_n_cnt):
        # 计算IV值的时候，分箱计算，存储中间值
        return pd.DataFrame.from_dict(
            {
                "total_p_cnt": total_p_cnt,
                "total_n_cnt": total_n_cnt,
                "p_cnt": df[df["label"] == 1].shape[0],
                "p_rate": df[df["label"] == 1].shape[0] * 1.0 / (total_p_cnt + 1e-20), # 该箱内坏样本/总坏样本
                "n_cnt": df[df["label"] == 0].shape[0],
                "n_rate": df[df["label"] == 0].shape[0] * 1.0 / (total_n_cnt + 1e-20), # 该箱内好样本/总好样本
            }, orient='index').T


    def __plot(self, df_result_cut, df_result_qcut):
        plt.figure(figsize=(20, 15))
        plt.tick_params(labelsize=8)  # tick_params可设置坐标轴刻度值属性
        for df_result,i in zip([df_result_cut, df_result_qcut], [0,3]):
            X = [str(x) for x in df_result.index]
            for col,j in zip(["distr","odds","woe"],[i+1,i+2,i+3]):
                plt.subplot(2,3,j)
                col1 = col+"_set1"
                col2 = col+"_set2"
                plt.bar(list(range(len(X))), df_result[col1],
                        label=col1,width=0.5, bottom=0, facecolor='blue', alpha=0.5)
                plt.plot(list(range(len(X))), df_result[col1], label=col1, color="blue")
                plt.bar(list(range(len(X))), df_result[col2], 
                        label=col2, width=0.5, bottom=0, facecolor='red', alpha=0.5)
                plt.plot(list(range(len(X))), df_result[col2], label=col2, color="red")
                plt.xticks(list(range(len(X))), tuple(X), color='black', rotation=45) # 横坐标旋转
                plt.title('%s_%s' % (self.__var_name, col))
                plt.xlabel("%s" % self.__var_name)
                plt.ylabel(col)
                plt.legend()  # 用来显示图例
        path = os.path.join(self.__save_path, self.__var_name+'.png')
        print("save_path is %s" % path)
        plt.savefig(path)
        plt.close()

if __name__ == '__main__':
     # test
    save_path = r"..\result\Odds"
    # df1与df2必须有label
    df1 = pd.read_csv(r"../data1.csv", encoding='utf-8', na_values=[-1.0, -2.0, -3.0, -99.0, -999.0])
    df2 = pd.read_csv(r"../data2.csv", encoding='utf-8', na_values=[-1.0, -2.0, -3.0, -99.0, -999.0])

    # feat_list
    var_list = df1.columns.tolist()
    
    # 参数
    var_threshold = 30  # 当unique(特征值)的个数<50的时候，该变量当做离散变量处理
    bins = 10  # 分成30箱
    divide_mode = "qcut"  # 分箱模式："cut"等宽，"qcut"等频
    for var in var_list:
        variabledistribution = VarDistrOddsWoe(
            var_name=var,
            value_list1=df_sub1[var].tolist(),label_list1=df_sub1["label"].tolist(), 
            value_list2=df_sub2[var].tolist(),label_list2=df_sub2["label"].tolist(), 
            var_threshold=var_threshold, 
            bins=bins,
            divide_mode=divide_mode,
            save_path=save_path)
