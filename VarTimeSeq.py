# -*- coding:utf-8 -*-
"""
@author: lisiqi
@time: 2019/9/17 20:00
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
try:
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

class VarTimeSeq(object):
    def __init__(self, var_name, df, split_col, save_path):
        """
        @目的：针对2分类任务的变量【时序稳定性】分析，下面三张图要结合在一起看，单独看不全面！
        1）分位数分布图：用时间进行分箱，粒度可以是月/周/天，查看每箱内该特征的分位数
        2）覆盖率分布图：用时间进行分箱，粒度可以是月/周/天，查看每箱内该特征的覆盖率
        3) odds图：用时间进行分箱，粒度可以是月/周/天，查看每箱内的逾期率
        :param var_name: 特征名称
        :param df: 包含特征var_name和label的dataframe
        :param split_col: 用来进行分箱的列，一般是时间，粒度可以是月/周/天
        :param save_path:
        """
        self.__var_name = var_name
        self.__df = df
        self.__split_col = split_col
        self.__save_path = save_path
        self.__get_result()

    def __get_result(self):
        df_quantile = self.get_var_quantile()
        df_coverage = self.get_var_coverage()
        df_odds = self.get_var_odds()
        df_distr = self.get_var_distr()
        df_result = pd.concat([df_quantile, df_coverage, df_odds, df_distr], axis=1, join='outer')  # 按照列进行拼接
        df_result.rename(columns={0:"coverage", 1:"odds", 2:"distr"}, inplace=True)

        df_result.to_csv(self.__save_path + '/%s_quantile_coverage_odds_distr.csv' % self.__var_name, index=True,encoding="utf-8")
        print(self.__save_path + '/%s_quantile_coverage_odds_distr.csv' % self.__var_name)
        self.__plot(df_result)

        
    def get_var_distr(self):
        """
        变量频数分布：用时间对该变量进行分箱，统计每一箱内的样本频数占比
        """
        print("get var distr".center(80, '*'))
        df_copy = self.__df.copy()
        df_distr = df_copy.groupby(self.__split_col).apply(lambda x: x.shape[0]*1.0 / (df_copy.shape[0]+1e-20))
        return df_distr 
        
        
    def get_var_odds(self):
        """
        变量odds：用时间对该变量进行分箱，统计每一箱内的label=1的样本个数/label=0的样本个数
        """
        print("get var odds".center(80,'*'))
        df_copy = self.__df.copy()
        df_odds = df_copy.groupby(self.__split_col).apply(lambda x: x["label"].sum()*1.0 / (x[x["label"] == 0].shape[0]+1e-20))
        return df_odds

    def get_var_coverage(self):
        """
        变量覆盖度：用时间对该变量进行分箱，统计每一箱内该变量的覆盖度。（ps：注意空值的判断逻辑，可根据实际情况调整）
        """
        print("get var coverage".center(80, '*'))
        df_copy = self.__df.copy()
        null_value_list = ["NULL", "null", "", np.nan]
        df_coverage = df_copy.groupby(self.__split_col).apply(lambda x: x[~x[self.__var_name].isin(null_value_list)][x[self.__var_name].notnull()].shape[0]*1.0 / (x.shape[0]+1e-20))
        return df_coverage

    def get_var_quantile(self):
        """
        变量各分位数：用时间对该变量进行分箱，统计每一箱内的各个分位数，观察其跨时间稳定性。
        （ps：统计的时候排除了空值!!!注意空值的判断逻辑，可以根据实际情况调整。
        """
        print("get var quantile".center(80, '*'))
        # 去掉空值
        df_copy = self.__df.copy()
        df_copy = df_copy[~df_copy[self.__var_name].isin(["NULL", "null", "", np.nan])]
        df_copy = df_copy[df_copy[self.__var_name].notnull()]  # 注意pandas 中的null，不是np.nan!!!
        # 计算每一分组内的分位数
        df_quantile = df_copy.groupby(self.__split_col).apply(lambda x: self.__cal_quantile(x)).reset_index(level=1, drop=True)
        return df_quantile

    def __cal_quantile(self, df):
    # 计算每一分组内该特征的分位数
        return pd.DataFrame.from_dict(
            {
                "min": df[self.__var_name].min(),
                "20%": df[self.__var_name].quantile(0.2),
                "40%": df[self.__var_name].quantile(0.4),
                "50%": df[self.__var_name].quantile(0.5),
                "60%": df[self.__var_name].quantile(0.6),
                "80%": df[self.__var_name].quantile(0.8),
                "90%": df[self.__var_name].quantile(0.9),
                "max": df[self.__var_name].max(),
                "mean": df[self.__var_name].mean()
            }, orient='index').T

    def __plot(self, df_result):
        print("start plot!")
        plt.figure(figsize=(20, 8))
        plt.tick_params(labelsize=8)  # tick_params可设置坐标轴刻度值属性
        X = [str(x) for x in df_result.index]
        for col,j in zip(["quantile","coverage","odds"],[1,2,3]):
            plt.subplot(1,3,j)
            if col == "quantile":
                for i in ["min", "20%", "40%", "50%", "60%", "80%", "90%", "max", "mean"]:
            #         plt.plot(list(range(len(X))), df_result[i], label=i) 
            #         plt.xticks(list(range(len(X))), tuple(X), color='black', rotation=45) # 横坐标旋转60度
                    plt.plot(X, df_result[i], label=i) 
                    plt.xticks(X, tuple(X), color='black', rotation=45) # 横坐标旋转60度
                    for a, b in zip(X, df_result[i]):
                        plt.text(a, b, '%s' % (round(b, 3)), ha='center', va='bottom', fontsize=10) # plt.text 在曲线上显示y值
            else:
                for i in ["coverage","odds","distr"]:
                #     plt.plot(list(range(len(X))), df_result[i], label=i, ior="green")
                #     plt.xticks(list(range(len(X))), tuple(X), ior='black', rotation=45) # 横坐标旋转60度
                    plt.plot(X, df_result[i], label=i)
                    plt.xticks(X, tuple(X), color='black', rotation=45) # 横坐标旋转60度
                    for a, b in zip(X, df_result[i]): 
                        plt.text(a, b, '%s' % (round(b, 3)), ha='center', va='bottom', fontsize=10)  # plt.text 在曲线上显示y值    
            plt.grid(axis='y', color='#8f8f8f', linestyle='--', linewidth=1)  # 显示网格
            plt.title('%s_%s' % (self.__var_name, col))
            plt.xlabel("%s" % self.__split_col)
            plt.ylabel(col)
            plt.legend()  # 用来显示图例
        path = os.path.join(self.__save_path, self.__var_name+'_quantile_coverage_odds_distr.png')
        print("save_path is %s" % path)
        plt.savefig(path)
        plt.close()
        
if __name__ == '__main__':
    save_path = r"./plot_feat"
    
    # load data. 包含label和apply_month
    df_all = pd.read_csv("data.csv")
    # feat_list
    var_list = [x for x in df_all.columns.tolist() if x not in ["user_id","label","apply_month"]] # 去掉不相关的字段

    # 用来进行分组的字段：时间-月份
    split_col = 'apply_month'
    for var in var_list:
        print(("feature %s" % var).center(80, '-'))
        var_time_seq = VarTimeSeq(var_name=var, df=df_all, split_col=split_col, save_path=save_path)
