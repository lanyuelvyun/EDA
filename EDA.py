# -*- coding:utf-8 -*-
"""
@author: lisiqi
@time: 2019/4/19 12:00
@目的：探索性数据分析（Exploratory Data Analysis，简称EDA）
"""
import pandas as pd
import random
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

pd.set_option('precision', 4)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('expand_frame_repr', False)


class DataAnalysis(object):
    def __init__(self, df, var_list, save_path):
        """
        探索性数据分析（Exploratory Data Analysis，简称EDA）
        :param df: pd.DataFrame, at least contains feature and label
        :param var_list: the list of feature_namw
        :param save_path:
        """
        self.df = df
        self.var_list = var_list
        self.save_path = save_path
        self.df_var_stat = pd.DataFrame({"var_name": self.var_list})
        self.cal_missing_rate()
        self.cal_std()
        self.cal_corr()
        self.df_var_stat.to_csv(self.save_path + r"\var_stat.csv", index=None)

    # 计算所有特征的缺失率
    def cal_missing_rate(self):
        total_count = self.df.shape[0]
        good_count = self.df[self.df["label"] == 0].shape[0]
        bad_count = self.df[self.df["label"] == 1].shape[0]

        for var in self.var_list:
            # 整体缺失率
            all_miss_rate = self.df[self.df[var].isnull()].shape[0] * 1.0 / total_count
            # good样本中的缺失率
            good_miss_rate = self.df[(self.df["label"] == 0) & (self.df[var].isnull())].shape[0] * 1.0 / good_count
            # 坏样本中的缺失率
            bad_miss_rate = self.df[(self.df["label"] == 1) & (self.df[var].isnull())].shape[0] * 1.0 / bad_count
            self.df_var_stat.ix[self.df_var_stat["var_name"] == var, "miss_rate_on_all"] = all_miss_rate
            self.df_var_stat.ix[self.df_var_stat["var_name"] == var, "miss_rate_on_good"] = good_miss_rate
            self.df_var_stat.ix[self.df_var_stat["var_name"] == var, "miss_rate_on_bad"] = bad_miss_rate
        print("cal_missing_rate finished!")

    # 处理缺失值
    def handle_nan(self):
        from sklearn.preprocessing import Imputer
        imputer = Imputer(missing_values='NaN', strategy="mean", axis=0, copy=True)	# 将空值填充成均值
        imputer = imputer.fit(self.df)
        df_copy = imputer.transform(self.df)
        return df_copy

    # 对类别特证进行编码映射
    def encode_categorical_var(self):
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        labelencoder_X = LabelEncoder()
        df_copy = self.df.copy()
        for var in self.df.columns:
            if self.df[var].dtype in ["str", "object"]:
                df_copy[var] = labelencoder_X.fit_transform(self.df[var])
        return df_copy

	# 计算单特征方差
    def cal_std(self):
        for var in self.var_list:
            self.df_var_stat.ix[self.df_var_stat["var_name"] == var, "var_std"] = self.df[var].std() * 100.0
        print("cal_var_std finished!")

    # 计算各个特征两两之间的pearson线性相关系数(注意，没有p值，只有r值)
    def cal_corr(self):
        df_corr = self.df[self.var_list].corr(method='pearson')
        # print("画相关性的热图")
        sns.heatmap(df_corr, cmap=plt.cm.RdYlBu_r, vmin=-0.25, annot=True, vmax=0.6)
        plt.title('Correlation Heatmap')
        plt.savefig(self.save_path + r"/CorrelationHeatmap.jpg")
        plt.close()
        df_corr.to_csv(self.save_path + r"/var_pearson.csv", float_format='%.4f')
        # df_corr = df.corr('kendall')  # Kendall Tau相关系数
        # df_corr = df.corr('spearman')  # spearman秩相关系数
        # df_corr1 = df.corr()["overdue_day"]  # 只显示其他特征与“overdue_day”的相关系数
        # df_corr2 = df["td_credit_score"].corr(df["td_id_risk"])  # 只计算2个变量之间的相关性系数
        print("cal_var_corr finished!")

    # 得到特征与标签label的散点图
    def get_var_label_scatter(self):
        df_copy = self.df.copy()
        df_copy.fillna(-1, inplace=True)
        for var in self.var_list:
            # plt.figure()
            plt.scatter(df_copy[var], df_copy["label"], edgecolors='red')
            plt.xlabel("%s" % var)
            plt.ylabel("label")
            # plt.show()
            plt.savefig(self.save_path + r"/" + var + "_label_scatter.jpg")
            plt.close()
        print("get_var_to_label_scatter finished!")

if __name__ == '__main__':
    df = pd.read_csv(r"E:\work\2 mission\acard\acard_dae\source\dt0709\final_main_dt0709_new_overdue.csv",
                     na_values=[-1.0, -2.0, -3.0, -99.0, -999.0])
    df["label"] = df.apply(lambda x: 1 if x["overdue_day"] > 7 else 0, axis=1)
    save_path = r"E:\work\2 mission\5 feature_engineer\result"
    var_list = [x for x in df.columns.tolist() if len(x) == 8]
    # var_list = [x for x in var_list if x not in ["99990002","99990034","99990035","99990061","99990062","99990082"]]

    data_analysis = DataAnalysis(df, var_list, save_path)
