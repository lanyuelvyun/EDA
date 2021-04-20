# -*- coding:utf-8 -*-
"""
@author: lanyue
@time: 2021/4/20 16:40
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.dpi'] = 150
try:
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

class VarIv(object):
    def __init__(self, var_name, value_list, label_list, var_threshold, bins, cut_mode, save_path):
        """
        @目的：针对2分类任务的变量计算IV值
        :param var_name: variable name
        :param value_list: the value_list of the variable
        :param label_list: label_list, one-to-one correspondence with value_list1. like[1,1,0,0,1,0,0,0,0]
        :param var_threshold: the number of unique values of this variable
        :param save_path: the path where the result is saved
        :return:
        """
        if len(value_list) != len(label_list):
            raise ValueError("len(value) and len(label) was not correct, please check!")
        self.__var_name = var_name
        self.__value_list = value_list
        self.__label_list = label_list
        self.__cnt_threshold = var_threshold
        self.__bins = bins
        self.__cut_mode = cut_mode
        self.__save_path = save_path
        
    def get_var_iv(self):
        """
        变量每一个分箱的woe/iv值：对该变量进行分箱（等频+等宽），计算每一箱内的woe/iv
        """
        print("get var woe".center(40, '_'))
        # 先分箱
        df_with_bins = self.__divide_var() 
        # 计算正负样本总量
        total_p_cnt = df_with_bins[(df_with_bins["label"]==1)].shape[0]
        total_n_cnt = df_with_bins[(df_with_bins["label"]==0)].shape[0]
        # 分组统计
        df_iv = df_with_bins.groupby('bins_cut').apply(lambda x: self._cal_woe_tmp(x, total_p_cnt, total_n_cnt)).reset_index(level=1, drop=True)
        df_iv["woe"] = np.log(df_iv["p_rate"] * 1.0 / (df_iv["n_rate"] + 1e-20))
        df_iv["iv"] = (df_iv["p_rate"] - df_iv["n_rate"]) * df_iv["woe"]
        df_iv["sum_iv"] = df_iv["iv"].sum()
        df_iv["max_abs_woe"] = max(df_iv["woe"].tolist(), key=abs)
        return df_iv    

    def __divide_var(self):
        """
        对变量进行分箱：
        1、为了使得两个集合具有相同的分箱区间，画图的时候方便：将两个集合的该变量值放在一起，再进行分箱；
        2、区分连续变量和离散变量：如果唯一值的个数>=self.__cnt_threshold，认为是连续变量，否则离散变量；
        3、连续变量：等宽+等频。离散：每个值单独分一箱
        """
        # 等频分箱
        def Rank_qcut(feat_list, k):
            quantile = np.array([float(i)*1.0/k for i in list(range(1,k+1))])
            funBounder = lambda x: (quantile>=x).argmax()
            return feat_list.rank(pct=True).apply(funBounder)

        df_var = pd.DataFrame({self.__var_name: self.__value_list, "label": self.__label_list})
        value_cnt = len(df_var[self.__var_name].unique())
        print("unique_value_cnt = ", value_cnt)

        if value_cnt >= self.__cnt_threshold: 
            # 连续变量
            if self.__cut_mode == 'qcut':
              # 等频分箱
              df_var["bins_cut"] = pd.qcut(df_var[self.__var_name], q=self.__bins,duplicates='drop') # pd.qcut分箱边界有重复
              #df_var["bins_qcut"] = Rank_qcut(df_var[self.__var_name], k=self.__bins) # 注意：空值被分在了第0箱
            if self.__cut_mode == 'cut':
              # 等宽分箱，左闭右开
              df_var["bins_cut"] = pd.cut(df_var[self.__var_name], bins=self.__bins, right=False, include_lowest=True)
        else: 
            # 离散变量
            df_var["bins_cut"] = df_var[self.__var_name]
        # df_var = df_var.sort_values(by="bins")
        print("bins_cut: ", df_var["bins_cut"].unique())
        return df_var

    @staticmethod
    def _cal_woe_tmp(df, total_p_cnt, total_n_cnt):
        # 计算IV值的时候，分箱计算，存储中间值
        return pd.DataFrame.from_dict(
            {
                "total_p_cnt": total_p_cnt,
                "total_n_cnt": total_n_cnt,
                "p_cnt": df[df["label"] == 1].shape[0],
                "p_rate": (df[df["label"] == 1].shape[0] + 1e-20) * 1.0 / (total_p_cnt + 1e-20), # 该箱内坏样本/总坏样本
                "n_cnt": df[df["label"] == 0].shape[0],
                "n_rate": (df[df["label"] == 0].shape[0] + 1e-20) * 1.0 / (total_n_cnt + 1e-20), # 该箱内好样本/总好样本
            }, orient='index').T

      
if __name__ == '__main__':

    # 特征分析
    save_path = r'/data/IV/'
    df = pd.read_csv(r'data.csv')
    print('df.shape = ', df.shape)
    df["label"] = df['lbl_user']
    split_col = 'credit_time'

    # 参数
    var_threshold = 20  # 当unique(特征值)的个数小于该值的时候，该变量当做离散变量处理
    bins = 20  # 分箱个数
    cut_mode = 'qcut' # 分箱方式
    
    ### 计算总的IV值
    df_iv = pd.DataFrame({"var":feat_list})
    if os.path.exists(save_path + r'feats_iv_details.csv'):
        os.remove(save_path + r'feats_iv_details.csv')
    for var in feat_list:
        print(("variable %s" % var).center(80, '*'))
        variv = VarIv(
            var_name=var,
            value_list=df[var].tolist(),
            label_list=df["label"].tolist(),
            var_threshold=var_threshold,
            bins=bins,
            cut_mode=cut_mode,
            save_path=save_path)
        df_sub_iv = variv.get_var_iv()
        # 保存每个特征的分箱结果，用于校验
        df_sub_iv['var'] = var
        df_sub_iv.to_csv(save_path + r'feats_iv_details.csv', mode='a',index=True)
        print('iv = ', list(set(df_sub_iv['sum_iv']))[0])
        df_iv.loc[df_iv['var']==var, 'IV'] = list(set(df_sub_iv['sum_iv']))[0]
    df_iv = df_iv.sort_values(by='IV', ascending=False)
    df_iv.to_csv(save_path + r"feats_iv.csv", index=None)
    
    ### 计算分月IV值
    df['credit_month'] = df['credit_time'].apply(lambda x: x[:7])
    month_list = sorted(set(df['credit_month']))
    print(month_list)
    df_iv_monthly = pd.DataFrame({"var":feat_list})
    for month in month_list:
        print('month = ', month)
        df_sub = df[df['credit_month'] == month]
        print('df_sub.shape = ', df_sub.shape)
        # 计算IV
        for var in feat_list:
            print(("variable %s" % var).center(80, '*'))
            variv = VarIv(
                var_name=var,
                value_list=df_sub[var].tolist(),
                label_list=df_sub["label"].tolist(),
                var_threshold=var_threshold,
                bins=bins,
                cut_mode=cut_mode,
                save_path=save_path)
            df_sub_iv = variv.get_var_iv()
        #     # 保存每个特征的分箱结果
        #     df_sub_iv['var'] = var
        #     df_sub_iv.to_csv(save_path + r'feats_iv_details.csv', mode='a',index=True)
            print('iv = ', list(set(df_sub_iv['sum_iv']))[0])
            df_iv_monthly.loc[df_iv_monthly['var']==var, str(month)+'_IV'] = list(set(df_sub_iv['sum_iv']))[0]
    df_iv_monthly.to_csv(save_path + r"feats_iv_monthly.csv", index=None)
    
    
