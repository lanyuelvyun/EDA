# -*- coding:utf-8 -*-
"""
@author: lanyue
@time: 2022/07/07 15:21
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


class VarDistrOddsWoeIv(object):
    def __init__(self, var_name, value_list, label_list, var_threshold, bins, cut_mode, save_path):
        """
        @目的：针对2分类任务的变量分析
        1）分布图：查看变量值的分布情况
        2）odds图：查看变量的区分能力
        3) woe图：查看变量的区分能力
        :param var_name: variable name
        :param value_list: the value_list of the variable
        :param label_list: sample label_list, like[1,1,0,0,1,0,0,0,0].one-to-one correspondence with value_list. 
        :param var_threshold: 阈值，用于区分连续变量和离散变量：如果该变量的唯一值个数>=var_threshold，认为是连续变量，否则离散变量；
        :param bins: 分箱个数（计算IV值的时候）
        :param cut_mode: 分箱方式，值为：cut等宽分箱、qcut等频分箱。
        :param save_path: 
        :return:
        """
        if len(value_list) != len(label_list):
            raise ValueError("len(value) and len(label) is not equal, please check!!!")
        self.__var_name = var_name
        self.__value_list = value_list
        self.__label_list = label_list
        self.__var_threshold = var_threshold
        self.__bins = bins
        self.__cut_mode = cut_mode
        self.__save_path = save_path
        #self.get_result()

    def get_result(self):
        df_distr_odds = self.get_var_distr_odds()
        df_woe_iv = self.get_woe_iv()
        # 合并
        df_result = df_distr_odds.merge(df_woe_iv, left_index=True, right_index=True, how="outer")
        df_result.to_csv(self.__save_path + '/%s_distr_odds_woe_iv.csv' % self.__var_name)    
        # 画图
        self.__plot(df_result)
        

    def __get_bins(self):
        """
        对变量进行分箱
        1、区分连续变量和离散变量：如果唯一值的个数>=self.__var_threshold，认为是连续变量，否则离散变量；
        2、连续变量：等宽+等频。离散：每个值单独分一箱。
        3、空值单独作为一箱。
        """
        # 手写等频分箱(#注意：空值被分在了第0箱)
        def Rank_qcut(feat_list, k):
            quantile = np.array([float(i)*1.0/k for i in list(range(1,k+1))])
            funBounder = lambda x: (quantile>=x).argmax()
            return feat_list.rank(pct=True).apply(funBounder)
        
        df_var = pd.DataFrame({self.__var_name: self.__value_list, "label": self.__label_list})
        value_cnt = len(df_var[self.__var_name].unique())
        print("唯一值个数 = ", value_cnt)
        if value_cnt >= self.__var_threshold:
            if self.__cut_mode == 'qcut': # 等频分箱
                df_var["bins"] = pd.qcut(df_var[self.__var_name], q=self.__bins, duplicates='drop')
            elif self.__cut_mode == 'cut': # 等宽分箱，左闭右开
                df_var["bins"] = pd.cut(df_var[self.__var_name], bins=self.__bins, right=False, include_lowest=True)
            else: 
                print("参数 cut_mode 不能输入这种值！！其输入值只有两种：cut、qcut，代表两种分箱方式。")
            # 将空值单独分一箱（Category数据，要想插入一个之前没有的值，首先需要将这个值添加到.categories的容器中，然后再添加值。）
            df_var["bins"] = df_var["bins"].cat.add_categories(['NAN'])
            df_var["bins"].fillna("NAN", inplace=True)
        else: 
            df_var["bins"] = df_var[self.__var_name]
            df_var["bins"].fillna("NAN", inplace=True)
        print("分箱 bins = ", df_var["bins"].unique())
        return df_var

    def get_var_distr_odds(self):
        """
        对该变量进行分箱
        1）变量频数分布图：统计每箱内的频数占比
        2）变量odds图：统计每箱内的label=1的样本个数/label=0的样本个数
        """
        df_with_bins = self.__get_bins() # 分箱
        
        print("-->> get var distr!")
        df_distr = df_with_bins.groupby('bins').apply(lambda x: x.shape[0] * 1.0 / (len(self.__value_list) + 1e-20))
        
        print("-->> get var odds!")
        result = df_with_bins.groupby('bins').apply(lambda x: (
            x[(x["label"] == 1)].shape[0]
            ,x[(x["label"] == 0)].shape[0]
            ))
        df_odds = result.map(lambda x: x[0]) * 1.0 / (result.map(lambda x: x[1]) + 1e-20)

        # 合并
        df_result = df_distr.merge(df_odds, left_index=True, right_index=True, how="outer")
        df_result.to_csv(self.__save_path + '/%s_distr_odds.csv' % self.__var_name)   
        return df_result


    def get_woe_iv(self):
        """
        变量woe/iv值：对该变量进行分箱，计算每箱内的woe/iv
        """
        print("-->> get var woe!")
        df_with_bins = self.__get_bins() # 分箱
        total_p_cnt = (df_with_bins["label"]==1).sum()
        total_n_cnt = (df_with_bins["label"]==0).sum()
        df_result = df_with_bins.groupby('bins').apply(lambda x: self._cal_woe_tmp(x, total_p_cnt, total_n_cnt)).reset_index(level=1, drop=True)
        df_result["woe"] = np.log(df_result["p_rate"] * 1.0 / (df_result["n_rate"] + 1e-20))
        df_result["iv"] = (df_result["p_rate"] - df_result["n_rate"]) * df_result["woe"]
        df_result["sum_iv"] = df_result["iv"].sum()
        df_result["max_abs_woe"] = max(df_result["woe"].tolist(), key=abs)
        return df_result

    @staticmethod
    def _cal_woe_tmp(df, total_p_cnt, total_n_cnt):
        # 计算IV值的时候，分箱计算，存储中间值
        return pd.DataFrame.from_dict(
            {
                "total_p_cnt": total_p_cnt,
                "total_n_cnt": total_n_cnt,
                "p_cnt": (df["label"] == 1).sum(),
                "p_rate": (df["label"] == 1).sum() * 1.0 / (total_p_cnt + 1e-20), # 该箱内坏样本/总坏样本
                "n_cnt": (df["label"] == 0).sum(),
                "n_rate": (df["label"] == 0).sum() * 1.0 / (total_n_cnt + 1e-20), # 该箱内好样本/总好样本
            }, orient='index').T


    def __plot(self, df_result):
        plt.figure(figsize=(20, 20))
        plt.tick_params(labelsize=8)  # tick_params可设置坐标轴刻度值属性
        X = [str(x) for x in df_result.index]
        need_to_plot = ["distr","odds=bad/good","woe"]
        for col,j in zip(need_to_plot, list(np.arange(1,len(need_to_plot)+1,1))):
            plt.subplot(1,len(need_to_plot),j) # 分画布
            # plt.bar(list(range(len(X))), df_result[col],label=col,width=0.5, bottom=0, facecolor='blue', alpha=0.5)
            # plt.plot(list(range(len(X))), df_result[col], label=col, color="blue")
            plt.bar(X, df_result[col],label=col,width=0.5, bottom=0, facecolor='blue', alpha=0.5)
            plt.plot(X, df_result[col], label=col, color="blue")
            # 在曲线上显示Y值
            for a, b in zip(X, df_result[col]):
                plt.text(a, b, '%s' % (round(b, 3)), ha='center', va='bottom', fontsize=10) # plt.text 在曲线上显示y值
            # 横坐标旋转
            plt.xticks(list(range(len(X))), tuple(X), color='black', rotation=30) 
            plt.title('%s_%s' % (self.__var_name, col))
            plt.xlabel("%s" % self.__var_name)
            plt.ylabel(col)
            plt.legend()  # 用来显示图例
        path = os.path.join(self.__save_path, self.__var_name+'_distr_odds_woe.png')
        print("save_path is %s" % path)
        plt.savefig(path)
        plt.close()


if __name__ == '__main__':
    # 使用jupyter调用本脚本的时候，需要使用这几句，进行加载
#     import imp
#     import VarDistrOddsWoeIv
#     imp.reload(VarDistrOddsWoeIv)

    # tmp
    save_path = r'/tmp'
    df = pd.read_csv(save_path + r'/TD评分测试结果_with_label.csv', sep='\t')
    print(df.shape)
    df['label'] = df['label_d7_all_max']
    feat_list = ['x1', 'x2', 'x3','x4', 'x5', 'x6']
    
    # 将指定值，变成空
    df[feat_list] = df[feat_list].applymap(lambda x: np.nan if x<0 else x)   

    # 保存整体iv值
    df_iv = pd.DataFrame(columns={'var_name','iv','max_abs_woe'})
    df_iv['var_name'] = feat_list
    for var in ['msg_loan_cnt_receive_top_a_sum_pt_avg_7d']:
        print((" var =   %s" % var).center(80, '*'))
        if len(df[var].unique())>=3: # 除去这种类型的特征：[1，1，1，nan,1,1] 和[1,1,1,1,1]
            var_instance = VarDistrOddsWoeIv.VarDistrOddsWoeIv(
                var_name=var,
                value_list=df[var].tolist(),
                label_list=df["label"].tolist(),
                var_threshold=5,
                bins=5, # 分箱个数
                cut_mode='qcut', # 等频分箱
                save_path=save_path)
            df_woe_iv = var_instance.get_woe_iv()
            df_woe_iv.to_csv(save_path + r'%s_woe_iv.csv' % var, index=None)
            df_iv.loc[ df_iv['var_name']==var, 'iv'] = df_woe_iv['sum_iv'].unique()
            df_iv.loc[ df_iv['var_name']==var, 'max_abs_woe'] = df_woe_iv['max_abs_woe'].unique()
    df_iv.to_csv(save_path + r'iv.csv', index=None)
            
          
