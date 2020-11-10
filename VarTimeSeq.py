# -*- coding:utf-8 -*-
"""
@author: lisiqi
@time: 2019/9/17 20:00
"""
# -*- coding:utf-8 -*-
import pandas as pd
import os
import numpy as np
import math
import random
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
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
    def __init__(self, df, var, label, split_col, save_path):
        """
        @目的：针对2分类任务的变量【时序性】分析，下面几张图要结合在一起看，单独看不全面！用时间进行分箱，粒度可以是月/周/天
        1）时序分布稳定性：统计每箱内，分位数+最大最小值+平均值+样本占比+psi。（ps:去掉空值计算），再结合该特征覆盖度一起看。
        2）时序表现稳定性：统计每箱内，该特征覆盖度+auc+ks+样本bad_rate。（ps:不去掉空值计算，因为在实际使用的时候，不会只使用该特征覆盖样本建模）
        :param var: 特征名称
        :param df: 包含特征var和label的dataframe
        :param split_col: 用来进行分箱的列，一般是时间，粒度可以是月/周/天
        :param save_path:
        """
        self.__df = df
        self.__var = var
        self.__label = label
        self.__split_col = split_col
        self.__save_path = save_path
        self.get_result()

    def get_result(self):
        df_coverage = self.get_var_coverage()
        df_quantile = self.get_var_quantile()
        df_odds = self.get_var_odds()
        df_distr = self.get_var_distr()
        df_psi = self.get_var_psi()
        df_auc = self.get_var_auc()
        df_result = pd.concat([df_quantile, df_coverage, df_odds, df_distr, df_psi, df_auc], axis=1, join='outer')  # 按照列进行拼接
        df_result.rename(columns={0:"coverage", 1:"odds", 2:"distr", 3:"psi",4:"auc"}, inplace=True)

        df_result.to_csv(self.__save_path + '/%s_quantile_coverage_odds_distr_psi_auc.csv' % self.__var, index=True,encoding="utf-8")
        print(self.__save_path + '/%s_quantile_coverage_odds_distr_psi_auc.csv' % self.__var)
        self.plot_result(df_result)
        
    def get_var_coverage(self):
        """
        变量覆盖度：用时间对该变量进行分箱，统计每一箱内该变量的覆盖度（不为空的样本占比）。（ps：注意空值的判断逻辑，可根据实际情况调整）
        """
        print("get var coverage".center(80, '*'))
        null_value_list = ["NULL", "null", "", np.nan]
        df_coverage = self.__df.groupby(self.__split_col).apply(lambda x: x[(~x[self.__var].isin(null_value_list)) & (x[self.__var].notnull())].shape[0]*1.0 / (x.shape[0]+1e-20))
        return df_coverage
    
    def get_var_odds(self):
        """
        样本odds：用时间对该变量进行分箱，统计每一箱内的label=1的样本个数/label=0的样本个数（不去掉空值样本）
        """
        print("get var odds".center(80,'*'))
        df_odds = self.__df.groupby(self.__split_col).apply(lambda x: x[self.__label].sum()*1.0 / (x[x[self.__label] == 0].shape[0]+1e-20))
        return df_odds
        
    @staticmethod
    def single_train(label_list,var_list):
        """
        用该特征单独训练一个模型，得到pvalue，用来计算AUC/KS。K折交叉赋分，比较准确             
        """
        print("用该特征单独训练一个模型，得到pvalue，用来计算AUC/KS。K折交叉赋分，比较准确!")
        df = pd.DataFrame({'label':label_list, 'var':var_list})
        df['label'] = df['label'].astype('int64')
        clf = LGBMClassifier(
                  metric='binary_logloss', is_unbalance=True, random_state=11, 
                  silent=True, n_jobs=10, reg_alpha=0.3, reg_lambda=0.3
                  ,learning_rate=0.01, n_estimators=2000, subsample=0.6, colsample_bytree=0.3
                  ,num_leaves=7, max_depth=3, min_child_samples=2000
                  ,min_split_gain=0.1, min_child_weight=0.1
                  #,is_training_metric=True, max_bin=255, subsample_for_bin=400, ,objective='binary'
                  ,importance_type='gain',
                  )     
        # kfold
        kf = KFold(n_splits=3, shuffle=True) # 注意交叉赋分的时候必须要打乱顺序，保证好坏样本比例类似！！！要不然每一折的test AUC不稳定
        df = df.reset_index(drop=True) # index必须从0开始，要不然交叉的时候出错
        for train_index, test_index in kf.split(df): 
            clf.fit(df.loc[train_index, 'var'].values.reshape(-1, 1),df.loc[train_index, 'label']) 
            prob_list = clf.predict_proba(df.loc[test_index, 'var'].values.reshape(-1, 1))[:, 1] # 给test集打分 
            df.loc[test_index, "pvalue"] = prob_list  # 保留交叉赋分之后的pvalue
        return df['pvalue'].tolist()
        
    @staticmethod 
    def cal_auc(label_list,prob_list):
        auc = metrics.roc_auc_score(label_list, prob_list)
        fpr,tpr,thresholds= metrics.roc_curve(label_list, prob_list)
        ks = max(tpr-fpr)
        return pd.DataFrame.from_dict({'auc':auc, 'ks':ks}, orient='index').T
    
    def get_var_auc(self):
        """
        变量的跨月表现AUC/KS：用时间对该变量进行分箱，计算每一箱内特征的AUC/KS（不去掉空值样本）.
        注意，此计算AUC的方法是，用每箱样本单独训练一个模型，然后计算该模型分的AUC，当每箱内样本少的时候，此计算方法不太准确
        """        
        print("get var bins_auc".center(80, '*'))        
        df_copy = self.__df[[self.__var, self.__label,self.__split_col]]
        df_copy['pvalue'] = self.single_train(df_copy[self.__var], df_copy[self.__label])        
        df_auc = df_copy.groupby(self.__split_col).apply(lambda x: self.cal_auc(x[self.__label],x['pvalue'])).reset_index(level=1, drop=True)
        return df_auc
               
    @staticmethod    
    def __cal_quantile(df,var):
    # 计算分位数
        return pd.DataFrame.from_dict(
            {
                "min": df[var].min(),
                "20%": df[var].quantile(0.2),
                "40%": df[var].quantile(0.4),
                "50%": df[var].quantile(0.5),
                "60%": df[var].quantile(0.6),
                "80%": df[var].quantile(0.8),
                "90%": df[var].quantile(0.9),
                "99%": df[var].quantile(0.99),
                "max": df[var].max(),
                "mean": df[var].mean()
            }, orient='index').T
    
    def get_var_distr(self):
        """
        样本的频数分布：用时间对该变量进行分箱，统计每一箱内的样本频数占比（去掉空值样本）
        """
        print("get var distr".center(80, '*'))
        # 去掉空值
        df_copy = self.__df[(~self.__df[self.__var].isin(["NULL", "null", "", np.nan])) & (self.__df[self.__var].notnull())][[self.__var, self.__label,self.__split_col]]
        df_distr = df_copy.groupby(self.__split_col).apply(lambda x: x.shape[0]*1.0 / (df_copy.shape[0]+1e-20))
        return df_distr 
        
    def get_var_quantile(self):
        """
        变量的分位数：用时间对该变量进行分箱，统计每一箱内该特征的各个分位数，观察其跨时间稳定性（去掉空值样本）
        """
        print("get var quantile".center(80, '*'))
        # 去掉空值
        df_copy = self.__df[(~self.__df[self.__var].isin(["NULL", "null", "", np.nan])) & (self.__df[self.__var].notnull())][[self.__var, self.__label,self.__split_col]]
        df_quantile = df_copy.groupby(self.__split_col).apply(lambda x: self.__cal_quantile(x,self.__var)).reset_index(level=1, drop=True)
        return df_quantile

    def get_var_psi(self, bins_num=5):
        """
        变量psi：用时间对该变量进行分箱，以第1箱为base集，后面每一箱分别作为test集，计算psi（去掉空值样本，并注意每一箱内的样本量！）
        """
        print("get var psi".center(80, '*')) 
        # 一定要copy，不能在原始的上面进行删改！！！！！
        df_copy = self.__df[[self.__var, self.__label,self.__split_col]]
        print('df_ori.shape = ', df_copy.shape)
        # 将第一箱，作为base集 
        df_base = df_copy[df_copy[self.__split_col] == sorted(df_copy[self.__split_col].unique(), reverse=True)[0]]
        print('df_base.shape = ', df_base.shape)      
        # 对base去掉空值之后，进行等频分箱，得到分箱区间
        df_base = df_base[~df_base[self.__var].isin(["NULL", "null", "", np.nan]) & (df_base[self.__var].notnull())] 
        print('去掉空值后 df_base.shape = ', df_base.shape)
        # 得到分箱区间
        bins_list = []
        for i in np.arange(0, 1.1, 1.0/bins_num): # 
            bins_list.append(df_base[self.__var].quantile(q=i))
        bins_list = sorted(list(set(bins_list)))  # 注意去掉相同的边界！！！
        df_base["bins"] = pd.cut(df_base[self.__var], bins=bins_list, duplicates='drop') # 注意去掉相同的边界！！！
        # 计算各分箱内占比
        df_base_result = df_base.groupby("bins").apply(lambda x: x.shape[0]*1.0 / (df_base.shape[0]+1e-20))
        # print('df_base_result = ', df_base_result)

        # 以每一箱作为test集
        df_psi = pd.DataFrame(columns={"psi"}, index=sorted(df_copy[self.__split_col].unique()) )
        for bins in sorted(df_copy[self.__split_col].unique()) : 
            # print("---bins = %s---" % bins)
            df_test = df_copy[df_copy[self.__split_col] == bins]
            # 去掉空值
            df_test = df_test[~df_test[self.__var].isin(["NULL", "null", "", np.nan]) & (df_test[self.__var].notnull())] 
            print("去掉空值之后df_test.shape = ", df_test.shape)
            # 使用df_base的分箱区间进行分箱
            df_test["bins"] = pd.cut(df_test[self.__var], bins=bins_list, duplicates='drop') # 注意去掉相同的边界！！！
            # 计算各分箱内占比
            df_test_result = df_test.groupby("bins").apply(lambda x: x.shape[0]*1.0 / (df_test.shape[0]+1e-20))           
            # psi          
            df_result = pd.concat([df_base_result, df_test_result], axis=1, keys=["distr_base", "distr_test"])
            df_result['psi_tmp'] = df_result.apply(lambda x: (x['distr_base'] - x['distr_test']) * math.log((x['distr_base']+1e-20) / (x['distr_test']+1e-20)), axis=1)           
            df_psi.loc[bins, 'psi'] = df_result['psi_tmp'].sum()
        return df_psi     

    @staticmethod 
    def __plot(X, Y, name):        
        #         plt.plot(list(range(len(X))), Y, label=label) 
        #         plt.xticks(list(range(len(X))), tuple(X), color='black', rotation=45) # 横坐标旋转60度
        plt.plot(X, Y, label=name) 
        plt.xticks(X, tuple(X), color='black', rotation=30) # 横坐标旋转
        for a, b in zip(X, Y):
            plt.text(a, b, '%s' % (round(b, 3)), ha='center', va='bottom', fontsize=10) # plt.text 在曲线上显示y值
    
    def plot_result(self, df_result):
        print("start plot!")
        plt.figure(figsize=(25, 8))
        plt.tick_params(labelsize=8)  # tick_params可设置坐标轴刻度值属性
        X = [str(x) for x in df_result.index]
        for col,j in zip(["quantile",'distr_psi_coverage','odds_auc_ks'],[1,2,3]):
            ax1 = plt.subplot(1,3,j)
            if col == "quantile":
                for i in ["min", "20%", "40%", "50%", "60%", "80%", "90%", "max", "mean"]:
                    self.__plot(X, df_result[i].tolist(), i)
                    plt.legend()  # 用来显示图例
            elif col == 'odds_auc_ks':
                for i in ['odds','auc','ks']:
                    self.__plot(X, df_result[i].tolist(), i)
                    plt.legend()  # 用来显示图例               
            else:
                for i in ["distr", "psi"]: # 这几个使用同一个Y轴，因为数量级差不多
                    self.__plot(X, df_result[i].tolist(), i)   
                    plt.legend(loc='upper left')  # 用来显示图例
                # coverge使用另一个Y轴
                ax2 = ax1.twinx()
                ax2.bar(X, df_result["coverage"], width=0.2, label="coverage",alpha=0.3)
                plt.xticks(X, tuple(X), color='black', rotation=30) # 横坐标旋转
                for a, b in zip(X, df_result["coverage"]): 
                    plt.text(a, b, '%s' % (round(b, 3)), ha='center', va='bottom', fontsize=10)  # plt.text 在曲线上显示y值    
                plt.legend(loc='upper right')  # 用来显示图例
                
            plt.grid(axis='y', color='#8f8f8f', linestyle='--', linewidth=1)  # 显示网格
            plt.title('%s_%s' % (self.__var, col))
            plt.xlabel("%s" % self.__split_col) 
            plt.ylabel(col) 
        path = os.path.join(self.__save_path, self.__var+'_quantile_coverage_odds_distr_psi_auc.png')
        print("save_path is %s" % path)
        plt.savefig(path)
        plt.close()
        
if __name__ == '__main__':
    save_path = r"./plot_feat"
    
    # load data. 包含label和apply_month
    df_all = pd.read_csv("data.csv")
    # feat_list
    feat_list = [x for x in df_all.columns.tolist() if x not in ["user_id","label","apply_month"]] # 去掉不相关的字段  

    # 找到用来分箱的变量：'split_time'
    df['split_time'] = df['apply_month']
    # 前N箱合并成一个
    N = 2
    time_list = sorted(df['split_time'].unique())
    df['split_time'] = df['split_time'].apply(lambda x: str(time_list[0])+'~'+str(time_list[N-1]) if x in time_list[0:N] else x)
    # 开始计算
    for var in feat_list:
        print(("feature %s" % var).center(80, '-'))
        var_time_seq = VarTimeSeq(var_name=var, df=df, split_col='split_time', save_path=save_path)

