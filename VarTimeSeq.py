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
        1）时序分布稳定性：统计每箱内，分位数+最大最小值+平均值+样本占比+psi（ps:去掉空值计算）+missrate
        2）时序表现稳定性：统计每箱内，auc+ks+样本positive_rate。（ps:不去掉空值计算，因为在实际使用的时候，不会只使用该特征覆盖样本建模）
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
#         self.get_all_result()

        
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
            
    @staticmethod
    def single_train(label_list,var_list):
        """
        用该特征单独训练一个模型，得到pvalue，用来计算AUC/KS。K折交叉赋分，比较准确             
        """
        print("用该特征单独训练一个模型，得到pvalue，用来计算AUC/KS。K折交叉赋分，比较准确!")
        df = pd.DataFrame({'label':label_list, 'var':var_list})
        #df['label'] = df['label'].astype('int64')
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
        print("根据pvalue，计算AUC、KS")
        auc = metrics.roc_auc_score(label_list, prob_list)
        fpr,tpr,thresholds= metrics.roc_curve(label_list, prob_list)
        ks = max(tpr-fpr)
        return pd.DataFrame.from_dict({'auc':auc, 'ks':ks}, orient='index').T
     
    def get_var_missrate(self):
        """
        变量缺失率：用时间对该变量进行分箱，统计每一箱内该变量的缺失率（为空的样本占比）。（ps：注意空值的判断逻辑，可根据实际情况调整）
        """
        print("get var missrate".center(80, '*'))
        null_value_list = ["NULL", "null", "", np.nan]
        df_missrate = self.__df.groupby(self.__split_col).apply(lambda x: x[(x[self.__var].isin(null_value_list)) | (x[self.__var].isnull())].shape[0]*1.0 / (x.shape[0]+1e-20))
        return df_missrate
    
    def get_var_positiveRate(self):
        """
        样本positiveRate：用时间对该变量进行分箱，统计每一箱内的label=1的样本个数/总样本个数（不去掉空值样本）
        """
        print("get var positiveRate".center(80,'*'))
        # df_positiveRate = self.__df.groupby(self.__split_col).apply(lambda x: x[self.__label].sum()*1.0 / (x[x[self.__label] == 0].shape[0]+1e-20))
        df_positiveRate = self.__df.groupby(self.__split_col).apply(lambda x: x[self.__label].sum()*1.0 / (x.shape[0]+1e-20))
        return df_positiveRate
    
    def get_var_auc(self):
        """
        变量的跨月表现AUC/KS：用时间对该变量进行分箱，计算每一箱内特征的AUC/KS（不去掉空值样本）
        """        
        print("get var bins_auc".center(80, '*')) 
        df_copy = self.__df[[self.__var, self.__label,self.__split_col]]
        # df_copy['pvalue'] = self.single_train(label_list=df_copy[self.__label], var_list=df_copy[self.__var]) # 用原值进行训练，然后计算AUC
        df_copy['pvalue'] = df_copy[self.__var].fillna(-1) # 用原值计算AUC
        # print(df_copy[self.__label].isnull().sum())
        df_auc = df_copy.groupby(self.__split_col).apply(lambda x: self.cal_auc(x[self.__label],x['pvalue'])).reset_index(level=1, drop=True)
        return df_auc
    
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
        df_copy = self.__df[(~self.__df[self.__var].isin(["NULL", "null", "", np.nan])) & (self.__df[self.__var].notnull())][[self.__var, self.__split_col]]
        df_quantile = df_copy.groupby(self.__split_col).apply(lambda x: self.__cal_quantile(x,self.__var)).reset_index(level=1, drop=True)
        return df_quantile

    def get_var_psi(self, bins_num=5, reverse_mode=False):
        """
        变量psi：用时间对该变量进行分箱，以第1箱为base集，后面每一箱分别作为test集，计算psi（去掉空值样本，并注意每一箱内的样本量！）
        输入：
        reverse_mode：根据时间进行分箱，选定时间最近的一个分箱还是最远一个分箱作为base集，计算psi。=TRUE，选定最远一个时间分箱作为base集；=FALSE，选定最近一个时间分箱，最为base集
        """
        print("get var psi".center(80, '*')) 
        # 一定要copy，不能在原始的上面进行删改！！！！！
        df_copy = self.__df[[self.__var, self.__split_col]]
        df_copy[self.__split_col] = df_copy[self.__split_col].astype(str)
        print('df_ori.shape = ', df_copy.shape)
        
        # base集 
        df_base = df_copy[df_copy[self.__split_col] == sorted(df_copy[self.__split_col].unique(), reverse=reverse_mode)[0]]
        print('df_base.shape = ', df_base.shape,'df_base time = ', df_base[self.__split_col].unique())      
        # 对base去掉空值
        df_base = df_base[~df_base[self.__var].isin(["NULL", "null", "", np.nan]) & (df_base[self.__var].notnull())] 
        print('去掉空值后 df_base.shape = ', df_base.shape)
        # 再对base进行等频分箱，得到分箱区间
        bins_list = []
        for i in np.arange(0, 1.1, 1.0/bins_num): #
            bins_list.append(df_base[self.__var].quantile(q=i))
        bins_list = sorted(list(set(bins_list)))  # 注意去掉相同的边界！！！
        # 由于边界值需要被包含在内，所以最大最小值分别处理一下
        min = bins_list[0] - 1e-20
        max = bins_list[-1] + 1e-20
        bins_list = [min] + bins_list[1:-1] + [max]
        # 分箱
        df_base["bins"] = pd.cut(df_base[self.__var], bins=bins_list, duplicates='drop') # 注意去掉相同的边界！！！
        # 计算各箱内占比
        df_base_result = df_base.groupby("bins").apply(lambda x: x.shape[0]*1.0 / (df_base.shape[0]+1e-20))
        # 以每一个时间分箱作为test集，计算psi
        df_psi = pd.DataFrame(columns={"psi"}, index=sorted(df_copy[self.__split_col].unique()) )
        for bins in sorted(df_copy[self.__split_col].unique()) : 
            # print("---bins = %s---" % bins)
            df_test = df_copy[df_copy[self.__split_col] == bins]
            # 去掉空值
            df_test = df_test[~df_test[self.__var].isin(["NULL", "null", "", np.nan]) & (df_test[self.__var].notnull())] 
            print("去掉空值之后df_test.shape = ", df_test.shape, '时间范围 = ', bins)
            # 使用df_base的分箱区间进行分箱
            df_test["bins"] = pd.cut(df_test[self.__var], bins=bins_list, duplicates='drop') # 注意去掉相同的边界！！！
            # 计算各分箱内占比
            df_test_result = df_test.groupby("bins").apply(lambda x: x.shape[0]*1.0 / (df_test.shape[0]+1e-20))           
            # psi          
            df_result = pd.concat([df_base_result, df_test_result], axis=1, keys=["distr_base", "distr_test"])
            df_result['psi_tmp'] = df_result.apply(lambda x: (x['distr_base'] - x['distr_test']) * math.log((x['distr_base']+1e-20) / (x['distr_test']+1e-20)), axis=1)           
            df_psi.loc[bins, 'psi'] = df_result['psi_tmp'].sum()
        return df_psi  
    
    def get_all_result(self):
        df_missrate = self.get_var_missrate()
        df_quantile = self.get_var_quantile()
        df_positiveRate = self.get_var_positiveRate()
        df_distr = self.get_var_distr()
        df_psi = self.get_var_psi()
        df_auc = self.get_var_auc()
        df_result = pd.concat([df_quantile, df_missrate, df_positiveRate, df_distr, df_psi, df_auc], axis=1, join='outer')  # 按照列进行拼接
        df_result.rename(columns={0:"missrate", 1:"positiveRate", 2:"distr", 3:"psi",4:"auc"}, inplace=True)
        df_result.to_csv(self.__save_path + '/%s_quantile_missrate_positiveRate_distr_psi_auc.csv' % self.__var, index=True,encoding="utf-8")
        print(self.__save_path + '/%s_quantile_missrate_positiveRate_distr_psi_auc.csv' % self.__var)
        self.plot_all(df_result)
        
    @staticmethod 
    def __plot(X, Y, name):        
        #         plt.plot(list(range(len(X))), Y, label=label) 
        #         plt.xticks(list(range(len(X))), tuple(X), color='black', rotation=45) # 横坐标旋转60度
        plt.plot(X, Y, label=name) 
        plt.xticks(X, tuple(X), color='black', rotation=90) 
        for a, b in zip(X, Y):
            plt.text(a, b, '%s' % (round(b, 3)), ha='center', va='bottom', fontsize=10) # plt.text 在曲线上显示y值 
        plt.legend(loc='upper left')  # 用来显示图例
    
    def plot_all(self, df_result):
        print("start plot!")
        plt.figure(figsize=(30, 8))
        plt.tick_params(labelsize=8)  # tick_params可设置坐标轴刻度值属性
        X = [str(x) for x in df_result.index] 
        need_to_plot = ["quantile",'distr_psi_missrate','positiveRate_auc_ks']
        for col,j in zip(need_to_plot, list(np.arange(1,len(need_to_plot)+1,1))):
            ax1 = plt.subplot(1,len(need_to_plot),j) # 分画布
            if col == "quantile": 
                for i in ["20%", "40%", "50%", "60%", "80%", "90%", "mean"]: # 太大或者太小的值都不要画出来，因为有可能存在异常值
                    self.__plot(X, df_result[i].tolist(), i)
            elif col == 'positiveRate_auc_ks':
                for i in col.split("_"):
                    self.__plot(X, df_result[i].tolist(), i)              
            elif col == 'distr_psi_missrate':
                for i in col.split("_")[:2]: # 这几个使用同一个Y轴，因为数量级差不多
                    self.__plot(X, df_result[i].tolist(), i)   
                # missrate使用另一个Y轴
                ax2 = ax1.twinx()
                ax2.bar(X, df_result["missrate"], width=0.2, label="missrate",alpha=0.3)
                plt.xticks(X, tuple(X), color='black', rotation=30) # 横坐标旋转
                for a, b in zip(X, df_result["missrate"]): 
                    plt.text(a, b, '%s' % (round(b, 3)), ha='center', va='bottom', fontsize=10)  # plt.text 在曲线上显示y值    
                plt.legend(loc='upper right')  # 用来显示图例
            plt.grid(axis='y', color='#8f8f8f', linestyle='--', linewidth=1)  # 显示网格
            plt.grid(axis='x', color='#8f8f8f', linestyle='--', linewidth=1)  # 显示网格
            plt.title('%s_%s' % (self.__var, col))
            plt.xlabel("%s" % self.__split_col) 
            plt.ylabel(col) 
        path = os.path.join(self.__save_path, self.__var+'_quantile_missrate_positiveRate_distr_psi_auc.png')
        print("save_path is %s" % path)
        plt.savefig(path)
        plt.close()
        
    def plot_quantile_psi_missrate(self, df_quantile, df_psi, df_missrate):
        df_result = pd.concat([df_quantile, df_psi, df_missrate], axis=1, join='outer')  # 按照列进行拼接
        df_result.rename(columns={0:"missrate"}, inplace=True)
        df_result.to_csv(self.__save_path + '/%s_quantile_psi_missrate.csv' % self.__var, index=True,encoding="utf-8")
        print("start plot!")
        plt.figure(figsize=(30, 8))
        plt.tick_params(labelsize=8)  # tick_params可设置坐标轴刻度值属性
        X = [str(x) for x in df_result.index]
        need_to_plot = ["quantile",'psi',"missrate"]
        for col,j in zip(need_to_plot, list(np.arange(1,len(need_to_plot)+1,1))):
            ax1 = plt.subplot(1,len(need_to_plot),j) # 分画布
            if col == "quantile": 
                for i in ["20%", "40%", "50%", "60%", "80%", "90%", "mean"]: # 太大或者太小的值都不要画出来，因为有可能存在异常值
                    self.__plot(X, df_result[i].tolist(), i)
            else:
                self.__plot(X, df_result[col].tolist(), col)   
            plt.grid(axis='y', color='#8f8f8f', linestyle='--', linewidth=1)  # 显示网格
            plt.grid(axis='x', color='#8f8f8f', linestyle='--', linewidth=1)  # 显示网格
            plt.title("%s_%s" % (self.__var,col))
            plt.xlabel("%s" % self.__split_col) 
            plt.ylabel("%s_%s" % (self.__var,col))
        path = os.path.join(self.__save_path, self.__var+'_quantile_psi_missrate.png')
        print("save_path is %s" % path)
        plt.savefig(path)
        plt.show()
        plt.close()
        
        
if __name__ == '__main__':
    # 将制指定值，变成空
    # df[feat_list] = df[feat_list].applymap(lambda x: np.nan if x in [-1,-2,-3,'-1','-2','-3'] else x)   

    # 按照时间月/周/日分箱   
    df['split_time'] = df[split_col].apply(lambda x: str(x)[0:7])
    print(sorted(df['split_time'].unique()))
    #df['split_time'] = df['split_time'].apply(lambda x: '2019-12-0' if x in [ '2019-12-1', '2019-12-2','2019-12-3'] else x)
    #df['split_time'] = df.apply(lambda x: x[split_time] if x[split_time] >= '2020-04-01' else x['split_time'], axis=1)
    
    # 如果某些月份样本过少，可以跟相邻月份合并在一起
    N = 1
    time_list = sorted(df['split_time'].unique(), reverse=True)
    df['split_time'] = df['split_time'].apply(lambda x: str(time_list[0])+'~'+str(time_list[N-1]) if x in time_list[0:N] else x)
    
    for var in feat_list:
        print(("feature %s" % var).center(80, '-'))
        var_time_seq = VarTimeSeq(df=df,
                                  var=var,
                                  label='label',
                                  split_col='split_time',
                                  save_path='/result/')
    df_quantile = var_time_seq.get_var_quantile()
    df_psi = var_time_seq.get_var_psi(bins_num=5, reverse_mode=False) # reverse_mode=False表示计算psi的时候，按照时间正序排列，以第一个月份为base集
    df_missrate = var_time_seq.get_var_missrate()
    var_time_seq.plot_quantile_psi_missrate(df_quantile, df_psi, df_missrate)
      
