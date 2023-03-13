# -*- coding:utf-8 -*-
"""
@author: lanyue
@startTime: 2019/09/17 20:00
@updateTime: 2023/03/13 16:10
"""
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
    def __init__(self, var_name, value_list, label_list, time_list, cnt_threshold, bins_mode, bins_num, cutoff_list, save_path):
        """
        @目的：针对2分类任务的变量【时序性】分析，下面几张图要结合在一起看，单独看不全面！用时间进行分箱，粒度可以是月/周/天
        1）时序分布稳定性：统计每箱内，分位数+最大最小值+平均值+样本占比+psi（ps:这些统计需要去掉空值）+missrate
        2）时序表现稳定性：统计每箱内，auc+ks+样本positive_rate（ps:不去掉空值计算，因为在实际使用的时候，不会只使用该特征覆盖样本建模）
        """
        if len(value_list) != len(label_list):
            raise ValueError("len(value) and len(label) is not equal, please check!!!")
        self.var_name = var_name # 变量名称
        self.value_list = value_list # 变量的值，示例[1,4,2]
        self.label_list = label_list # 二分类label_list, 与value_list一一对应，示例[1,1,0]
        self.time_list = time_list # 时间列，与value_list一一对应，示例['2023-01','2023-02','2023-03']
        self.cnt_threshold = cnt_threshold # 变量唯一值个数的阈值。该变量的唯一值个数>cnt_threshold，被认为是连续变量，否则离散变量
        self.bins_mode = bins_mode # 分箱方法，='cut'代表等宽分箱，='qcut'代表等频分箱。
        self.bins_num = bins_num # 分箱个数。搭配cnt_threshold+bins_mode一起使用。计算IV值的时候会用到
        self.cutoff_list = cutoff_list # 自定义分箱的切分点cutoff_list，如果该值不为空，会优先使用该分箱方法，此时参数bins_num+bing_mode失效
        self.save_path = save_path # 结果保存路径

        self.df = pd.DataFrame({self.var_name: self.value_list, "label": self.label_list, 'time_bins':self.time_list})
        self.df['time_bins'] = self.df['time_bins'].astype(str)
        # 去掉空值之后的数据（ps：注意空值的判断逻辑，可根据实际情况调整）
        self.null_value_list = ["NULL", "null", ""]
        self.df_notnull = self.df[(self.df[self.var_name].notnull()) & (~self.df[self.var_name].isin(self.null_value_list))] 

     
    
    # @staticmethod
    # def cal_ks_auc(label_list, prob_list):
    #     print("计算特征的AUC、KS（PS：该特征值的取值范围只能是0~1之间！！）...")
    #     auc = metrics.metrics.roc_auc_score(label_list, prob_list)
    #     fpr,tpr,thresholds= metrics.roc_curve(label_list, prob_list)
    #     ks = max(tpr-fpr)
    #     return pd.DataFrame.from_dict({'auc':auc, 'ks':ks}, orient='index').T
        
    @staticmethod   
    def ks_stat(label_list, value_list):
        # 100等频分箱计算ks
        df = pd.DataFrame({'X': value_list, 'Y': label_list})
        floor = df.X.min()
        ceiling = df.X.max()
        step = -(ceiling - floor) / 100
        cut_list = np.arange(ceiling, floor, step)
        ks_thgrouphold = 0
        max_distance = 0
        for cut in cut_list:
            tp = max(0.00001, df.loc[(df.X >= cut) & (df.Y == 1), :].shape[0])
            fp = max(0.00001, df.loc[(df.X >= cut) & (df.Y == 0), :].shape[0])
            fn = max(0.00001, df.loc[(df.X < cut) & (df.Y == 1), :].shape[0])
            tn = max(0.00001, df.loc[(df.X < cut) & (df.Y == 0), :].shape[0])

            tpr = round(float(tp) / (tp + fn), 4)
            fpr = round(float(fp) / (fp + tn), 4)
            distance = abs(tpr - fpr)

            if distance > max_distance:
                max_distance = distance
                ks_thgrouphold = cut
        return round(max_distance, 3), ks_thgrouphold

     
    def cal_ks_auc(self, label_list, value_list, fix_auc=True):
        # 计算特征的AUC、KS
        f = [float(item) for item in value_list]
        label_cnt = len(set(label_list))
        value_cnt = len(set(value_list))
        if label_cnt != 2:
            print(f'label distinct is {label_cnt}!!!') 
            return pd.DataFrame.from_dict({'auc':-1, 'ks':-1}, orient='index').T
        elif value_cnt == 1:
            print(f'value distinct is {value_cnt}!!!')
            return pd.DataFrame.from_dict({'auc':-1, 'ks':-1}, orient='index').T
        ks, ksth = self.ks_stat(label_list, f)
        auc = metrics.roc_auc_score(label_list, f)
        if fix_auc:
            auc = max(auc, 1-auc)
        return pd.DataFrame.from_dict({'auc':auc, 'ks':ks}, orient='index').T
     
    def get_bins_auc(self):
        """
        变量的跨月表现AUC/KS：用时间对该变量进行分箱，计算每箱内特征的AUC/KS
        """        
        print("--> get var bins_auc...") 
        # 新方法计算AUC/KS(ps: 传入的特征值的取值范围,没必要必须是0到1之间)
        df_auc = self.df.groupby('time_bins').apply(lambda x: self.cal_ks_auc(x['label'], x[self.var_name])).reset_index(level=1, drop=True)
        return df_auc    
      

        
    def get_bins_missrate(self):
        """
        变量缺失率：用时间对该变量进行分箱，统计每箱内该变量的缺失率
        """
        print("--> get var missrate...")
        df_missrate = self.df.groupby('time_bins').apply(lambda x: ((x[self.var_name].isin(self.null_value_list)) | (x[self.var_name].isnull())).sum()*1.0 / (x.shape[0]+1e-20))
        return df_missrate
    
    

    def get_bins_positiveRate(self):
        """
        样本positiveRate：用时间对该变量进行分箱，统计每箱内的label=1的样本个数/总样本个数
        """
        print("--> get var positiveRate...")
        df_positiveRate = self.df.groupby('time_bins').apply(lambda x: x['label'].sum()*1.0 / (x.shape[0]+1e-20))
        return df_positiveRate
    


    
    def get_bins_distr(self):
        """
        样本的频数分布：用时间对该变量进行分箱，统计每箱内的样本频数占比
        """
        print("--> get var cnt...")
        df_distr = self.df.groupby('time_bins').apply(lambda x: x.shape[0]*1.0)
        return df_distr 
      

        

    @staticmethod    
    def cal_quantile(df, var):
    # 计算分位数
        return pd.DataFrame.from_dict(
            {
                "min": df[var].min(),
                "10%": df[var].quantile(0.1),
                "20%": df[var].quantile(0.2),
                "30%": df[var].quantile(0.3),
                "40%": df[var].quantile(0.4),
                "50%": df[var].quantile(0.5),
                "60%": df[var].quantile(0.6),
                "70%": df[var].quantile(0.7),
                "80%": df[var].quantile(0.8),
                "90%": df[var].quantile(0.9),
                "99%": df[var].quantile(0.95),
                "99%": df[var].quantile(0.99),
                "max": df[var].max(),
                "mean": df[var].mean()
            }, orient='index').T 

    def get_bins_quantile(self):
        """
        变量的分位数：用时间对该变量进行分箱，统计每箱内该特征的各个分位数，观察其跨时间稳定性（去掉空值样本!）
        """
        print("--> get var quantile(在非空样本上)...")
        # 在非空样本上统计
        df_quantile = self.df_notnull.groupby('time_bins').apply(lambda x: self.cal_quantile(x, self.var_name)).reset_index(level=1, drop=True)
        return df_quantile
    
 


    def binning_var(self, df):
        """
        对变量进行分箱(PS：用于计算psi)
        1、自定义分箱：传入自定义cutoff_list
        2、自动分箱：区分连续变量和离散变量：如果该变量唯一值的个数>=self.cnt_threshold，认为是连续变量，否则离散变量；
        1）连续变量：等宽/等频分箱；
        2）离散：每个值单独分一箱；
        3）空值单独分一箱;
        """        
        null_value_list = ["NULL", "null", ""]
        df_notnull = df[(~df[self.var_name].isin(null_value_list)) & (df[self.var_name].notnull())]
        unique_value_cnt = len(df[self.var_name].unique())

        # 确定分箱区间
        if self.cutoff_list is not None: # 如果传入了自定义的分箱cutoff_list
            print(f"自定义分箱(左开右闭)...")
            cutoff_list = sorted(set(self.cutoff_list)) # 去重排序
            #cutoff_list[0] = cutoff_list[0] - 1e-20 # 最小值额外减去一个极小数，为了能包含原bins的最小值
        elif unique_value_cnt < self.cnt_threshold:
            print(f"类别变量【该变量的唯一值个数({unique_value_cnt}) < 阈值({self.cnt_threshold})】。每个值单独分一箱...")
            cutoff_list = sorted(df_notnull[self.var_name].unique()) # 去掉空值
            cutoff_list = [cutoff_list[0] - 1e-20] + cutoff_list # 左侧额外加一个值，为了能包含原bins的最小值
        elif unique_value_cnt >= self.cnt_threshold and self.bins_mode == 'cut': 
            print(f"连续变量【该变量的唯一值个数({unique_value_cnt}) >= 阈值({self.cnt_threshold})】。等宽分箱(左开右闭)...")
            min_value = df_notnull[self.var_name].min()
            max_value = df_notnull[self.var_name].max()
            step = (max_value - min_value)*1.0 / self.bins_num # 去掉空值求等宽cutoff
            cutoff_list = list(np.arange(min_value, max_value+step, step))
            cutoff_list = sorted(set(cutoff_list)) # 去重排序
            cutoff_list[0] = cutoff_list[0] - 1e-20 # 最小值额外减去一个极小数，为了能包含原bins的最小值
        elif unique_value_cnt >= self.cnt_threshold and self.bins_mode == 'qcut':  
            print(f"连续变量【该变量的唯一值个数({unique_value_cnt}) >= 阈值({self.cnt_threshold})】。等频分箱(左开右闭)...")
            step = 1.0 / self.bins_num
            cutoff_list = [df_notnull[self.var_name].quantile(i) for i in np.arange(0, 1+step, step)] # 去掉空值求等频cutoff
            cutoff_list = sorted(set(cutoff_list))
            cutoff_list[0] = cutoff_list[0] - 1e-20 
               
        # 分箱 
        df["bins"], cutoff_list = pd.cut(df[self.var_name], bins=cutoff_list, 
                                include_lowest=False, right=True, # 左开右闭 
                                retbins=True, precision=3)                
        # 将空值单独分一箱（Category数据，要想插入一个之前没有的值，首先需要将这个值添加到.categories的容器中，然后再添加值。）
        df['bins'] = df['bins'].cat.add_categories(['NAN'])
        df['bins'] = df['bins'].fillna('NAN')
        # df['bins'].cat.categories
        print(f"cutoff_list = {cutoff_list}")
        return df, cutoff_list  

    def get_bins_psi(self, reverse_mode=False):
        """
        计算变量psi：用时间对该变量进行分箱，计算psi（计算每月psi的时候还要分箱，空值单独分一箱。注意每一箱内的样本量！）
        reverse_mode：=TRUE，选定最远一个时间分箱作为base集；=FALSE，选定最近一个时间分箱，作为base集。其余作为test集。
        """
        print("--> get var psi(计算psi的时候，空值单独分一箱)...")     
        # base集 
        base_time_bin = sorted(self.df['time_bins'].unique(), reverse=reverse_mode)[0]
        df_base = self.df[self.df['time_bins'] == base_time_bin]
        print('df_base.shape = ', df_base.shape, '时间范围 = ', df_base['time_bins'].unique())     
        # 分箱，得到cutoff_list        
        df_base, cutoff_list = self.binning_var(df=df_base)
        df_base_result = df_base.groupby("bins").apply(lambda x: x.shape[0]*1.0 / (df_base.shape[0]+1e-20)) # 计算各箱内占比
        
        # test集
        df_psi = pd.DataFrame(columns={"psi"}, index=sorted(self.df['time_bins'].unique()) )
        df_psi.index.name = 'bins'
        # 使用df_base的cutoff_list进行分箱
        self.cutoff_list = cutoff_list  
        for bins in sorted(self.df['time_bins'].unique()) : 
            # print("---bins = %s---" % bins)
            df_test = self.df[self.df['time_bins'] == bins]           
            print("df_test.shape = ", df_test.shape, '时间范围 = ', bins)
            # 使用df_base的cutoff_list进行分箱
            df_test, test_cutoff_list = self.binning_var(df=df_test)
            df_test_result = df_test.groupby("bins").apply(lambda x: x.shape[0]*1.0 / (df_test.shape[0]+1e-20)) # 计算各分箱内占比
            # psi          
            df_result = pd.concat([df_base_result, df_test_result], axis=1, keys=["distr_base", "distr_test"])
            df_result['psi_tmp'] = df_result.apply(lambda x: (x['distr_base'] - x['distr_test']) * math.log((x['distr_base']+1e-20) / (x['distr_test']+1e-20)), axis=1)           
            df_psi.loc[bins, 'psi'] = df_result['psi_tmp'].sum()
            # print(df_result)
        return df_psi  




        
    @staticmethod 
    def plot(X, Y, name):        
        #         plt.plot(list(range(len(X))), Y, label=label) 
        #         plt.xticks(list(range(len(X))), tuple(X), color='black', rotation=45) # 横坐标旋转60度
        plt.plot(X, Y, label=name) 
        plt.xticks(X, tuple(X), color='black', rotation=90) 
        for a, b in zip(X, Y):
            plt.text(a, b, '%s' % (round(b, 3)), ha='center', va='bottom', fontsize=10) # plt.text 在曲线上显示y值 
        plt.legend(loc='upper left')  # 用来显示图例
    
    def plot_distribution_performance(self):
        print("--> plot distribution_erformance_byTime...")
        df_missrate = self.get_bins_missrate()
        df_quantile = self.get_bins_quantile()
        df_positiveRate = self.get_bins_positiveRate()
        df_distr = self.get_bins_distr()
        df_psi = self.get_bins_psi()
        df_auc = self.get_bins_auc()
        df_result = pd.concat([df_quantile, df_missrate, df_positiveRate, df_distr, df_psi, df_auc], axis=1, join='outer')  # 按照列进行拼接
        df_result.rename(columns={0:"missrate", 1:"positiveRate", 2:"cnt", 3:"psi", 4:"auc"}, inplace=True)
        df_result.to_csv(self.save_path + f"/{self.var_name}_distribution_performance_byTime.csv", index=True, encoding="utf-8")
        print(self.save_path + f"/{self.var_name}_distribution_performance_byTime.csv")

        # 画图
        plt.figure(figsize=(30, 8))
        plt.tick_params(labelsize=10)  # tick_params可设置坐标轴刻度值属性
        X = [str(x) for x in df_result.index] 
        need_to_plot = ["quantile", 'cnt', 'psi_missrate', 'positiveRate_auc_ks']
        for col,j in zip(need_to_plot, list(np.arange(1,len(need_to_plot)+1,1))):
            ax1 = plt.subplot(1,len(need_to_plot),j) # 分画布
            if col == "quantile": 
                for i in ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "mean"]: # 太大或者太小的值都不要画出来，因为有可能存在异常值
                    self.plot(X, df_result[i].tolist(), i)
            elif col == 'cnt':
                self.plot(X, df_result[col].tolist(), i)   
            elif col == 'psi_missrate':
                for i in col.split("_"):
                    self.plot(X, df_result[i].tolist(), i)   
                # missrate使用另一个Y轴
                #ax2 = ax1.twinx()
                #ax2.bar(X, df_result["missrate"], width=0.2, label="missrate",alpha=0.3)
            elif col == 'positiveRate_auc_ks':
                for i in col.split("_"):
                    self.plot(X, df_result[i].tolist(), i) 
            plt.grid(axis='y', color='#8f8f8f', linestyle='--', linewidth=1)  # 显示网格
            plt.grid(axis='x', color='#8f8f8f', linestyle='--', linewidth=1)  # 显示网格
            plt.title(col)
            plt.xlabel('time_bins') 
            # plt.ylabel(col) 
        plt.suptitle(f"{self.var_name}: distribution_performance_byTime", fontsize=15) # 画布总标题
        path = os.path.join(self.save_path, self.var_name+'_distribution_performance_byTime.png')
        print("save_path is %s" % path)
        plt.savefig(path)
        plt.show()
        plt.close()
        
    def plot_distribution(self):
        print("--> plot distribution_byTime...")
        df_missrate = self.get_bins_missrate()
        df_quantile = self.get_bins_quantile()
        df_distr = self.get_bins_distr()
        df_psi = self.get_bins_psi()
        df_result = pd.concat([df_quantile, df_missrate, df_distr, df_psi], axis=1, join='outer')  # 按照列进行拼接
        df_result.rename(columns={0:"missrate", 1:"cnt", 2:"psi"}, inplace=True)
        df_result.to_csv(self.save_path + f"/{self.var_name}_distribution_byTime.csv", index=True, encoding="utf-8")
        print(self.save_path + f"/{self.var_name}_distribution_byTime.csv")
        
        # 画图
        plt.figure(figsize=(30, 8))
        plt.tick_params(labelsize=10)  # tick_params可设置坐标轴刻度值属性
        X = [str(x) for x in df_result.index]
        need_to_plot = ["quantile", 'psi', "missrate", 'cnt']
        for col, j in zip(need_to_plot, list(np.arange(1,len(need_to_plot)+1, 1))):
            ax1 = plt.subplot(1,len(need_to_plot), j) # 分画布
            if col == "quantile": 
                for i in ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "mean"]: # 太大或者太小的值都不要画出来，因为有可能存在异常值
                    self.plot(X, df_result[i].tolist(), i)
            else:
                self.plot(X, df_result[col].tolist(), col)   
            plt.grid(axis='y', color='#8f8f8f', linestyle='--', linewidth=1)  # 显示网格
            plt.grid(axis='x', color='#8f8f8f', linestyle='--', linewidth=1)  # 显示网格
            plt.title(col)
            plt.xlabel('time_bins') 
            # plt.ylabel(col)
        plt.suptitle(f"{self.var_name}: distribution_byTime", fontsize=15) # 画布总标题
        path = self.save_path + f'{self.var_name}_distribution_byTime.png'
        print("save_path is %s" % path)
        plt.savefig(path)
        plt.show()
        plt.close()
        
        
if __name__ == '__main__':
    # 引入自定义包
    import sys
    sys.path.append('/data/users/lisiqi/')
    from src import VarTimeSeq

    import importlib
    importlib.reload(VarTimeSeq)

    df = df_ori
    feat_list = ['score']

    # 将制指定值，变成空
    # df[feat_list] = df[feat_list].applymap(lambda x: np.nan if x in [-1,-2,-3,'-1','-2','-3'] else x)   

    # 按照时间月/周/日分箱   
    df['split_time'] = df['sample_time_x'].apply(lambda x: str(x)[0:7])
    print(sorted(df['split_time'].unique()))

    # 保存整体psi值
    df_psi = pd.DataFrame({'var_name': feat_list})
    for var in feat_list:
        # if len(df[var].unique())>=3: # 除去这种类型的特征：[1，1，1，nan,1,1] 和[1,1,1,1,1]
        print((" var = %s" % var).center(80, '*'))
        var_time_seq = VarTimeSeq.VarTimeSeq(var_name=var,
                                            value_list=df[var],
                                            label_list=df['d5_y'],
                                            time_list=df['split_time'],
                                            cnt_threshold=30, 
                                            bins_mode='qcut', 
                                            bins_num=10, 
                                            cutoff_list=None, 
                                            save_path='/data/users/lisiqi/tmp/')
        # var_time_seq.plot_distribution() # 只画分布
        var_time_seq.plot_distribution_performance() # 画分布+表现
    #     df_result = var_time_seq.get_bins_psi(reverse_mode=True)
    #     df_result.to_csv(save_path + '%s_psi.csv' % var, index=True,encoding="utf-8")
    #     df_psi.loc[df_psi['var_name']==var, 'psi'] = df_result.loc['OOT[2022-01-01, 2022-06-30]','psi']
    # df_psi.sort_values(by='psi', ascending=False, na_position='last').to_csv(save_path + r'psi.csv', index=None)
