# -*- coding:utf-8 -*-
"""
@author: lanyue
@startTime: 2019/05/10 17:20
@updateTime: 2023/03/09 11:36
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体为黑体
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
    def __init__(self, var_name, value_list, label_list, cnt_threshold, bins_mode, bins_num, bins_list, save_path):
        """
        @目的：对2分类任务的变量，进行分箱分析
        1）对变量进行分箱，分箱方式支持：等宽、等频、自定义分箱；
        2）计算各分箱内
        样本频数占总体样本的占比：查看变量值的分布情况
        woe值：查看变量对label的区分能力
        正样本占比：查看label情况
        """
        if len(value_list) != len(label_list):
            raise ValueError("len(value) and len(label) is not equal, please check!!!")
        self.var_name = var_name # 变量名称
        self.value_list = value_list # 变量的值
        self.label_list = label_list # 二分类label_list, 与value_list是一一对应的，示例[1,1,0,0,1,0,0]
        self.cnt_threshold = cnt_threshold # 变量唯一值个数的阈值。该变量的唯一值个数>cnt_threshold，被认为是连续变量，否则被认为是离散变量
        self.bins_mode = bins_mode # 分箱方法，其值有：'cut'代表等宽分箱；'qcut'代表等频分箱；
        self.bins_num = bins_num # 分箱个数。搭配bins_mode一起使用
        self.bins_list = bins_list # 自定义分箱cutoff_list，跟参数(bins_mode+bins_num)的组合不能同时使用，只能用其一。如果该值不为空，会优先使用该分箱方法
        self.save_path = save_path

        
    def get_result(self):
        df_distr, bins_list = self.get_var_distr()
        df_rate, bins_list = self.get_var_positiveRate()
        df_woe, bins_list = self.get_var_woe()
        # 合并
        df_result = df_distr.merge(df_rate, on='bins', how="outer").merge(df_woe, on='bins', how="outer")
        #df_result.to_csv(self.save_path + '/%s_distrWoePositiveRate_bins.csv' % self.var_name, index=None)    
        return df_result, bins_list
        

    def divide_var(self):
        """
        对变量进行分箱：
        1、区分连续变量和离散变量：如果该变量唯一值的个数>=self.cnt_threshold，认为是连续变量，否则离散变量；
        2、连续变量：等宽+等频分箱；
        3、离散：每个值单独分一箱；
        """        
        df_var = pd.DataFrame({self.var_name: self.value_list, "label": self.label_list})
        unique_value_cnt = len(df_var[self.var_name].unique())
        df_var_notnull = df_var[df_var[self.var_name].notnull()] # 去掉空值之后的数据

        # 确定分箱区间
        if self.bins_list is not None: # 如果传入了自定义的分箱cutoff_list
            print(f"变量正在进行自定义分箱...")
            bins_list = sorted(set(self.bins_list)) # 去重排序
            #bins_list[0] = bins_list[0] - 1e-20 # 最小值额外减去一个极小数，为了能包含原bins的最小值
        elif unique_value_cnt < self.cnt_threshold:
            print(f"该变量的唯一值个数({unique_value_cnt}) < 阈值({self.cnt_threshold})，被认为是类别变量。每个值单独分一箱...")
            bins_list = sorted(df_var_notnull[self.var_name].unique()) # 去掉空值
            bins_list = [bins_list[0] - 1e-20] + bins_list # 左侧额外加一个值，为了能包含原bins的最小值
        elif unique_value_cnt >= self.cnt_threshold and self.bins_mode == 'cut': 
            print(f"该变量的唯一值个数({unique_value_cnt}) >= 阈值({self.cnt_threshold})，被认为是连续变量。现在进行的是等宽分箱...")
            min_value = df_var_notnull[self.var_name].min()
            max_value = df_var_notnull[self.var_name].max()
            step = (max_value - min_value)*1.0 / self.bins_num # 去掉空值求等宽cutoff
            bins_list = list(np.arange(min_value, max_value+step, step))
            bins_list = sorted(set(bins_list)) # 去重排序
            bins_list[0] = bins_list[0] - 1e-20 # 最小值额外减去一个极小数，为了能包含原bins的最小值
        elif unique_value_cnt >= self.cnt_threshold and self.bins_mode == 'qcut':  
            print(f"该变量的唯一值个数({unique_value_cnt}) >= 阈值({self.cnt_threshold})，被认为是连续变量。现在进行的是等频分箱...")
            step = 1.0 / self.bins_num
            bins_list = [df_var_notnull[self.var_name].quantile(i) for i in np.arange(0, 1+step, step)] # 去掉空值求等频cutoff
            bins_list = sorted(set(bins_list))
            bins_list[0] = bins_list[0] - 1e-20 
               
        # 分箱 
        df_var["bins"], bins_list = pd.cut(df_var[self.var_name], bins=bins_list, 
                                include_lowest=False, right=True, # 左开右闭 
                                retbins=True, precision=3)                
        # 将空值单独分一箱（Category数据，要想插入一个之前没有的值，首先需要将这个值添加到.categories的容器中，然后再添加值。）
        df_var['bins'] = df_var['bins'].cat.add_categories(['NAN'])
        df_var['bins'] = df_var['bins'].fillna('NAN')
        #df_var['bins'].cat.categories
        print(f"bins_list = {bins_list}")
        return df_var, bins_list

    
    def get_var_distr(self):
        """
        频数分布：对该变量进行分箱，统计每一箱内样本个数/总样本个数
        """
        print("-->> get var distr...")
        df_with_bins, bins_list = self.divide_var() # 分箱
        df_distr = df_with_bins.groupby('bins').apply(lambda x: x.shape[0] * 1.0 / (len(self.value_list) + 1e-20)).reset_index()
        df_distr = df_distr.rename(columns={0: 'distr'})
        return df_distr, bins_list


    def get_var_positiveRate(self):
        """
        正样本占比：对该变量进行分箱，统计每一箱内的label=1的样本个数/总样本个数
        """
        print("-->> get var positiveRate...")
        df_with_bins, bins_list = self.divide_var() # 分箱
        df_rate = df_with_bins.groupby('bins').apply(lambda x: x[x["label"] == 1].shape[0]*1.0 / (x.shape[0] + 1e-20)).reset_index()
        df_rate = df_rate.rename(columns={0: 'positiveRate'})
        return df_rate, bins_list


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

    
    def get_var_woe(self):
        """
        woe/iv值：对该变量进行分箱，计算每一箱内的woe/iv
        """
        print("-->> get var woe...")
        df_with_bins, bins_list = self.divide_var() # 分箱
        total_p_cnt = df_with_bins[df_with_bins["label"]==1].shape[0]
        total_n_cnt = df_with_bins[df_with_bins["label"]==0].shape[0]
        df_woe = df_with_bins.groupby('bins').apply(lambda x: self._cal_woe_tmp(x, total_p_cnt, total_n_cnt)).reset_index(level=1, drop=True)
        df_woe["woe"] = np.log(df_woe["p_rate"] * 1.0 / (df_woe["n_rate"] + 1e-20))
        df_woe["iv"] = (df_woe["p_rate"] - df_woe["n_rate"]) * df_woe["woe"]
        df_woe["sum_iv"] = df_woe["iv"].sum()
        df_woe["max_absWoe"] = max(df_woe["woe"].tolist(), key=abs)
        return df_woe, bins_list


    def plot_result(self, df_result_list):
        """
        将分箱内的统计量画成曲线。
        输入是N个样本集的分箱结果，画在一张图上，注意：这N个结果的分箱区间，要一致，才能画在一张图上
        """
        # 合并所有结果
        df_result = df_result_list[0]
        df_result.columns = [x+'_set0' if x!='bins' else x for x in df_result.columns]
        for i in range(0, len(df_result_list), 1)[1:]:
            tmp = df_result_list[i]
            tmp.columns = [x+f'_set{i}' if x!='bins' else x for x in tmp.columns]
            df_result = df_result.merge(tmp, on='bins', how='inner')
        df_result.to_csv(self.save_path + f'/{self.var_name}_distrWoePositiveRate_bins.csv', index=None)  
        
        # 画图
        plt.figure(figsize=(15, 5))
        plt.tick_params(labelsize=5) # tick_params可设置坐标轴刻度值属性
        X = [str(x) for x in df_result['bins']]
        need_to_plot = ["distr","positiveRate","woe"]
        cnt = len(need_to_plot)
        for col, j in zip(need_to_plot, range(1, cnt+1, 1)):
            plt.subplot(1, cnt, j)
            Y_name_list = [x for x in df_result.columns if col in x]
            for Y_name in Y_name_list:
                Y = df_result[Y_name]
                plt.bar(X, Y, label=Y_name.split('_')[1], width=0.5, bottom=0, alpha=0.4)
                plt.plot(X, Y, label=Y_name.split('_')[1])
                # 在曲线上显示Y值
                for a, b in zip(X, Y):
                    plt.text(a, b, '%s' % (round(b, 3)), ha='center', va='bottom', fontsize=10) # plt.text 在曲线上显示y值
            # 横坐标旋转
            plt.xticks(list(range(len(X))), tuple(X), color='black', rotation=90) 
            plt.title(f"{col}")
            plt.xlabel("value_bins")
            # plt.ylabel(col)
            plt.legend()  # 用来显示图例
        plt.suptitle(f"{self.var_name}: distrWoePositiveRate_bins", fontsize=10) # 画布总标题
        save_path = self.save_path + f"/{self.var_name}_distrWoePositiveRate_bins.png"
        print(f"save_path is {save_path}")
        plt.savefig(save_path)
        plt.show()
        plt.close()


if __name__ == '__main__':
    import imp
    import VarDistrOddsWoeIv
    imp.reload(VarDistrOddsWoeIv)

    save_path = r'/data/result/'
    feat_list = ['score']
    
    """单独分析一个样本的特征"""
    # 保存整体iv值
    df_iv = pd.DataFrame(columns={'var_name','iv','max_abs_woe'})
    df_iv['var_name'] = feat_list
    for var in feat_list:
        print((" var = %s " % var).center(80, '-'))
        var_instance = VarDistrOddsWoeIv.VarDistrOddsWoeIv(
            var_name=var,
            value_list=df[var].tolist(),
            label_list=df["d7"].tolist(),
            cnt_threshold=20, # 当unique(特征值)的个数<cnt_threshold，该变量当做类别变量处理，否则当做连续变量处理
            bins_mode='qcut', # 分箱模式
            bins_num=10, # # 分箱个数，与bins_mode搭配使用
            bins_list=None, # 自定义分箱cutoff_list。此参数不为None的时候，参数cnt_threshold、bins_mode、bins_num失效
            save_path=save_path)
        # 所有分箱结果
        df_result, bins_list = var_instance.get_result()
        df_result_list = [df_result]
        # 画图
        var_instance.plot_result(df_result_list) 
        
        # 只需要计算&保存iv值的时候
        df_woe_iv = var_instance.get_var_woe()
        df_woe_iv.to_csv(save_path + r'%s_woe_iv.csv' % var, index=None)
        df_iv.loc[ df_iv['var_name']==var, 'iv'] = df_woe_iv['sum_iv'].unique()
        df_iv.loc[ df_iv['var_name']==var, 'max_abs_woe'] = df_woe_iv['max_abs_woe'].unique()
    df_iv.to_csv(save_path + r'/iv.csv', index=None)
        
        
    """对比2个样本集的特征"""
    for var in feat_list:
        print((" variable = %s " % var).center(80, '-'))
        # 第一个样本集合
        var_instance = VarDistrOddsWoeIv.VarDistrOddsWoeIv(
            var_name=var,
            value_list=df_1[var].tolist(),
            label_list=df_1["d7"].tolist(),
            cnt_threshold=20, # 当unique(特征值)的个数<cnt_threshold，该变量当做类别变量处理，否则当做连续变量处理
            bins_mode='qcut', # 分箱模式
            bins_num=10, # # 分箱个数，与bins_mode搭配使用
            bins_list=None, # 自定义分箱cutoff_list。此参数不为None的时候，参数cnt_threshold、bins_mode、bins_num失效
            save_path=save_path)
        df_result_1, bins_list = var_instance.get_result()

        # 第二个样本集合
        var_instance = VarDistrOddsWoeIv.VarDistrOddsWoeIv(
            var_name=var,
            value_list=df_2[var].tolist(),
            label_list=df_2["d7"].tolist(),
            cnt_threshold=20, 
            bins_mode='qcut', 
            bins_num=5, 
            bins_list=bins_list, # 使用上一个集合相同的分箱cutoff进行分箱（注意：如果第2个样本集有超出第1个样本集范围的特征值，分箱的时候就不会被包含在内）
            save_path=save_path)
        df_result_2, bins_list = var_instance.get_result()   

        # 将多个样本集的结果画在一张图上
        df_result_list = [df_result_1, df_result_2]
        var_instance.plot_result(df_result_list) 

