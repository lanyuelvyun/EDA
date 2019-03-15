# -*- coding:utf-8 -*-
"""
@author: lisiqi
@time: 2019/03/12 19:30
"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.dpi'] = 150
from VariableDistribution import VariableDistribution
import EDA

# 加载数据集
df1 = pd.read_csv(r"source/set1.csv")
df2 = pd.read_csv(r"source/set2.csv")
var_list = df1.columns.tolist()
var_list = [x for x in var_list if x not in ["apply_time"]]

# 设置参数
# 变量拥有的唯一值的阈值：大于该阈值，当做连续变量处理，否则当做离散变量处理
var_threshold = 50  
bins = 30  #分箱个数
binning_mode = "cut"  #分箱模式："cut"等宽，"qcut"等频
save_path = r"result"
for var in var_list:  # 针对每一个变量
    # 去掉空值
    df_sub1 = df1[df1[var].notnull()]
    df_sub2 = df2[df2[var].notnull()]
    VariableDistribution(var_name=var,
                         value_list1=df_sub1[var].tolist(), label_list1=df_sub1["label"].tolist(),
                         value_list2=df_sub2[var].tolist(), label_list2=df_sub2["label"].tolist(),
                         var_threshold=var_threshold, bins=bins, binning_mode="cut", save_path=save_path)


