# -*- coding:utf-8 -*-
"""
@author: lisiqi
@time: 2018/7/31 19:23
@目的：特征工程+建模
"""
import pandas as pd
import random
from scipy import stats
import matplotlib.pyplot as plt
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


# 得到特征的code和describe。（conf文件是一个特征的配置文件，有3列，分别为：特征code，特征英文名，特征中文描述）
def get_varures_list():
    list_name = []
    list_desc = []
    with codecs.open(r"E:\work\2 mission\1 acard\acard_dae\f_conf\acard_zx_dae_v1.conf", 'r', encoding="utf-8") as fr:
        for l in fr.readlines():
            if len(l) > 4 and l[0] != '#':
                ll = l.strip().split('\t')
                list_name.append(ll[0])
                list_desc.append(ll[2])
    return list_name, list_desc


# 计算所有特征的缺失率
def get_var_miss_rate(df):
    print("计算所有特征的缺失率")
    # 在整体样本上的缺失率
    var_miss_rate_all = dict(df.apply(lambda x: x.isnull().sum()*1.0 / x.shape[0]))
    df_var_miss_rate_all = pd.DataFrame({"var_name": list(var_miss_rate_all.keys()), "var_miss_rate_all": list(var_miss_rate_all.values())})
    # 在好样本上的缺失率
    df_good = df[df["label"] == 0]
    var_miss_rate_good = dict(df_good.apply(lambda x: x.isnull().sum()*1.0 / x.shape[0]))
    df_var_miss_rate_good = pd.DataFrame({"var_name": list(var_miss_rate_good.keys()), "var_miss_rate_good": list(var_miss_rate_good.values())})
    # 在坏样本上的缺失率
    df_bad = df[df["label"] == 1]
    var_miss_rate_bad = dict(df_bad.apply(lambda x: x.isnull().sum()*1.0 / x.shape[0]))
    df_var_miss_rate_bad = pd.DataFrame({"var_name": list(var_miss_rate_bad.keys()), "var_miss_rate_bad": list(var_miss_rate_bad.values())})

    df_var_miss_rate = pd.merge(df_var_miss_rate_all, df_var_miss_rate_good, on="var_name", how="left")
    df_var_miss_rate = pd.merge(df_var_miss_rate, df_var_miss_rate_bad, on="var_name", how="left")
    # 按照缺失率降序排列
    df_var_miss_rate = df_var_miss_rate[["var_name", "var_miss_rate_all", "var_miss_rate_good", "var_miss_rate_bad"]]
    df_var_miss_rate = df_var_miss_rate.sort_values(by=["var_miss_rate_all"], ascending=False)
    print("正在保存特征的缺失率，保存路径为：result/var_miss_rate.csv")
    df_var_miss_rate.to_csv(r"result/var_miss_rate.csv", index=False, float_format='%.4f')
    print("get_var_missing_rate finished!")



# 处理缺失值
def missing_value(df):
    print("处理缺失值")
    from sklearn.preprocessing import Imputer
    imputer = Imputer(strategy='median')  # 将missing value用中值填充
    imputer.fit(df)  # Fit on the df
    print("处理缺失值")
    var_array = imputer.transform(df)  # Transform df 返回一个array
    df = pd.DataFrame(var_array, columns=df.columns)  # 变成DataFrame
    return df


# 计算所有特征的方差,并且将方差=0的特征过滤掉
def get_var_std(df, save_path):
    df1 = df.copy()
    print("计算方差")
    df_std = df1.apply(lambda x: x.std())  # 计算每一个特征的方差，返回一个series
    df_std = dict(df_std)
    df_std = pd.DataFrame({"var_name": list(df_std.keys()), "std": list(df_std.values())})
    df_std.sort_values(by="std", ascending=True, inplace=True)
    print("保存特征的方差，保存路径为：", save_path)
    df_std.to_csv(save_path, index=None)

    var_list = list(df_std.loc[df_std["std"] == 0, "var_name"]) 
    print("方差=0的特征有", var_list)
    print("过滤掉方差=0的特征")
    df = df.drop(columns=var_list)

    # 返回方差>0.1的特征
    from sklearn.varure_selection import VarianceThreshold
    # array_var = VarianceThreshold(threshold=0.1).fit_transform(df)  # 返回一个array
    return df

     

# 计算各个特征两两之间的pearson线性相关系数(注意，没有p值，只有r值)
def get_var_corr(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("只保留var_importance>0且AUC>=0.52且miss_rate<0.8的特征，然后计算两两之间的皮尔逊相关系数")
    df_stat = pd.read_csv("result/feat_miss_auc_importance_iv.csv")
    var_list = df_stat[(df_stat['var_importance'] > 0)
                       & (df_stat["var_avg_auc"] >= 0.52)
                       & (df_stat["var_miss_rate_all"] < 0.8)
                       & (df_stat["all_iv_sum"] >= 0.02)]['var_name'].tolist()
    print("len(var_list)", len(var_list))
    
    df_corr = df_tmp[var_list].corr(method='pearson')
    print("画相关性的热图")
    sns.heatmap(df_corr, cmap=plt.cm.RdYlBu_r, vmin=-0.25, annot=True, vmax=0.6)
    plt.title('Correlation Heatmap')
    plt.show()
    print("保存相关性矩阵,保存路径为：result/var_corr.csv")
    df_corr.to_csv("result/feat_corr.csv", float_format='%.4f')
    # df_corr = df.corr('kendall')  # Kendall Tau相关系数
    # df_corr = df.corr('spearman')  # spearman秩相关系数
    # df_corr1 = df.corr()["overdue_day"]  # 只显示其他特征与“overdue_day”的相关系数
    # df_corr2 = df["td_credit_score"].corr(df["td_id_risk"])  # 只计算2个变量之间的相关性系数
    print("get feat corr finished!")
    
    
    
# 对离散特征进行one-hot编码
def one_hot(df, categorical_var_list):
    '''
    :param df: 原数据
    :param poly_var_base: 一个list,离散特征的名称
    :return:
    '''
    print("开始对离散特征进行one-hot编码")
    # original_columns = list(df.columns)
    df = pd.get_dummies(df_model, columns=categorical_var_list, dummy_na=True)
    # new_columns = [c for c in df_model.columns if c not in original_columns]
    print("输出one-hot编码之后的特征名称")
    print(df.columns)
    return df 
    
    

# 构造多项式特征
def create_poly_var(df, poly_var_base):
    '''
    :param df: 原数据
    :param poly_var_base: 一个存有特征名称的list，以这个为基础特征来构造多项式特征
    :return:
    '''
    from sklearn.preprocessing import PolynomialFeatures
    print("构造多项式特征")
    df1 = df
    df_poly = df1[poly_var_base]
    poly_transformer = PolynomialFeatures(degree=3)  # Create the polynomial object with specified degree
    poly_transformer.fit(df_poly)  # Train the polynomial varures
    df_poly = poly_transformer.transform(df_poly)  # Transform the varures，返回一个array
    print('Polynomial Features shape: ', df_poly.shape)
    print('Polynomial Features name: ', poly_transformer.get_varure_names(input_varures=poly_var_base))

    # 将生成的多项式特征放在一个dataframe里面
    df_poly = pd.DataFrame(df_poly, columns=poly_transformer.get_varure_names(poly_var_base))
    # 看一下生成的多项式特征与label的相关性，跟原来的比较，是否有提高
    df_poly["label"] = df1["label"]
    print("看一下生成的多项式特征与label的spearson相关系数，跟原来的进行比较，是否有提高\n", df_poly.corr()["label"].sort_values())

    # 如果生成的多项式特征与label的spearson相关系数，跟原来的进行比较有提高，将多项式特征与原特征用关联键merge在一起
    df_poly["user_id"] = df1["user_id"]
    df_poly = df_poly.drop(columns=poly_var_base)
    df_poly = df_poly.drop(columns=["label"])
    df_poly_final = pd.merge(df1, df_poly, on="user_id", how="left")
    print("加入多项式特征后的shape", df_poly_final.shape)
    print("加入多项式特征后的columns", df_poly_final.columns)
    return df_poly_final


    
# lgb计算每一个特征的单独auc(交叉验证求平均AUC)
def get_var_avg_auc(df):
    print("lgb计算每一个特征的单独auc(交叉验证求平均AUC)")
    from lightgbm.sklearn import LGBMClassifier
    from sklearn import metrics
    from sklearn.model_selection import KFold
    import numpy as np

    df1 = df.copy()
    # df1["label"] = df1.apply(lambda x: 1 if x["overdue_day"]>7 else 0, axis=1)
    var_var = df1.columns.tolist()
    var_list = [x for x in var_var if x not in ['user_id', 'ds', 'loan_id', 'create_time', 'created',
                                                 'dt', 'type', "overdue_day", "overdue", "label"]]
    auc_dict = {}
    for i, var in enumerate(var_list):
        print(var.center(80, '='))
        print('i=%s' % i)
        print("若该特征miss_rate>=0.9，AUC直接赋值成-99")
        if df_test[df_test[var].isnull()].shape[0] >= 0.99 * df_test.shape[0]:
            auc = -99
        else:
        try:
            kf = KFold(n_splits=3, random_state=0)
            train_auc_list = []
            test_auc_list = []
            classifier = LGBMClassifier(verbose=-1, max_depth=3, silent=True)
			# 交叉验证求平均AUC
            for train_index, test_index in kf.split(df1):
                train_cross = df1.iloc[train_index, :]
                test_cross = df1.iloc[test_index, :]
                classifier.fit(train_cross[var].values.reshape(-1, 1), train_cross['label'])  # 训练
                train_prob = classifier.predict_proba(matrix_train_cross[var].values.reshape(-1, 1))[:, 1]  # 预测
                train_auc = metrics.roc_auc_score(train_cross['label'], train_prob)
                train_auc_list.append(train_auc)
                test_prob = classifier.predict_proba(matrix_test_cross[var].values.reshape(-1, 1))[:, 1]  # 预测
                test_auc = metrics.roc_auc_score(test_cross['label'], test_prob)
                test_auc_list.append(test_auc)
                
            test_auc = np.mean(test_auc_list) # 求平均AUC
            auc = test_auc
            auc_var = np.var(test_auc_list)
        except Exception as e:
            auc = -88
        auc_dict[var] = auc

    df_auc = pd.DataFrame({'var_name': list(auc_dict.keys()), 'var_avg_auc': list(auc_dict.values())})
    df_auc = df_auc.sort_values(by='var_avg_auc', ascending=False)  # 根据AUC降序排列
    print("保存平均AUC,保存地址：result/var_auc.csv")
    df_auc.to_csv("result/var_auc.csv", index=None, float_format='%.4f')
    print("get var auc finished!")


# 用lgb计算单特征的importance
def get_varure_importance(df, split_train_test_mode):
    print("将特征放在一起，lgb计算每一个特征的importance，用于筛选特征")
    from lightgbm.sklearn import LGBMClassifier
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    
    # 随机划分训练集和测试集
    def split_train_test_random(df_tmp):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(df_tmp, df_tmp["label"], test_size=0.3,
                                                            random_state=0)
        df_train = pd.DataFrame(X_train)
        df_test = pd.DataFrame(X_test)
        print("df_train", df_train.shape, df_train[df_train['label'] == 1].shape[0] * 1.0 / df_train.shape[0])
        print("df_test", df_test.shape, df_test[df_test['label'] == 1].shape[0] * 1.0 / df_test.shape[0])
        return df_train, df_test

    # 按照时间来划分训练集和测试集
    def split_train_test_on_time(df_tmp):
        df_tmp["credit_time"] = df_tmp["credit_time"].apply(lambda x: x[0:10])
        df_train = df_tmp[df_tmp["credit_time"] < '2018-09-01'].reset_index()
        df_test = df_tmp[df_tmp["credit_time"] >= '2018-09-01'].reset_index()
        print("df_train", df_train.shape, df_train[df_train['label'] == 1].shape[0] * 1.0 / df_train.shape[0])
        print("df_test", df_test.shape, df_test[df_test['label'] == 1].shape[0] * 1.0 / df_test.shape[0])
        return df_train, df_test
        
    df_tmp = df.copy()
    df_auc_miss = pd.read_csv(r"./result/var_miss_auc.csv")
    print("去掉auc<0.52且miss_rate>=0.8的特征")
    df_auc_miss = df_auc_miss[(df_auc_miss["var_auc"] >= 0.52) & (df_auc_miss["var_all_miss_rate"] < 0.8)]
    var_list = df_auc_miss["var_name"].tolist()
    # print(var_list)
    var_list = [var for var in var_list if var not in ["user_id", "label", "id_card",
                                                       "mobile", "ic_number", "app_dt",
                                                       "etl_dt", "dt", "id", "certno"]]

    # 设定划分训练集和测试集的模式
    # split_train_test_mode = 'random'
    if split_train_test_mode == 'random':
        # 随机划分训练集和测试集
        df_train, df_test = split_train_test_random(df_tmp)
    if split_train_test_mode == 'on_time':
        # 按照时间来划分训练集和测试集
        df_tmp["credit_time"] = df_tmp["credit_time"].apply(lambda x: x[0:10])
        df_train, df_test = split_train_test_on_time(df_tmp)

    clf = LGBMClassifier(n_estimators=4000, learning_rate=0.002, max_depth=6, num_leaves=20,
                         min_child_samples=4500, reg_alpha=0.4, reg_lambda=0.4, is_unbalance=True,
                         max_bin=150, min_child_weight=10,
                         verbose=-1)
    clf.fit(df_train[var_list], df_train["label"])
    var_importance = clf.varure_importances_
    df_var_importance = pd.DataFrame({"var_name": var_list, "var_importance": var_importance})
    df_var_importance = df_var_importance.sort_values(by="var_importance", ascending=False)
    df_var_importance.to_csv(r"./result/var_importance.csv", index=None)
    print("get var importance finished!")

    prob_test = clf.predict_proba(df_test[var_list])[:, 1]
    prob_train = clf.predict_proba(df_train[var_list])[:, 1]
    train_auc = metrics.roc_auc_score(df_train['label'], prob_train)
    test_auc = metrics.roc_auc_score(df_test['label'], prob_test)

    print("train auc:", train_auc)
    print("test auc:", test_auc)


# merge 缺失率、auc
def merge_miss_auc():
    df_miss = pd.read_csv("result/var_miss.csv")
    df_auc = pd.read_csv("result/var_auc.csv")
    df_stat = pd.merge(df_miss, df_auc, on='var_name', how='left')
    df_stat = df_stat.sort_values(by=['var_avg_auc', 'var_miss_rate_all'], ascending=False)
    df_stat = df_stat[['var_name', 'var_avg_auc', 'var_miss_rate_all']]
    df_stat = df_stat.sort_values(by=['var_avg_auc'], ascending=False)
    df_stat.to_csv("result/var_miss_auc.csv", index=None, float_format='%.4f')
    print(df_stat)
    print("merge_miss_auc finished!")


# merge 缺失率、auc、importance
def merge_miss_auc_importance():
    df_miss = pd.read_csv("result/var_miss.csv")
    df_auc = pd.read_csv("result/var_auc.csv")
    df_importance = pd.read_csv("result/var_importance.csv")

    df_stat = pd.merge(df_miss, df_auc, on='var_name', how='left')
    df_stat = pd.merge(df_stat, df_importance, on='var_name', how='left')
    df_stat = df_stat.sort_values(by=['var_auc', 'var_all_miss_rate','var_importance'], ascending=False)
    df_stat = df_stat[['var_name', 'var_auc', 'var_all_miss_rate','var_importance']]
    df_stat.to_csv("result/var_miss_auc_importance.csv", index=None, float_format='%.4f')
    print(df_stat)
    print("merge_miss_auc_importance finished!")
    
    
# 根据pkl，得到预测样本df_test的输出概率值和每一个特征的importance
def get_proba_result(df_test, pkl_path, save_proba_path):
    print("根据pkl，得到预测样本df_test的输出概率值pvalue和每一个特征的importance")
    from sklearn.externals import joblib
    classifier = joblib.load(pkl_path)  # 加载模型
    # 按照特征的编号从小到大进行排序。
    # 因为模型训练的时候，特征是有顺序的。使用训练好的pkl时，测试的特征也应该是一样的顺序
    columns = df_test.columns.tolist().sort() 
    df_importance = pd.DataFrame({"varures":columns, "importances":classifier.varure_importances_})
    print("varure importances:", df_importance)
    
    df_test_var = df_test.drop(columns={"user_id"})    
    p_values = classifier.predict_proba(df_test_var)[:, 1]  # df_test_var中只有特征，没有label
    df_test["p_value"] = p_values
    df_test.to_csv(save_proba_path,index=None)
    


if __name__ == '__main__':
    # 加载数据
    df = pd.read_csv(r"E:\work\2 mission\1 acard\acard_dae\source\final_main_0814_1007_1.csv", na_values=[-1.0,-2.0,-3.0,-99.0])
    df["label"] = df.apply(lambda x: 1 if x["overdue_day"] >7 else 0, axis=1)


    