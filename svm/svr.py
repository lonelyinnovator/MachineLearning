# -*- coding: utf-8 -*-
"""
author: 沈佳军
datetime:2020/12/13 20:50
"""
import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def svr():
    """
    使用svr对四姑娘山的客流量进行预测
    @return: None
    """

    # 从excel中读取三个sheet
    df1 = pd.read_excel('四姑娘山数据汇总（3个sheet）.xlsx', sheet_name='Sheet1')
    df2 = pd.read_excel('四姑娘山数据汇总（3个sheet）.xlsx', sheet_name='Sheet2')
    df3 = pd.read_excel('四姑娘山数据汇总（3个sheet）.xlsx', sheet_name='Sheet3')

    # 将三个sheet中的特征值和目标值合并
    data = pd.DataFrame(np.hstack(
        (df1[df1.columns[1:4]], df1[df1.columns[6:10]], df1[df1.columns[11:14]], df2[df2.columns[1:3]],
         df3[df3.columns[2:3]][:1523], df1[df1.columns[14:15]])))

    # 去除NaN
    data = data.dropna()

    # 重置标签
    data.reset_index(drop=True, inplace=True)

    # 使用正则表达式提取气温中的数字
    pattern = re.compile(r'-?\d+')
    for i in range(10, 12):
        for j in range(len(data[data.columns[i]])):
            data[data.columns[i]][j] = pattern.findall(data[data.columns[i]][j])[0]

    # 分割训练集和预测集
    x_train, x_test, y_train, y_test = train_test_split(data[data.columns[0:-1]], data[data.columns[-1]],
                                                        test_size=0.25)

    # pca降维
    pca = PCA(n_components=3)
    x_train = pd.DataFrame(
        np.hstack((pd.DataFrame(pca.fit_transform(x_train[x_train.columns[:9]])).astype(np.int64),
                   x_train[x_train.columns[9:]])))
    x_test = pd.DataFrame(
        np.hstack((pd.DataFrame(pca.fit_transform(x_test[x_train.columns[:9]])).astype(np.int64),
                   x_test[x_test.columns[9:]])))
    result = pd.DataFrame(np.vstack((x_train[x_train.columns[:3]], x_test[x_test.columns[:3]])))
    result.to_excel("百度搜索指数降维.xlsx", index=False)

    # 标准化
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test = std_y.transform(y_test.values.reshape(-1, 1))

    # 进行SVR回归预测
    model = SVR(C=1e3)
    model.fit(x_train, y_train.ravel())
    y_predict = std_y.inverse_transform(model.predict(x_test))
    print(y_predict.astype(np.int64))
    print('score: ', model.score(x_test, y_test))
    print('平均绝对误差：', mean_absolute_error(std_y.inverse_transform(y_test), y_predict))
    print("r2score: ", r2_score(std_y.inverse_transform(y_test), y_predict))
    return None


if __name__ == '__main__':
    svr()
