# -*- coding: utf-8 -*-
"""
author: lonelyinnovator
datetime:2020/11/25 0:26
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def logistic():
    """
    逻辑回归做二分类进行癌症预测（根据细胞的属性特征）
    @return:None
    """

    # 构造列标签名字
    column = ['Sample code number ', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
              'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
              'Class']

    # 读取数据
    data = pd.read_csv('breast-cancer-wisconsin.data', names=column)

    print(data)

    # 缺失值处理
    data = data.replace(to_replace='?', value=np.nan)
    data = data.dropna

    # 进行数据分割
    x_train, x_test, y_train, y_test = train_test_split(data[column[1:10]], data[column[10]], test_size=0.25)

    # 进行标准化处理(分类问题，目标值不需要标准化)
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 逻辑回归预测
    




    return None


if __name__ == '__main__':
    logistic()
