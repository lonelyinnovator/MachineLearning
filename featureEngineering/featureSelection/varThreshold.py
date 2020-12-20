# -*- coding: utf-8 -*-
from sklearn.feature_selection import VarianceThreshold



# 特征选择-删除低方差的特征
def varThreshold():
    var = VarianceThreshold(threshold=1.0)
    data = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    print(data)



if __name__ == '__main__':
    varThreshold()