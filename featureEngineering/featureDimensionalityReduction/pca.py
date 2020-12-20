# -*- coding: utf-8 -*-
from sklearn.decomposition import PCA

# 主成分分析进行特征降维
def pca():
    pca = PCA(n_components=0.9)
    data = pca.fit_transform([[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]])
    print(data)


if __name__ == '__main__':
    pca()