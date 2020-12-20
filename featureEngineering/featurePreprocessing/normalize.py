# -*- coding: utf-8 -*-
from sklearn.preprocessing import MinMaxScaler

# 归一化处理
def normalize():
    mm = MinMaxScaler(feature_range=(3,4))
    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
    print(data)


if __name__ == '__main__':
    normalize()