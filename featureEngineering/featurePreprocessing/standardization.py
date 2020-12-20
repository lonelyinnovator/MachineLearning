# -*- coding: utf-8 -*-
from sklearn.preprocessing import StandardScaler


# 标准化缩放
def stand():
    std = StandardScaler()
    data = std.fit_transform([[1, -1, 3], [2, 4, 2], [4, 6, -1]])
    print(data)


if __name__ == '__main__':
    stand()