# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# 对鸢尾花数据进行k-近邻yuc
def iris_knn():
    # 读取数据
    li = load_iris()

    # 分割训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, train_size=0.25)

    # knn算法
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(x_train, y_train)
    y_predict = knn.predict(x_test)
    print(y_predict)
    print('*' * 100)
    print(y_test)
    print('*' * 100)
    print(knn.score(x_test, y_test))


if __name__ == '__main__':
    iris_knn()

