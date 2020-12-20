# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris, load_boston, fetch_20newsgroups
from sklearn.model_selection import train_test_split


def iris():
    li = load_iris()
    # print(li.data)
    # print(li.target)
    # print(li.DESCR)

    # x_train, y_train 训练集的特征值和目标值，x_test, y_test 测试集的特征值和目标值
    x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25)
    print('训练集的特征值和目标值: ', x_train, y_train)
    print('测试集的特征值和目标值: ', x_test, y_test)


def boston():
    lb = load_boston()
    print(lb.data)
    print(lb.target)


def news():
    news_twenty = fetch_20newsgroups()
    print(news_twenty.data)
    print(news_twenty.target)

if __name__ == '__main__':
    news()
