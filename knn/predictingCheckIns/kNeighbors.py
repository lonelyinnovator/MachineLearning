# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd


# k-近邻预测用户签到位置
def knncls():
    # 读取数据
    data = pd.read_csv('./data/train.csv')
    # print(data.head(10))

    # 处理数据
    # 1.缩小数据范围
    data = data.query('x > 1.0 & x < 1.25 & y > 2.5 & y < 2.75')

    # 处理时间的数据
    time_value = pd.to_datetime(data.loc[:, 'time'], unit='s')

    # 将日期格式转换成字典格式
    time_value = pd.DatetimeIndex(time_value)

    # 构造一些特征
    dataCopy = data.copy()
    dataCopy.loc[:, 'day'] = time_value.day
    dataCopy.loc[:, 'hour'] = time_value.hour
    dataCopy.loc[:, 'weekday'] = time_value.weekday

    # 把时间戳特征删除
    dataCopy = dataCopy.drop(['time'], axis=1)
    # print(data1)

    # 把签到数量少于n个目标位置删除
    place_count = dataCopy.groupby('place_id').count()
    tf = place_count[place_count.row_id > 10].reset_index()
    dataCopy = dataCopy[dataCopy['place_id'].isin(tf.place_id)]

    # print(dataCopy)

    # 取出数据中的特征值和目标值
    y = dataCopy['place_id']
    x = dataCopy.drop(['place_id'], axis=1).drop(['row_id'], axis=1)

    # 进行数据的分割训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程（标准化）
    std = StandardScaler()

    # 对测试集和训练集的特征值进行标准化
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 进行算法流程
    knn = KNeighborsClassifier()
    # knn.fit(x_train, y_train)
    # y_predict = knn.predict(x_test)
    # print(y_predict)
    # print(knn.score(x_test, y_test))

    # 构造一些参数的值用于网格搜索
    param = {'n_neighbors': [3, 5, 10]}

    # 进行网格搜索
    gc = GridSearchCV(knn, param_grid=param, cv=5)
    gc.fit(x_train, y_train)

    # 预测准确率
    print('在测试集上的准确率：', gc.score(x_test, y_test))
    print('在交叉验证中最好的结果：', gc.best_score_)
    print('选择的最好的模型是：', gc.best_estimator_)
    print('每个超参数每次验证的结果：', gc.cv_results_)


if __name__ == '__main__':
    knncls()
