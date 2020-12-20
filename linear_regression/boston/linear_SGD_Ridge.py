# -*- coding: utf-8 -*-
# author: lonelyinnovator
# datetime:2020/11/20 14:37
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error
import joblib


def boston():
    """
    用正规方程和梯度下降进行线性回归预测房价
    @return:
    """

    # 获取数据
    bos = load_boston()

    # 分割数据集
    x_train, x_test, y_train, y_test = train_test_split(bos.data, bos.target, test_size=0.25)
    # 标准化, 特征值和目标值都必须进行标准化，实例化两个标准化的对象
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test.reshape(-1, 1))

    # 正规方程预测
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print(lr.coef_)
    y_lr_predict = std_y.inverse_transform(lr.predict(x_test))
    print(y_lr_predict)
    print('正规方程的均方误差：', mean_squared_error(std_y.inverse_transform(y_test), y_lr_predict))

    # 保存训练好的模型
    joblib.dump(lr, 'test.pkl')

    # 加载保存的模型
    model = joblib.load('test.pkl')
    print('*' * 100)
    y_model_predict = std_y.inverse_transform(model.predict(x_test))
    print(y_model_predict)
    print('模型的均方误差：', mean_squared_error(std_y.inverse_transform(y_test), y_model_predict))

    # 梯度下降预测
    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)
    print(sgd.coef_)
    y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test))
    print(y_sgd_predict)
    print('梯度下降的均方误差：', mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict))

    # 岭回归预测
    rd = Ridge(alpha=1.0)
    rd.fit(x_train, y_train)
    print(rd.coef_)
    y_rd_predict = std_y.inverse_transform(rd.predict(x_test))
    print(y_rd_predict)
    print('岭回归下降的均方误差：', mean_squared_error(std_y.inverse_transform(y_test), y_rd_predict))

    return None


if __name__ == '__main__':
    boston()
