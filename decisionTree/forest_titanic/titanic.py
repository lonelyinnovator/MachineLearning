# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def titanic():
    """
    决策树对泰坦尼克号进行预测生死
    @return: None
    """

    # 读取数据
    titan = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

    # 处理数据，找出特征值和目标值
    x = titan[['pclass', 'age', 'sex']]
    y = titan['survived']

    # 缺失值处理
    x['age'].fillna(x['age'].mean(), inplace=True)

    # 分割数据
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程
    dic = DictVectorizer(sparse=False)
    x_train = dic.fit_transform(x_train.to_dict(orient='records'))
    x_test = dic.transform(x_test.to_dict(orient='records'))

    # 随机森林进行预测
    rf = RandomForestClassifier()
    param = {'n_estimators': [120, 200, 300, 500, 800, 1200], 'max_depth': [5, 8, 15, 20, 30]}

    # 网格搜索和交叉验证
    gc = GridSearchCV(rf, param_grid=param, cv=2)
    gc.fit(x_train, y_train)
    print(gc.score(x_test, y_test))
    print(gc.best_estimator_)

    return None


if __name__ == '__main__':
    titanic()
