# -*- coding: utf-8 -*-
from sklearn.feature_extraction import DictVectorizer

# 字典数据抽取
def dictVectorizer():
    dic = DictVectorizer(sparse=False)
    data = dic.fit_transform([{'name':'fsd', 'age':21}, {'name':'jhg', 'age':42}, {'name':'lo', 'age':324}])
    print(dic.get_feature_names())
    print(data)
    return None
if __name__ == '__main__':
    dictVectorizer()