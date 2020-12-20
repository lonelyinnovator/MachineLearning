# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
import jieba


# 对文本进行特征值化
def textVectorizer():
    tv = CountVectorizer()
    data1 = tv.fit_transform(['Life is too short, I love python', 'Life is too long, I hate python'])
    data2 = tv.fit_transform([' '.join(list(jieba.cut('人生苦短，我用python'))), ' '.join(list(jieba.cut('人生漫长，不用python')))])
    print(tv.get_feature_names())
    print(data2.toarray())



if __name__ == '__main__':
    textVectorizer()