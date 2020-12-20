# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


# 对20组新闻进行朴素贝叶斯文本分类
def naive_bayes():
    news = fetch_20newsgroups(subset='all')

    # 进行数据分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)

    # 对数据集进行特征抽取
    tf = TfidfVectorizer()

    # 以训练集当中的词的列表进行每篇文章重要性统计
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)

    # 进行朴素贝叶斯算法的计算
    mlt = MultinomialNB(alpha=1.0)
    mlt.fit(x_train, y_train)
    y_predict = mlt.predict(x_test)
    print(y_predict)
    # 得出准确率, 精确率和召回率
    print(mlt.score(x_test, y_test))
    print(classification_report(y_test, y_predict))


if __name__ == '__main__':
    naive_bayes()
