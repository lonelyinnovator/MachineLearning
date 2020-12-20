# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

def tfidfVectorizer():
    tf = TfidfVectorizer()
    data1 = tf.fit_transform(['Life is too short, I love python', 'Life is too long, I hate python'])
    data2 = tf.fit_transform([' '.join(list(jieba.cut('人生苦短，我用python'))), ' '.join(list(jieba.cut('人生漫长，不用python')))])
    print(tf.get_feature_names())
    print(data2.toarray())


if __name__ == '__main__':
    tfidfVectorizer()