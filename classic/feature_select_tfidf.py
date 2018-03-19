"""
该文件使用ridge对tfidf的特征进行feature select
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack


def pre_process(train_df, test_df):
    data_df = pd.concat([train_df, test_df])
    nrw_train, train_y, test_id = train_df.shape[0], train_df['Score'].values, test_df['Id']

    vec = TfidfVectorizer(ngram_range=(1, 4), min_df=2, max_df=0.8, strip_accents='unicode',
                          use_idf=1, smooth_idf=1, sublinear_tf=1)
    discuss_tf = vec.fit_transform(data_df['Discuss'])
    data = hstack((discuss_tf, )).tocsr()
    return data[:nrw_train], data[nrw_train:], train_y, test_id, vec


def get_features(select_k_feature=2000):
    """:param select_k_feature: 从ridge里面选择k个feature. defaults=2000"""
    train_df = pd.read_csv('../input/features/train_first.csv')
    train_df.drop_duplicates(subset='Discuss', keep='first', inplace=True)

    test_df = pd.read_csv('../input/features/predict_first.csv')


    train_x, test, train_y, test_id, vec = pre_process(train_df, test_df)

    model = Ridge(solver='auto', fit_intercept=True, alpha=1, max_iter=1000, normalize=False, tol=0.01)
    model.fit(train_x, train_y)

    items = sorted(vec.vocabulary_.items(), key=lambda item: item[1])
    _feature = [it[0] for it in items]
    feature = np.array(_feature)
    x = np.argpartition(model.coef_, -select_k_feature)[-select_k_feature:]

    im_vec = CountVectorizer(vocabulary=feature[x])
    im_train = im_vec.fit_transform(train_df.Discuss)
    im_test = im_vec.transform(test_df.Discuss)

    other_feature_name = test_df.columns.drop(['Discuss', 'Id', 'jieba']).tolist()
    other_feature_train = train_df[other_feature_name]
    other_feature_test = test_df[other_feature_name]

    train_feature = np.hstack([im_train.toarray(), other_feature_train])
    test_feature = np.hstack([im_test.toarray(), other_feature_test])
    return train_feature, test_feature, train_df['Score'].values, train_df['Id'].values, test_df['Id'].values
