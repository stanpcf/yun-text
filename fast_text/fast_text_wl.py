import pandas as pd
import numpy as np

import jieba
import re
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor
import random
import fasttext
import lightgbm as lgb

import sys
sys.path.append("../")
from data_process import get_stop_words

data_path = '../input/YNU.EDU2018-ScenicWord/train_first.csv'
df = pd.read_csv(data_path,header=0, encoding='utf8')

test_data_path = '../input/YNU.EDU2018-ScenicWord/predict_first.csv'
test_df = pd.read_csv(test_data_path,header=0, encoding='utf8')

stop_word = get_stop_words()


def clean_str(stri):
    stri = re.sub(r'[0-9a-zA-Z]+', '', stri)
    cut_str = jieba.cut(stri.strip())
    list_str = [word for word in cut_str if word not in stop_word]
    stri = ' '.join(list_str)
    return stri


def fillnull(x):
    if x == '':
        return '空白'
    else:
        return x


df['Discuss'] = df['Discuss'].map(lambda x: clean_str(x))
test_df['Discuss'] = test_df['Discuss'].map(lambda x: clean_str(x))

df['Discuss'] = df['Discuss'].map(lambda x: fillnull(x))
test_df['Discuss'] = test_df['Discuss'].map(lambda x: fillnull(x))

vec = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, strip_accents='unicode', max_features=20000,
                      use_idf=1, smooth_idf=1, sublinear_tf=1)
vec.fit(pd.concat([df['Discuss'], test_df['Discuss']]))
test_df_tfidf_vec = vec.transform(test_df['Discuss'])

score_5_idx = df[df['Score']==5].index.tolist()
score_4_idx = df[df['Score']==4].index.tolist()
score_3_idx = df[df['Score']==3].index.tolist()
score_2_idx = df[df['Score']==2].index.tolist()
score_1_idx = df[df['Score']==1].index.tolist()


def spilt_sample(sample,n=4):
    num_sample = len(sample)
    sub_lenth = int(1/n * num_sample)
    sub_sample = []
    for i in range(n):
        sub = sample[i*sub_lenth:(i+1)*sub_lenth]
        sub_sample.append(sub)
    return sub_sample


score_5_sample = spilt_sample(score_5_idx)
score_4_sample = spilt_sample(score_4_idx)
score_3_sample = spilt_sample(score_3_idx)

df1_idx = [score_5_sample[0],score_4_sample[0],score_3_sample[0],score_2_idx,score_1_idx]
df1_idx = [idx for i_sample in df1_idx for idx in i_sample]
random.shuffle(df1_idx)

df2_idx = [score_5_sample[1],score_4_sample[1],score_3_sample[1],score_2_idx,score_1_idx]
df2_idx = [idx for i_sample in df2_idx for idx in i_sample]
random.shuffle(df2_idx)

df3_idx = [score_5_sample[2],score_4_sample[2],score_3_sample[2],score_2_idx,score_1_idx]
df3_idx = [idx for i_sample in df3_idx for idx in i_sample]
random.shuffle(df3_idx)

df4_idx = [score_5_sample[3],score_4_sample[3],score_3_sample[3],score_2_idx,score_1_idx]
df4_idx = [idx for i_sample in df4_idx for idx in i_sample]
random.shuffle(df4_idx)


df1 = df.loc[df1_idx,:]
df1 = df1.sample(frac = 1)
df2 = df.loc[df2_idx,:]
df2 = df2.sample(frac = 1)
df3 = df.loc[df3_idx,:]
df3 = df3.sample(frac = 1)
df4 = df.loc[df4_idx,:]
df4 = df4.sample(frac = 1)


def fasttext_data(data,label):
    fasttext_data = []
    for i in range(len(label)):
        sent = data[i]+"\t__label__"+str(int(label[i]))
        fasttext_data.append(sent)
    with open('train_wl.txt','w') as f:
        for data in fasttext_data:
            f.write(data)
            f.write('\n')
    return 'train_wl.txt'

def get_predict(pred):
    score = np.array([1,2,3,4,5])
    pred2 = []
    for p in pred:
        pr = np.sum(p * score)
        pred2.append(pr)
    return np.array(pred2)

def rmsel(true_label,pred):
    true_label = np.array(true_label)
    pred = np.array(pred)
    n = len(true_label)
    a = true_label - pred
    rmse = np.sqrt(np.sum(a * a)/n)
    b = 1/(1+rmse)
    return b


def fast_cv(df):
    df = df.reset_index(drop=True)
    X = df['Discuss'].values
    y = df['Score'].values
    fast_pred = []
    #     folds = list(KFold(n_splits=5, shuffle=True, random_state=2017).split(X, y))
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=2017).split(X, y))
    for train_index, test_index in folds:
        X_train, X_test = X[train_index], X[test_index]
        X_train = vec.transform(X_train)
        X_test = vec.transform(X_test)
        y_train, y_test = y[train_index], y[test_index]

        #         classifier = LogisticRegression(C=4, dual=True)
        #         classifier = GradientBoostingRegressor(max_depth=5)
        classifier = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05,
                                       n_estimators=20)
        classifier.fit(X_train, y_train)
        x_pred = classifier.predict(X_test)
        # print(x_pred)
        #         print(result)
        # pred = [compute_score(result_i) for result_i in result]
        pred = x_pred
        print(rmsel(y_test, pred))

        test_result = classifier.predict(test_df_tfidf_vec)
        fast_predi = test_result
        fast_pred.append(fast_predi)

    fast_pred = np.array(fast_pred)
    fast_pred = np.mean(fast_pred, axis=0)
    return fast_pred

print("pred1")
test_pred1 = fast_cv(df1)
print("pred2")
test_pred2 = fast_cv(df2)
print("pred3")
test_pred3 = fast_cv(df3)
print("pred4")
test_pred4 = fast_cv(df4)


data = np.zeros((len(test_df),5))
sub_df = pd.DataFrame(data)
sub_df.columns = ['Id','fast1','fast2','fast3','fast4']
sub_df['Id'] = test_df['Id'].values
sub_df['fast1'] = test_pred1
sub_df['fast2'] = test_pred2
sub_df['fast3'] = test_pred3
sub_df['fast4'] = test_pred4


sub_df['mean'] = sub_df.mean(axis=1)

print("pred")
test_pred = fast_cv(df)
sub_df['mean2'] = test_pred

print(sub_df.describe())

pred = sub_df['mean2'].values
pred = np.where(pred>4.7,5,pred)
sub_df['mean2'] = pred

from datetime import datetime
subfix = datetime.now().strftime('%y%m%d%H%M')
sub_df[['Id','mean2']].to_csv('fastsub2_wl_'+ subfix + '.csv',header=None,index=False)
