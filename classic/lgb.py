import numpy as np
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb

import sys
sys.path.append("../")

from feature_select_tfidf import get_features


def rmsel(true_label,pred):
    true_label = np.array(true_label)
    pred = np.array(pred)
    n = len(true_label)
    a = true_label - pred
    rmse = np.sqrt(np.sum(a * a)/n)
    b = 1/(1+rmse)
    return b


param = {'boosting_type': 'gbdt',
         'objective': 'regression',  # regression, multiclass
         'num_leaves': 200,
         'learning_rate': 0.01,
         'n_estimators': 1000,
         'silent': False
         }


def compute_score(result_pred):
    prob = []
    score = []
    for sco, pr in result_pred:
        prob.append(pr)
        score.append(int(sco))
    prob = np.array(prob)
    score = np.array(score)
    prob = prob / prob.sum()
    return sum(prob * score)


def train_lgb(select_k_feature=2315):
    X, test, y, train_id, test_id = get_features(select_k_feature)
    print("train %s, test %s" % (X.shape, test.shape))
    fast_pred = []
    n_split = 5
    folds = list(StratifiedKFold(n_splits=n_split, shuffle=True, random_state=2017).split(X, y))
    for train_index, test_index in folds:
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier = lgb.LGBMRegressor(**param)
        classifier.fit(x_train, y_train, verbose=1)

        x_pred = classifier.predict(x_test)
        pred = x_pred
        print(rmsel(y_test, pred))

    fast_pred = np.array(fast_pred)
    fast_pred = np.mean(fast_pred, axis=0)
    return fast_pred


if __name__ == '__main__':
    train_lgb()
