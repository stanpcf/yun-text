# coding: utf-8


def yun_metric(y_true, y_pred):
    from math import sqrt
    from sklearn.metrics import mean_squared_error
    rmse = sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))
    score = 1./(1+rmse)
    return score


def tensor_yun_loss(y_true, y_pred):
    """该函数用于keras model 的compile阶段的评价"""
    from keras.losses import mean_squared_error
    import keras.backend as K
    rmse = K.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))
    score = 1. / (1+rmse)
    return score
