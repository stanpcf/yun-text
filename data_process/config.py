# coding: utf8


class cfg:
    """
    该类保存一些全局的网络配置
    """
    MAX_FEATURE = 200000
    SAMPLE_WEIGHT = [10, 6, 3, 1.2, 1]
    ONE_HOT_NUM = 5

    MY_EMBED_WIND = 5   # 该变量保存的是自己训练的词向量的context窗口长度
    MY_EMBED_SIZE_CHOICE = {100, 200, 300}      # 可选的词向量维度

    MODEL_FIT_validation_split = 0.1            # model.fit, keras

    TRAIN_TEST_SPLIT_random_state = None         # train_test_split,  sklearn

    TEXT_CNN_filters = [2, 3, 4]    # text cnn 多卷积通道的卷积核大小
    TEXT_CNN_CONV_NUM = 1           # text cnn 的多通道卷积的深度

    LSTM_hidden_size = 128
