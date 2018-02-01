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

    TRAIN_TEST_SPLIT_random_state = 123         # train_test_split,  sklearn
