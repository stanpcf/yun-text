import pandas as pd
import fire


def adjust_score(prd_file):
    """把预测的数据里面的3分调整成4分, 1调整为3, 2调整为3"""
    prd = pd.read_csv(prd_file, header=None, names=['Id', 'prd'])
    prd.loc[prd[prd.prd == 3].index, 'prd'] = 4
    prd.loc[prd[prd.prd == 1].index, 'prd'] = 3
    prd.loc[prd[prd.prd == 2].index, 'prd'] = 3
    tmp = prd_file.split("/")
    tmp = tmp[:-1] + ["adjust_"+tmp[-1]]
    adjust_file = "/".join(tmp)
    prd.to_csv(adjust_file, index=False, header=False)


if __name__ == '__main__':
    fire.Fire()
