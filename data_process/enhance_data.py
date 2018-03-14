#!/usr/bin/env python
# coding: utf8

import uuid
import os
import pandas as pd

try:
    from .config import cfg
except ModuleNotFoundError as e:
    from config import cfg


class Enhance:
    def __init__(self, raw_df, concat_col, id_len=len('201e8bf2-77a2-3a98-9fcf-4ce03914e712'), id_prefix='enhances'):
        self.raw_df = raw_df
        self.concat_col = concat_col
        self.id_len = id_len
        self.prefix = id_prefix
        self.prefix_len = len(id_prefix)

    def _get_Id(self, samp_num):
        return [self.prefix + str(uuid.uuid1())[self.prefix_len:self.id_len] for _ in range(samp_num)]

    def get_fake_df(self):
        dts = [self.raw_df]
        for sco, num in enumerate(cfg.enhance_to, 1):
            df = self.raw_df[self.raw_df['Score'] == sco]
            samp_num = num - df.shape[0]
            if samp_num > 0:
                lh = df.sample(n=samp_num, replace=True, random_state=123)
                lh.reset_index(inplace=True)
                rh = df.sample(n=samp_num, replace=True, random_state=321)
                rh.reset_index(inplace=True)
                fake_data = dict()
                fake_data['Id'] = self._get_Id(samp_num)
                for col in self.concat_col:
                    fake_data[col] = lh[col] + " " + rh[col]
                fake_data['Score'] = [sco for _ in range(samp_num)]
                dt = pd.DataFrame(data=fake_data)
                dts.append(dt)
        data = pd.concat(dts)
        data.sample(frac=1).reset_index(drop=True)
        return data


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input")
    file = os.path.join(data_dir, "processed", "train_first.csv")
    enh_file = file[:-4] + "_enhance" + file[-4:]
    train = pd.read_csv(file)
    print(train.shape)
    cols = ['Discuss', 'fool', 'jieba', 'pynlpir', 'thulac']
    enh = Enhance(train, cols)
    data = enh.get_fake_df()
    print(data.shape)
    data.to_csv(enh_file, index=False)

