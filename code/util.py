import numpy as np
import logging
import sys, os
import joblib
import yaml
import pandas as pd

"""
ファイル入出力
ログの出力・表示
計算結果の出力・表示
"""

CONFIG_FILE = '../configs/config.yaml'

with open(CONFIG_FILE, encoding="utf-8_sig") as file:
    yml = yaml.load(file, Loader=yaml.SafeLoader)
RAW_DATA_DIR_NAME = yml['SETTING']['RAW_DATA_DIR_NAME']
SUB_DIR_NAME = yml['SETTING']['SUB_DIR_NAME']

# tensorflowとloggingのcollisionに対応
try:
    import absl.logging
    # https://github.com/abseil/abseil-py/issues/99
    logging.root.removeHandler(absl.logging._absl_handler)
    # https://github.com/abseil/abseil-py/issues/102
    absl.logging._warn_preinit_stderr = False
except Exception:
    pass

class Util:
    @classmethod
    def dump(cls, value, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path):
        return joblib.load(path)

    @classmethod
    def dump_df_pickle(cls, df, path):
        df.to_pickle(path)

    @classmethod
    def load_df_pickle(cls, path):
        return pd.read_pickle(path)

class Logger:

    def __init__(self, path):
        self.result_logger = logging.getLogger(path + 'result')
        stream_handler = logging.StreamHandler()
        file_result_handler = logging.FileHandler(path + 'result.log')
        self.result_logger.addHandler(stream_handler)
        self.result_logger.addHandler(file_result_handler)
        self.result_logger.setLevel(logging.INFO)

    def result(self, message):
        self.result_logger.info(message)

    def result_scores(self, run_name, scores):
        # 計算結果をコンソールと計算結果用ログに出力
        dic = dict()
        dic['name'] = run_name
        dic['score'] = np.mean(scores)
        for i, score in enumerate(scores):
            dic[f'score{i}'] = score
        self.result(self.to_ltsv(dic))

    def to_ltsv(self, dic):
        return '\t'.join(['{}:{}'.format(key, value) for key, value in dic.items()])

class Submission:
    @classmethod
    def create_submission(cls, run_name, path, sub_y_column):
        submission = pd.read_csv(RAW_DATA_DIR_NAME + 'sample_submission.csv')
        pred = Util.load_df_pickle(path + f'{run_name}-pred.pkl')
        submission[sub_y_column] = pred
        submission.to_csv(path + f'{run_name}_submission.csv', index=False)
