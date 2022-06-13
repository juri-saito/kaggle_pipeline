import pandas as pd
import sys, os
from sklearn.preprocessing import LabelEncoder
import yaml
import csv
from base import Feature, get_arguments, generate_features

sys.path.append(os.pardir)

CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE, encoding='utf-8_sig') as file:
    yml = yaml.load(file)

RAW_DATA_DIR_NAME = yml['SETTING']['RAW_DATA_DIR_NAME']
Feature.dir = yml['SETTING']['FEATURE_PATH']
feature_memo_path = Feature.dir + '_features_memo.csv'

# Target
class survived(Feature):
    def create_features(self):
        self.train['Survived'] = train['Survived']
        create_memo('Survived', '生存フラグ。今回の目的変数。')

class sex(Feature):
    def create_features(self):
        self.train['Sex'] = train['Sex']
        self.test['Sex'] = test['Sex']
        create_memo('Sex', '性別')

class sex_label_encoder(Feature):
    def create_features(self):
        cols = 'Sex'
        tmp_df = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)
        le = LabelEncoder().fit(tmp_df[cols])
        self.train['sex_label_encoder'] = le.transform(train[cols])
        self.test['sex_label_encoder'] = le.transform(test[cols])
        create_memo('sex_label_encoder', '性別をラベルエンコーディングしたもの')


# 特徴量メモのCSVファイルを作成
def create_memo(col_name, desc):

    file_path = Feature.dir + '_features_memo.csv'
    if not os.path.isfile(file_path):
        with open(file_path, "w", encoding="utf-8_sig"):pass # 新規作成

    with open(file_path, 'r+', encoding="utf-8_sig") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines] # 改行文字の除去

        # 既に特徴量メモファイルに書き込まれた特徴量でないか確認
        col = [line for line in lines if line.split(',')[0] == col_name]
        if len(col) != 0:return

        writer = csv.writer(f)
        writer.writerow([col_name, desc])

if __name__ == '__main__':
    # CSVのヘッダーを書き込み
    create_memo('特徴量', 'メモ')

    args = get_arguments()
    train = pd.read_csv(RAW_DATA_DIR_NAME + 'train.csv')
    test = pd.read_csv(RAW_DATA_DIR_NAME + 'test.csv')
    
    # globals()でtrain,testのdictionaryを渡す
    generate_features(globals(), args.force)
