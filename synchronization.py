import os
import pandas as pd
Bt_train = pd.read_csv('Features/Bert/train_data.csv')
Bt_validation = pd.read_csv('Features/Bert/validation_data.csv')
Bt_test_data = pd.read_csv('Features/Bert/test_data.csv')

Fb_train = pd.read_csv('Features/Face Body/train_data.csv')
Fb_validation = pd.read_csv('Features/Face Body/validation_data.csv')
Fb_test_data = pd.read_csv('Features/Face Body/test_data.csv')

# z = Fb_train.head()
# Fb_drop_cols = [(f'{i}') for i in range(0, 552)]
# Fb_train.drop([*Fb_drop_cols], axis=1, inplace=True)
# Bt_drop_cols = [(f'{i}') for i in range(0, 512)]
# Bt_train.drop([*Bt_drop_cols], axis=1, inplace=True)

Fb_train.drop(['OPENMINDEDNESS_Z', 'CONSCIENTIOUSNESS_Z', 'EXTRAVERSION_Z', 'AGREEABLENESS_Z', 'NEGATIVEEMOTIONALITY_Z', 'gender', 'Unnamed: 0'], axis=1, inplace=True)

def convert(l, suffix):
    # it = iter(l)
    key_ori = [str(i) for i in l]
    key_ori = iter(key_ori)
    key = [str(i) + '_' + suffix for i in l]
    key = iter(key)
    res_dct = dict(zip(key_ori, key))
    return res_dct
fb_rename_cols = [i for i in range(0, 552)]
fb_rename_cols = convert(fb_rename_cols, 'fb')
Fb_train.rename(columns=fb_rename_cols, inplace=True)

bt_rename_cols = [i for i in range(0, 512)]
bt_rename_cols = convert(bt_rename_cols, 'bt')
Bt_train.rename(columns=bt_rename_cols, inplace=True)


f_merged = pd.merge(Bt_train, Fb_train, on=['ID_y', 'minute',
                                            'session'], how='inner')
# f_merged = pd.merge(Bt_train, Fb_train, on=['ID_y', 'minute', 'session'], how='inner')
f_merged = Bt_train.merge(Fb_train, on=['ID_y', 'minute', 'session', 'Video'], how='inner')
f_merged.drop(['Video'], axis=1, inplace=True)

# bert_test.to_csv('./bert_formated_csv/test_data.csv')
# bert_train.to_csv('./bert_formated_csv/train_data.csv')
# bert_validation.to_csv('./bert_formated_csv/validation_data.csv')
# drop duplicates based on ID_y, minute, session
# f_merged.drop_duplicates(subset=['ID_y', 'minute', 'session'], inplace=True)

# both of them have the same number of unique Id_y
# fb_id = Fb_train['ID_y'].unique()
# bt_id = Bt_train['ID_y'].unique()
k=1


