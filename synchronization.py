import os
import pandas as pd

# Bt = pd.read_csv('Features/Bert/train_data.csv')
# Bt_validation = pd.read_csv('Features/Bert/validation_data.csv')
# Bt_test_data = pd.read_csv('Features/Bert/test_data.csv')
#
# Fb = pd.read_csv('Features/Face Body/train_data.csv')
# Fb_validation = pd.read_csv('Features/Face Body/validation_data.csv')
# Fb_test_data = pd.read_csv('Features/Face Body/test_data.csv')

def convert(l, suffix):
    # it = iter(l)
    key_ori = [str(i) for i in l]
    key_ori = iter(key_ori)
    key = [str(i) + '_' + suffix for i in l]
    key = iter(key)
    res_dct = dict(zip(key_ori, key))
    return res_dct

def merge(modality1, modality2, mode):
    Fb = pd.read_csv(f'Features/{modality1}/{mode}_data.csv')
    Bt = pd.read_csv(f'Features/{modality2}/{mode}_data.csv')
    drop_cols = ['Unnamed: 0', 'OPENMINDEDNESS_Z', 'CONSCIENTIOUSNESS_Z', 'EXTRAVERSION_Z', 'AGREEABLENESS_Z', 'NEGATIVEEMOTIONALITY_Z', 'gender']
    for col in drop_cols:
        try:
            Fb.drop(
                [col], axis=1, inplace=True)
        except:
            pass
    
    #todo a check_moda func
    fb_rename_cols = [i for i in range(0, 552)]
    
    fb_rename_cols = convert(fb_rename_cols, 'fb')
    Fb.rename(columns=fb_rename_cols, inplace=True)

    bt_rename_cols = [i for i in range(0, 512)]
    bt_rename_cols = convert(bt_rename_cols, 'bt')
    Bt.rename(columns=bt_rename_cols, inplace=True)

    f_merged = pd.merge(Bt, Fb, on=['ID_y', 'minute',
                                          'session'], how='inner')
    # f_merged = pd.merge(Bt, Fb, on=['ID_y', 'minute', 'session'], how='inner')
    f_merged = Bt.merge(Fb, on=['ID_y', 'minute', 'session', 'Video'], how='inner')

    drop_cols = ['Unnamed: 0', 'Video']
    for col in drop_cols:
        try:
            f_merged.drop(
                [col], axis=1, inplace=True)
        except:
            pass
    # f_merged.drop(['Video'], axis=1, inplace=True)
    output_dir = 'multi_modality_csv'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    f_merged.to_csv(f'{output_dir}/{mode}.csv')
    return f_merged

for mode in ['train', 'test', 'validation']:
    merge('Face Body', 'Bert',  mode)
# Bt = pd.read_csv('Features/Bert/train_data.csv')
# Bt_validation = pd.read_csv('Features/Bert/validation_data.csv')
# Bt_test_data = pd.read_csv('Features/Bert/test_data.csv')
#
# Fb = pd.read_csv('Features/Face Body/train_data.csv')
# z = Fb.head()
# Fb_drop_cols = [(f'{i}') for i in range(0, 552)]
# Fb.drop([*Fb_drop_cols], axis=1, inplace=True)
# Bt_drop_cols = [(f'{i}') for i in range(0, 512)]
# Bt.drop([*Bt_drop_cols], axis=1, inplace=True)






# bert_test.to_csv('./bert_formated_csv/test_data.csv')
# bert_train.to_csv('./bert_formated_csv/train_data.csv')
# bert_validation.to_csv('./bert_formated_csv/validation_data.csv')
# drop duplicates based on ID_y, minute, session
# f_merged.drop_duplicates(subset=['ID_y', 'minute', 'session'], inplace=True)

# both of them have the same number of unique Id_y
# fb_id = Fb['ID_y'].unique()
# bt_id = Bt['ID_y'].unique()
k = 1


