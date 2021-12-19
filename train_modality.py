#%%

import os
import h5py
import numpy as np
import pickle
import autokeras as ak
import tensorflow as tf
import pandas as pd
import argparse
#%%
def df_drop_cols(df, cols):
    for col in cols:
        try:
            df.drop(
                [col], axis=1, inplace=True)
        except:
            pass
    return df

def read_data_gender(path):
    data = pd.read_csv(path)
    M = data[data['gender'] == 'M']
    F = data[data['gender'] == 'F']
    return data, M, F

def read_data_age(path):
    data = pd.read_csv(path)

    Y = data[data['age'] <= 30]
    O = data[data['age'] > 30]
    return data, Y, O

def build_dataset(root, mode, seperator):
    print('build dataset...')
    labels = ['OPENMINDEDNESS_Z', 'CONSCIENTIOUSNESS_Z', 'EXTRAVERSION_Z', 'AGREEABLENESS_Z', 'NEGATIVEEMOTIONALITY_Z']

    if seperator == 'age':
        read_data = read_data_age
    elif seperator == 'gender':
        read_data = read_data_gender
    if mode == 'train':
        train_path = os.path.join(root, f'train_data.csv')
        data, M_data, F_data = read_data(train_path)

        val_path = os.path.join(root, f'validation_data.csv')
        val_data, val_M, val_F = read_data(val_path)

        data = data.append(val_data)
        M_data = M_data.append(val_M)
        F_data = F_data.append(val_F)

    else:
        path = os.path.join(root, f'{mode}_data.csv')
        data, M_data, F_data = read_data(path)

    # these are ground truth for each OCEAN; we will use them to evaluate the performance of the model.
    # [1,N]
    def get_OCEAN(d):
        OCEAN = []
        for i in range(0, 5):
            OCEAN.append(d.pop(labels[i]).to_numpy())
        return OCEAN

    comb_OCEAN = get_OCEAN(data)
    M_OCEAN = get_OCEAN(M_data)
    F_OCEAN = get_OCEAN(F_data)


    #%%
    # following train data only has the feature data
    drop_cols = ['ID_y', 'minute', 'session', 'gender', 'age','Unnamed: 0', 'Video', 'Unnamed: 0.1']

    # combined
    comb = df_drop_cols(data, drop_cols)
    M = df_drop_cols(M_data, drop_cols)
    F = df_drop_cols(F_data, drop_cols)

    Fb_drop_cols = [(f'{i}_fb') for i in range(0, 552)]
    Bt_drop_cols = [(f'{i}_bt') for i in range(0, 512)]

    comb_bt = np.array(comb.drop([*Fb_drop_cols], axis=1))
    comb_fb = np.array(comb.drop([*Bt_drop_cols], axis=1))
    # Gender

    M_bt = np.array(M.drop([*Fb_drop_cols], axis=1))
    M_fb = np.array(M.drop([*Bt_drop_cols], axis=1))

    F_bt = np.array(F.drop([*Fb_drop_cols], axis=1))
    F_fb = np.array(F.drop([*Bt_drop_cols], axis=1))

    # Combined
    comb_dataset = []
    M_dataset = []
    F_dataset = []
    for i in range(0, 5):
        comb_dataset.append((comb_fb, comb_bt, comb_OCEAN[i]))

    for i in range(0, 5):
        M_dataset.append((M_fb, M_bt, M_OCEAN[i]))

    for i in range(0, 5):
        F_dataset.append((F_fb, F_bt, F_OCEAN[i]))

    return comb_dataset, M_dataset, F_dataset

def main(args):
    comb_train_set, M_train_set, F_train_set = build_dataset('multi_modality_csv', 'train', 'gender')

    comb_val_set, M_val_set, F_val_set = build_dataset('multi_modality_csv', 'validation', 'gender')
    #
    # comb_train_set, Y_train_set, O_train_set = build_dataset('multi_modality_csv', 'train', 'age')
    #
    # comb_val_set, Y_val_set, O_val_set = build_dataset('multi_modality_csv', 'validation', 'age')

    # comb_test_set , M_test_set, F_test_set = build_dataset('multi_modality_csv', 'test')

    model_names = ['Model_O','Model_C','Model_E','Model_A','Model_N']

    #%%
    dump = './Dump Data'
    path = './Textual Models'
    training_name = 'multiModality_outputs'
    models_path = path+'/'+training_name
    # models_path = path +'/' + 'multiModality_outputs'
    combined_path = models_path+'/Combined'
    age_path = models_path+'/Age'
    gender_path = models_path+'/Gender'
    male_path = models_path+'/Gender/Male'
    female_path = models_path+'/Gender/Female'
    old_path = models_path+'/Age/Old'
    young_path = models_path+'/Age/Young'

    folders = [ dump
    ,path
    ,models_path
    ,combined_path
    ,age_path
    ,gender_path
    ,male_path
    ,female_path
    ,old_path
    ,young_path]

    #%%

    for folder in folders:
        # exist mkdir
        if not os.path.exists(folder):
            os.mkdir(folder)

    def train_setting(setting, train_set, val_set, path):
        print(f"{setting} TRAINING BEGINS")
        # iter for 5 OCEAN dataset
        i = int(args.OCEAN_id)

        train = train_set[i]
        val = val_set[i]
        print(model_names[i])
        # Define a regressor
        total_reg = ak.AutoModel(
          # by default, RegrssionHead uses MSE
          inputs=[ak.StructuredDataInput(), ak.StructuredDataInput()],
          outputs=[ak.RegressionHead()],
          max_trials=100, overwrite=True, project_name=setting+model_names[i], directory='./Dump Data')

        # Feed the tensorflow Dataset to the regressor.
        total_reg.fit([train[0], train[1]], [train[2]], epochs=1000, validation_split=0.15)
        # Convert to model
        total_model = total_reg.export_model()
        # Evaluate on validation set
        evaluation = total_reg.evaluate([val[0], val[1]], [val[2]])
        # Write loss and error to a file
        with open('./Dump Data/'+training_name + setting+'_eval_val.txt', 'a') as f:
          f.write(training_name+setting+'_'+model_names[i]+' -> ')
          f.write(str(evaluation))
          f.write('\n')
        # Save current model
        total_model.save(path+'/'+model_names[i])

        print(f"{setting} training done!!")


    # train_setting('Female', F_train_set, F_val_set, female_path)
    # train_setting('Male', M_train_set, M_val_set, male_path)
    # train_setting('Female', F_train_set, F_val_set, female_path)
    print('start training..')
    if args.setting == 'comb':
        train_setting('Combined', comb_train_set, comb_val_set, combined_path)
    elif args.setting == 'male':
        train_setting('Male', M_train_set, M_val_set, male_path)
    elif args.setting == 'female':
        train_setting('Female', F_train_set, F_val_set, female_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--OCEAN_id", type=str)
    parser.add_argument("--setting", type=str)
    args = parser.parse_args()
    main(args)
