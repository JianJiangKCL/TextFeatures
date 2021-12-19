#%%

import os
import h5py
import numpy as np
import pickle
import autokeras as ak
import tensorflow as tf
import pandas as pd

#%%

# Reading files.

# change paths here.
# meta-data


# feature data
# train_path = '../Features/Face Body/train_data.csv'
# test_path = '../Features/Face Body/test_data.csv'
# val_path = '../Features/Face Body/validation_data.csv'
# train_path = '../Features/TextualFeatures_csv/alltextual_nosenti_train.csv'
# val_path = '../Features/TextualFeatures_csv/alltextual_nosenti_valid.csv'
# test_path = '../Features/TextualFeatures_csv/alltextual_nosenti_test.csv'

def df_drop_cols(df, cols):
    for col in cols:
        try:
            df.drop(
                [col], axis=1, inplace=True)
        except:
            pass
    return df

def build_dataset(root):

    train_path = os.path.join(root, 'train_data.csv')
    test_path = os.path.join(root, 'test_data.csv')
    val_path = os.path.join(root, 'validation_data.csv')

    data_validation = pd.read_csv(val_path)
    data_train = pd.read_csv(train_path)
    data_test = pd.read_csv(test_path)

    data_test.head()

    labels = ['OPENMINDEDNESS_Z', 'CONSCIENTIOUSNESS_Z', 'EXTRAVERSION_Z', 'AGREEABLENESS_Z', 'NEGATIVEEMOTIONALITY_Z']


    #%%

    M_train = data_train[data_train['gender'] == 'M'
                         ]
    F_train = data_train[data_train['gender'] == 'F']

    M_val = data_validation[data_validation['gender'] == 'M']
    F_val = data_validation[data_validation['gender'] == 'F']

    M_test = data_test[data_test['gender'] == 'M']
    F_test = data_test[data_test['gender'] == 'F']



    #%%

    M_train = M_train.append(M_val)
    F_train = F_train.append(F_val)


    #%%

    df_train = data_train.append(data_validation)
    df_validation = data_validation
    df_test = data_test

    #%%

    # these are ground truth for each OCEAN; we will use them to evaluate the performance of the model.
    # [1,N]
    # Combined
    train_O = df_train.pop(labels[0]).to_numpy()
    train_C = df_train.pop(labels[1]).to_numpy()
    train_E = df_train.pop(labels[2]).to_numpy()
    train_A = df_train.pop(labels[3]).to_numpy()
    train_N = df_train.pop(labels[4]).to_numpy()

    val_O = df_validation.pop(labels[0]).to_numpy()
    val_C = df_validation.pop(labels[1]).to_numpy()
    val_E = df_validation.pop(labels[2]).to_numpy()
    val_A = df_validation.pop(labels[3]).to_numpy()
    val_N = df_validation.pop(labels[4]).to_numpy()

    # male train OCEAN
    M_train_O = M_train.pop(labels[0]).to_numpy()
    M_train_C = M_train.pop(labels[1]).to_numpy()
    M_train_E = M_train.pop(labels[2]).to_numpy()
    M_train_A = M_train.pop(labels[3]).to_numpy()
    M_train_N= M_train.pop(labels[4]).to_numpy()

    # female train OCEAN
    F_train_O = F_train.pop(labels[0]).to_numpy()
    F_train_C = F_train.pop(labels[1]).to_numpy()
    F_train_E = F_train.pop(labels[2]).to_numpy()
    F_train_A = F_train.pop(labels[3]).to_numpy()
    F_train_N= F_train.pop(labels[4]).to_numpy()

    # male validation OCEAN

    M_val_O = M_val.pop(labels[0]).to_numpy()
    M_val_C = M_val.pop(labels[1]).to_numpy()
    M_val_E = M_val.pop(labels[2]).to_numpy()
    M_val_A = M_val.pop(labels[3]).to_numpy()
    M_val_N = M_val.pop(labels[4]).to_numpy()

    # female validation OCEAN

    F_val_O = F_val.pop(labels[0]).to_numpy()
    F_val_C = F_val.pop(labels[1]).to_numpy()
    F_val_E = F_val.pop(labels[2]).to_numpy()
    F_val_A = F_val.pop(labels[3]).to_numpy()
    F_val_N = F_val.pop(labels[4]).to_numpy()

    #%%


    #%%

    data_cols = [ 'ID_y', 'minute']
    drop_cols = []

    train_data = df_train[data_cols]
    val_data   = df_validation[data_cols]
    test_data  = df_test[data_cols]


    M_train_data = M_train[data_cols]
    F_train_data = F_train[data_cols]
    M_val_data = M_val[data_cols]
    F_val_data = F_val[data_cols]
    M_test_data = M_test[data_cols]
    F_test_data = F_test[data_cols]

    #%%
    # following train data only has the feature data

    data_cols = ['ID_y', 'minute', 'session', 'gender', 'age','Unnamed: 0', 'Video', 'Unnamed: 0.1']

    # combined
    train_com = df_drop_cols(df_train, data_cols+drop_cols)
    val_com = df_drop_cols(df_validation, data_cols+drop_cols)
    test_com = df_drop_cols(df_test, data_cols+drop_cols)


    Fb_drop_cols = [(f'{i}_fb') for i in range(0, 552)]
    Bt_drop_cols = [(f'{i}_bt') for i in range(0, 512)]

    train_comb_bt = np.array(train_com.drop([*Fb_drop_cols], axis=1))
    train_comb_fb = np.array(train_com.drop([*Bt_drop_cols], axis=1))

    val_comb_bt =   np.array(val_com.drop([*Fb_drop_cols], axis=1))
    val_comb_fb = np.array(val_com.drop([*Bt_drop_cols], axis=1))

    test_comb_bt = np.array(test_com.drop([*Fb_drop_cols], axis=1))
    test_comb_fb = np.array(test_com.drop([*Bt_drop_cols], axis=1))


    # Gender
    # M_train = M_train.drop(data_cols+drop_cols, axis = 1)
    # M_val = M_val.drop(data_cols+drop_cols, axis = 1)
    # M_test = M_test.drop(data_cols+drop_cols, axis = 1)
    #
    # F_train = F_train.drop(data_cols+drop_cols, axis = 1)
    # F_val = F_val.drop(data_cols+drop_cols, axis = 1)
    # F_test = F_test.drop(data_cols+drop_cols, axis = 1)

    #%%

    # Combined
    train_data.reset_index(drop = True, inplace = True)
    val_data.reset_index(drop = True, inplace = True)
    test_data.reset_index(drop = True, inplace = True)

    # Gender
    # M_train.reset_index(drop=True, inplace=True)
    # M_train_data.reset_index(drop=True, inplace=True)
    # M_test.reset_index(drop=True, inplace=True)
    # M_test_data.reset_index(drop = True, inplace = True)
    # M_val.reset_index(drop=True, inplace=True)
    # M_val_data.reset_index(drop=True, inplace=True)
    #
    # F_train.reset_index(drop=True, inplace=True)
    # F_train_data.reset_index(drop=True, inplace=True)
    # F_val.reset_index(drop=True, inplace=True)
    # F_val_data.reset_index(drop=True, inplace=True)
    # F_test.reset_index(drop=True, inplace=True)
    # F_test_data.reset_index(drop = True, inplace = True)


    OCEAN_models = ['Model_O', 'Model_C', 'Model_E', 'Model_A', 'Model_N']


    # GEnder
    # M_train_np = np.array(M_train)
    # M_test_np  = np.array(M_test)
    # M_val_np   = np.array(M_val)
    #
    #
    # F_train_np = np.array(F_train)
    # F_val_np   = np.array(F_val)
    # F_test_np  = np.array(F_test)



    #%%
    def tmp(m1, m2, gt):
        return (m1, m2 ,gt)
    # Combined
    comb_O_train_set = tmp(train_comb_fb, train_comb_bt, train_O)
    comb_C_train_set = tmp(train_comb_fb, train_comb_bt, train_C)
    comb_E_train_set = tmp(train_comb_fb, train_comb_bt, train_E)
    comb_A_train_set = tmp(train_comb_fb, train_comb_bt, train_A)
    comb_N_train_set = tmp(train_comb_fb, train_comb_bt, train_N)

    comb_O_validation_set = tmp(val_comb_fb, val_comb_bt, val_O)
    comb_C_validation_set = tmp(val_comb_fb, val_comb_bt, val_C)
    comb_E_validation_set = tmp(val_comb_fb, val_comb_bt, val_E)
    comb_A_validation_set = tmp(val_comb_fb, val_comb_bt, val_A)
    comb_N_validation_set = tmp(val_comb_fb, val_comb_bt, val_N)


    # # Gender
    # maleO_train_set = tmp(M_train_np, M_train_O)
    # maleC_train_set = tmp(M_train_np, M_train_C)
    # maleE_train_set = tmp(M_train_np, M_train_E)
    # maleA_train_set = tmp(M_train_np, M_train_A)
    # maleN_train_set = tmp(M_train_np, M_train_N)
    #
    # femaleO_train_set = tmp(F_train_np, F_train_O)
    # femaleC_train_set = tmp(F_train_np, F_train_C)
    # femaleE_train_set = tmp(F_train_np, F_train_E)
    # femaleA_train_set = tmp(F_train_np, F_train_A)
    # femaleN_train_set = tmp(F_train_np, F_train_N)
    #
    # maleO_validation_set = tmp(M_val_np, M_val_O)
    # maleC_validation_set = tmp(M_val_np, M_val_C)
    # maleE_validation_set = tmp(M_val_np, M_val_E)
    # maleA_validation_set = tmp(M_val_np, M_val_A)
    # maleN_validation_set = tmp(M_val_np, M_val_N)
    #
    # femaleO_validation_set = tmp(F_val_np, F_val_O)
    # femaleC_validation_set = tmp(F_val_np, F_val_C)
    # femaleE_validation_set = tmp(F_val_np, F_val_E)
    # femaleA_validation_set = tmp(F_val_np, F_val_A)
    # femaleN_validation_set = tmp(F_val_np, F_val_N)


    #%%

    # combined
    comb_train_set = [comb_O_train_set,
                      comb_C_train_set,
                      comb_E_train_set,
                      comb_A_train_set,
                      comb_N_train_set]

    comb_validation_set = [comb_O_validation_set,
                           comb_C_validation_set,
                           comb_E_validation_set,
                           comb_A_validation_set,
                           comb_N_validation_set]

    # Gender
    # male_train_set =      [maleO_train_set,
    #                        maleC_train_set,
    #                        maleE_train_set,
    #                        maleA_train_set,
    #                        maleN_train_set]
    # male_validation_set = [maleO_validation_set,
    #                        maleC_validation_set,
    #                        maleE_validation_set,
    #                        maleA_validation_set,
    #                        maleN_validation_set]
    #
    # female_train_set =      [femaleO_train_set,
    #                          femaleC_train_set,
    #                          femaleE_train_set,
    #                          femaleA_train_set,
    #                          femaleN_train_set]
    # female_validation_set = [femaleO_validation_set,
    #                          femaleC_validation_set,
    #                          femaleE_validation_set,
    #                          femaleA_validation_set,
    #                          femaleN_validation_set]
    return comb_train_set, comb_validation_set

comb_train_set, comb_validation_set = build_dataset('multi_modality_csv')

# Bt_comb_train_set, Bt_comb_validation_set, Bt_male_train_set, Bt_male_validation_set, Bt_female_train_set, Bt_female_validation_set = build_dataset('../Features/Bert', 'Bert')

# Facebody_comb_train_set = build_dataset()
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
    try:
        os.mkdir(folder)
    except OSError as error:
        print(error)

#%% md

# Training

#%% md

## Combined

#%%

"""# Train
Currently:  
Validation split: 0.15  
Epochs: 1000  
Trials: 100  
## Combined models
"""
print("Combined TRAINING BEGINS")
# iter for 5 OCEAN dataset
for i in range(5):
    #train_comb_fb, train_comb_bt, train_O
    # comb_train_set, comb_validation_set


    train = comb_train_set[i]
    val = comb_train_set[i]

    print(model_names[i])
    # Define a regressor
    # total_reg = ak.StructuredDataRegressor(max_trials=100, overwrite=True, project_name = 'comb' +model_names[i], directory = './Dump Data')
    # Fb_train_ = Fb_train.element_spec[0]
    total_reg= ak.AutoModel(
      # by default, RegrssionHead uses MSE
      inputs=[ak.StructuredDataInput(), ak.StructuredDataInput()],
      outputs=[ak.RegressionHead()],
      max_trials=100, overwrite=True, project_name='comb'+model_names[i], directory='./Dump Data')

    # Feed the tensorflow Dataset to the regressor.
    total_reg.fit([train[0], train[1]], [train[2]], epochs=1000, validation_split=0.15)
    # Convert to model
    total_model = total_reg.export_model()
    # Evaluate on validation set
    evaluation = total_reg.evaluate([val[0], val[1]],[val[2]])
    # Write loss and error to a file
    with open('./Dump Data/'+training_name+'combined_eval_val.txt', 'a') as f:
      f.write(training_name+'combined_'+model_names[i]+' -> ')
      f.write(str(evaluation))
      f.write('\n')
    # Save current model
    total_model.save(combined_path+'/'+model_names[i])

print("Combined training done!!")


#%% md

## Gender

#%%

"""# Train
Currently:  
Validation split: 0.15  
Epochs: 1000  
Trials: 100  
## Age models
"""
# print("Male TRAINING BEGINS")
# for i in range(5):
#   train = Fb_male_train_set[i]
#   val = Fb_male_validation_set[i]
#   print(model_names[i])
#   # Define a regressor
#   total_reg = ak.StructuredDataRegressor(max_trials=100, overwrite=True,project_name = training_name+'male_'+model_names[i], directory = './Dump Data')
#   # Feed the tensorflow Dataset to the regressor.
#   total_reg.fit(train, epochs=1000, validation_split=0.15)
#   # Convert to model
#   total_model = total_reg.export_model()
#   # Evaluate on validation set
#   evaluation = total_reg.evaluate(val)
#   # Write loss and error to a file
#   with open('./Dump Data/'+training_name+'male_eval_val.txt', 'a') as f:
#       f.write(training_name+'male_'+model_names[i]+' -> ')
#       f.write(str(evaluation)
#       f.write('\n')
#   # Save current model
#   total_model.save(male_path+'/'+model_names[i])
#
# # """## Female models"""
# print("Female TRAINING BEGINS")
#
# for i in range(5):
#   train = Fb_female_train_set[i]
#   val = Fb_female_validation_set[i]
#   # Define a regressor
#   total_reg = ak.StructuredDataRegressor(max_trials=100, overwrite=True,project_name = training_name+'female_'+model_names[i], directory = './Dump Data')
#   # Feed the tensorflow Dataset to the regressor.
#   total_reg.fit(train, epochs=1000, validation_split=0.15)
#   # Convert to model
#   total_model = total_reg.export_model()
#   # Evaluate on validation set
#   evaluation = total_reg.evaluate(val)
#   # Write loss and error to a file
#   with open('./Dump Data/'+training_name+'female_eval_val.txt', 'a') as f:
#       f.write(training_name+'female_'+model_names[i]+' -> ')
#       f.write(str(evaluation)
#       f.write('\n')
#   # Save current model
#   total_model.save(female_path+'/'+model_names[i])
# print("Gender TRAINING DONE")

#%%


