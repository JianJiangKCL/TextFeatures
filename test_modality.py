#%%

import os
import h5py
import numpy as np
import pickle
import autokeras as ak
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from train_modality import build_dataset

comb_test_set, M_test_set, F_test_set = build_dataset('multi_modality_csv', 'test', 'gender')

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

def test_setting(setting, test_set, path):
    print(f"{setting} TRAINING BEGINS")
    # iter for 5 OCEAN dataset
    for i in range(0, 5):

        test = test_set[i]
        print(model_names[i])
        # total_model.save(path + '/' + model_names[i])
        loaded_model = load_model(f"{path}/{model_names[i]}/", custom_objects=ak.CUSTOM_OBJECTS)
        # loaded_model = load_model('saved_model.pb')
        evaluation = loaded_model.evaluate([test[0], test[1]], [test[2]])
        pred = loaded_model.predict([test[0], test[1]])
        # Write loss and error to a file
        with open('./Dump Data/'+training_name + setting+'_eval_test.txt', 'a') as f:
          f.write(training_name+setting+'_'+model_names[i]+' -> ')
          f.write(str(evaluation))
          f.write('\n')

        with open('./Dump Data/'+training_name + setting+'_eval_test_pred.txt', 'a') as f:
          f.write(training_name+setting+'_'+model_names[i]+' -> ')
          f.write(str(pred))
          f.write('\n')

    print(f"{setting} test done!!")

#
test_setting('Male', M_test_set, male_path)
test_setting('Female', F_test_set, female_path)
test_setting('Combined', comb_test_set, combined_path)
