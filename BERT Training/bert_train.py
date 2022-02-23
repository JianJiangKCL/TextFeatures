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
# data_train = pd.read_csv('./Features/BERT/bert_train.csv')
# data_validation = pd.read_csv('./Features/BERT/bert_validation.csv')
# data_test = pd.read_csv('./Features/BERT/bert_test.csv')
data_train = pd.read_csv('bert_train.csv')
data_validation = pd.read_csv('bert_validation.csv')
data_test = pd.read_csv('bert_test.csv')
"""# Data Handling"""
data_train.drop(['Unnamed: 0'], axis = 1, inplace = True)
data_validation.drop(['Unnamed: 0'], axis = 1, inplace = True)
data_test.drop(['Unnamed: 0'], axis = 1, inplace = True)




#%%

data_train.head()

#%%

labels = ['OPENMINDEDNESS_Z', 'CONSCIENTIOUSNESS_Z', 'EXTRAVERSION_Z', 'AGREEABLENESS_Z', 'NEGATIVEEMOTIONALITY_Z']


#%% md

# Separating male, female, young and old

#%%

M_train = data_train[data_train['GENDER'] == 'M']
F_train = data_train[data_train['GENDER'] == 'F']

M_val = data_validation[data_validation['GENDER'] == 'M']
F_val = data_validation[data_validation['GENDER'] == 'F']

M_test = data_test[data_test['GENDER'] == 'M']
F_test = data_test[data_test['GENDER'] == 'F']

#%%

Y_train = data_train[data_train['AGE'] <= 30]
O_train = data_train[data_train['AGE'] > 30]

Y_val = data_validation[data_validation['AGE'] <= 30]
O_val = data_validation[data_validation['AGE'] > 30]

Y_test = data_test[data_test['AGE'] <= 30]
O_test = data_test[data_test['AGE'] > 30]

#%% md

# Appending validation data to train foro training with both sets

#%%

M_train = M_train.append(M_val)
F_train = F_train.append(F_val)

#%%

Y_train = Y_train.append(Y_val)
O_train = O_train.append(O_val)

#%%

df_train = data_train.append(data_validation)
df_validation = data_validation
df_test = data_test

#%% md

# Data handling

# Here we just pop out the labels from the main dataframe into separate dataframes for each set.

#%%

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



# male train OCEAN
Y_train_O = Y_train.pop(labels[0]).to_numpy()
Y_train_C = Y_train.pop(labels[1]).to_numpy()
Y_train_E = Y_train.pop(labels[2]).to_numpy()
Y_train_A = Y_train.pop(labels[3]).to_numpy()
Y_train_N=  Y_train.pop(labels[4]).to_numpy()

# female train OCEAN
O_train_O = O_train.pop(labels[0]).to_numpy()
O_train_C = O_train.pop(labels[1]).to_numpy()
O_train_E = O_train.pop(labels[2]).to_numpy()
O_train_A = O_train.pop(labels[3]).to_numpy()
O_train_N=  O_train.pop(labels[4]).to_numpy()

# male validation OCEAN

Y_val_O = Y_val.pop(labels[0]).to_numpy()
Y_val_C = Y_val.pop(labels[1]).to_numpy()
Y_val_E = Y_val.pop(labels[2]).to_numpy()
Y_val_A = Y_val.pop(labels[3]).to_numpy()
Y_val_N = Y_val.pop(labels[4]).to_numpy()

# female validation OCEAN

O_val_O = O_val.pop(labels[0]).to_numpy()
O_val_C = O_val.pop(labels[1]).to_numpy()
O_val_E = O_val.pop(labels[2]).to_numpy()
O_val_A = O_val.pop(labels[3]).to_numpy()
O_val_N = O_val.pop(labels[4]).to_numpy()

#%% md

## Here we separate the data i.e only ID and minutes in this case.

#%%

data_cols = ['ID_y', 'minutes']
drop_cols = []

train_data = df_train[data_cols]
val_data   = df_validation[data_cols]
test_data  = df_test[data_cols]

Y_train_data = Y_train[data_cols]
Y_val_data   = Y_val[data_cols]
Y_test_data  = Y_test[data_cols]

O_train_data = O_train[data_cols]
O_val_data   = O_val[data_cols]
O_test_data  = O_test[data_cols]

M_train_data = M_train[data_cols]
F_train_data = F_train[data_cols]
M_val_data = M_val[data_cols]
F_val_data = F_val[data_cols]
M_test_data = M_test[data_cols]
F_test_data = F_test[data_cols]

#%%

print(M_train_data.index)
print(F_train_data.index)


#%% md

## Popping the data columns, to leave only the BERT features to later convert to tensors

#%%

data_cols = [ 'ID_y', 'GENDER', 'AGE', 'minutes']
# combined
train_com = df_train.drop(data_cols+drop_cols, axis = 1)
val_com   = df_validation.drop(data_cols+drop_cols, axis = 1)
test_com  = df_test.drop(data_cols+drop_cols, axis = 1)



# Age
Y_train = Y_train.drop(data_cols+drop_cols, axis = 1)
Y_val   = Y_val.drop(data_cols+drop_cols, axis = 1)
Y_test  = Y_test.drop(data_cols+drop_cols, axis = 1)

O_train = O_train.drop(data_cols+drop_cols, axis = 1)
O_val   = O_val.drop(data_cols+drop_cols, axis = 1)
O_test  = O_test.drop(data_cols+drop_cols, axis = 1)



# Gender
M_train = M_train.drop(data_cols+drop_cols, axis = 1)
M_val = M_val.drop(data_cols+drop_cols, axis = 1)
M_test = M_test.drop(data_cols+drop_cols, axis = 1)

F_train = F_train.drop(data_cols+drop_cols, axis = 1)
F_val = F_val.drop(data_cols+drop_cols, axis = 1)
F_test = F_test.drop(data_cols+drop_cols, axis = 1)



#%% md

## Reseting indices to make it easier to combine later on

#%%

# Comvined
train_data.reset_index(drop = True, inplace = True)
val_data.reset_index(drop = True, inplace = True)
test_data.reset_index(drop = True, inplace = True)


# Age

Y_train.reset_index(drop=True, inplace=True)
Y_train_data.reset_index(drop=True, inplace=True)
Y_test.reset_index(drop=True, inplace=True)
Y_test_data.reset_index(drop = True, inplace = True)
Y_val.reset_index(drop=True, inplace=True)
Y_val_data.reset_index(drop=True, inplace=True)

O_train.reset_index(drop=True, inplace=True)
O_train_data.reset_index(drop=True, inplace=True)
O_val.reset_index(drop=True, inplace=True)
O_val_data.reset_index(drop=True, inplace=True)
O_test.reset_index(drop=True, inplace=True)
O_test_data.reset_index(drop = True, inplace = True)

# Gender
M_train.reset_index(drop=True, inplace=True)
M_train_data.reset_index(drop=True, inplace=True)
M_test.reset_index(drop=True, inplace=True)
M_test_data.reset_index(drop = True, inplace = True)
M_val.reset_index(drop=True, inplace=True)
M_val_data.reset_index(drop=True, inplace=True)

F_train.reset_index(drop=True, inplace=True)
F_train_data.reset_index(drop=True, inplace=True)
F_val.reset_index(drop=True, inplace=True)
F_val_data.reset_index(drop=True, inplace=True)
F_test.reset_index(drop=True, inplace=True)
F_test_data.reset_index(drop = True, inplace = True)

#%%

F_train.head()

#%%

train_C.shape

#%% md

## Converting to numpy arrays to convert to tensors

#%%


OCEAN_models = ['Model_O', 'Model_C', 'Model_E', 'Model_A', 'Model_N']
# Combined
train_com_np = np.array(train_com)
val_com_np = np.array(val_com)
test_com_np = np.array(test_com)

# Age
Y_train_np = np.array(Y_train)
Y_test_np  = np.array(Y_test)
Y_val_np   = np.array(Y_val)


O_train_np = np.array(O_train)
O_val_np   = np.array(O_val)
O_test_np  = np.array(O_test)

# GEnder
M_train_np = np.array(M_train)
M_test_np  = np.array(M_test)
M_val_np   = np.array(M_val)


F_train_np = np.array(F_train)
F_val_np   = np.array(F_val)
F_test_np  = np.array(F_test)



#%% md

## Converting to tensor slices

#%%

# Combined
comb_O_train_set = tf.data.Dataset.from_tensor_slices((train_com_np, train_O))
comb_C_train_set = tf.data.Dataset.from_tensor_slices((train_com_np, train_C))
comb_E_train_set = tf.data.Dataset.from_tensor_slices((train_com_np, train_E))
comb_A_train_set = tf.data.Dataset.from_tensor_slices((train_com_np, train_A))
comb_N_train_set = tf.data.Dataset.from_tensor_slices((train_com_np, train_N))

comb_O_validation_set = tf.data.Dataset.from_tensor_slices((val_com_np, val_O))
comb_C_validation_set = tf.data.Dataset.from_tensor_slices((val_com_np, val_C))
comb_E_validation_set = tf.data.Dataset.from_tensor_slices((val_com_np, val_E))
comb_A_validation_set = tf.data.Dataset.from_tensor_slices((val_com_np, val_A))
comb_N_validation_set = tf.data.Dataset.from_tensor_slices((val_com_np, val_N))


# Age

youngO_train_set      = tf.data.Dataset.from_tensor_slices((Y_train_np, Y_train_O))
youngC_train_set      = tf.data.Dataset.from_tensor_slices((Y_train_np, Y_train_C))
youngE_train_set      = tf.data.Dataset.from_tensor_slices((Y_train_np, Y_train_E))
youngA_train_set      = tf.data.Dataset.from_tensor_slices((Y_train_np, Y_train_A))
youngN_train_set      = tf.data.Dataset.from_tensor_slices((Y_train_np, Y_train_N))
youngO_validation_set = tf.data.Dataset.from_tensor_slices((Y_val_np, Y_val_O))
youngC_validation_set = tf.data.Dataset.from_tensor_slices((Y_val_np, Y_val_C))
youngE_validation_set = tf.data.Dataset.from_tensor_slices((Y_val_np, Y_val_E))
youngA_validation_set = tf.data.Dataset.from_tensor_slices((Y_val_np, Y_val_A))
youngN_validation_set = tf.data.Dataset.from_tensor_slices((Y_val_np, Y_val_N))

oldO_validation_set = tf.data.Dataset.from_tensor_slices((O_val_np, O_val_O))
oldC_validation_set = tf.data.Dataset.from_tensor_slices((O_val_np, O_val_C))
oldE_validation_set = tf.data.Dataset.from_tensor_slices((O_val_np, O_val_E))
oldA_validation_set = tf.data.Dataset.from_tensor_slices((O_val_np, O_val_A))
oldN_validation_set = tf.data.Dataset.from_tensor_slices((O_val_np, O_val_N))
oldO_train_set      = tf.data.Dataset.from_tensor_slices((O_train_np, O_train_O))
oldC_train_set      = tf.data.Dataset.from_tensor_slices((O_train_np, O_train_C))
oldE_train_set      = tf.data.Dataset.from_tensor_slices((O_train_np, O_train_E))
oldA_train_set      = tf.data.Dataset.from_tensor_slices((O_train_np, O_train_A))
oldN_train_set      = tf.data.Dataset.from_tensor_slices((O_train_np, O_train_N))


# Gender
maleO_train_set = tf.data.Dataset.from_tensor_slices((M_train_np, M_train_O))
maleC_train_set = tf.data.Dataset.from_tensor_slices((M_train_np, M_train_C))
maleE_train_set = tf.data.Dataset.from_tensor_slices((M_train_np, M_train_E))
maleA_train_set = tf.data.Dataset.from_tensor_slices((M_train_np, M_train_A))
maleN_train_set = tf.data.Dataset.from_tensor_slices((M_train_np, M_train_N))

femaleO_train_set = tf.data.Dataset.from_tensor_slices((F_train_np, F_train_O))
femaleC_train_set = tf.data.Dataset.from_tensor_slices((F_train_np, F_train_C))
femaleE_train_set = tf.data.Dataset.from_tensor_slices((F_train_np, F_train_E))
femaleA_train_set = tf.data.Dataset.from_tensor_slices((F_train_np, F_train_A))
femaleN_train_set = tf.data.Dataset.from_tensor_slices((F_train_np, F_train_N))

maleO_validation_set = tf.data.Dataset.from_tensor_slices((M_val_np, M_val_O))
maleC_validation_set = tf.data.Dataset.from_tensor_slices((M_val_np, M_val_C))
maleE_validation_set = tf.data.Dataset.from_tensor_slices((M_val_np, M_val_E))
maleA_validation_set = tf.data.Dataset.from_tensor_slices((M_val_np, M_val_A))
maleN_validation_set = tf.data.Dataset.from_tensor_slices((M_val_np, M_val_N))

femaleO_validation_set = tf.data.Dataset.from_tensor_slices((F_val_np, F_val_O))
femaleC_validation_set = tf.data.Dataset.from_tensor_slices((F_val_np, F_val_C))
femaleE_validation_set = tf.data.Dataset.from_tensor_slices((F_val_np, F_val_E))
femaleA_validation_set = tf.data.Dataset.from_tensor_slices((F_val_np, F_val_A))
femaleN_validation_set = tf.data.Dataset.from_tensor_slices((F_val_np, F_val_N))


#%% md

## Defining the train, validation and test sets for all to loop through while training

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

# Age
young_train_set =     [youngO_train_set,
                       youngC_train_set,
                       youngE_train_set,
                       youngA_train_set,
                       youngN_train_set]
young_validation_set =[youngO_validation_set,
                       youngC_validation_set,
                       youngE_validation_set,
                       youngA_validation_set,
                       youngN_validation_set]

old_train_set =         [oldO_train_set,
                         oldC_train_set,
                         oldE_train_set,
                         oldA_train_set,
                         oldN_train_set]
old_validation_set =    [oldO_validation_set,
                         oldC_validation_set,
                         oldE_validation_set,
                         oldA_validation_set,
                         oldN_validation_set]

# Gender
male_train_set =      [maleO_train_set,
                       maleC_train_set,
                       maleE_train_set,
                       maleA_train_set,
                       maleN_train_set]
male_validation_set = [maleO_validation_set,
                       maleC_validation_set,
                       maleE_validation_set,
                       maleA_validation_set,
                       maleN_validation_set]

female_train_set =      [femaleO_train_set,
                         femaleC_train_set,
                         femaleE_train_set,
                         femaleA_train_set,
                         femaleN_train_set]
female_validation_set = [femaleO_validation_set,
                         femaleC_validation_set,
                         femaleE_validation_set,
                         femaleA_validation_set,
                         femaleN_validation_set]

model_names = ['Model_O','Model_C','Model_E','Model_A','Model_N']


#%% md

## Making folders to save models and dump training data

#%%

dump = './Dump Data'
path = './BERT Models'
combined_path = './BERT Models/Combined'
age_path = './BERT Models/Age'
gender_path = './BERT Models/Gender'
male_path = './BERT Models/Gender/Male'
female_path = './BERT Models/Gender/Female'
old_path = './BERT Models/Age/Old'
young_path = './BERT Models/Age/Young'

folders = [ dump
,path
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

# Combined Training

#%%

"""# Train
Currently:  
Validation split: 0.15  
Epochs: 1000  
Trials: 100  
## Combined models
"""
print("Combined TRAINING BEGINS")
for i in range(5):
  train = comb_train_set[i]
  val = comb_validation_set[i]
  print(model_names[i])
  # Define a regressor
  total_reg = ak.StructuredDataRegressor(max_trials=100, overwrite=True,project_name = 'bert_combined_'+model_names[i], directory = './Dump Data')
  # Feed the tensorflow Dataset to the regressor.
  total_reg.fit(train, epochs=1000, validation_split=0.15)
  # Convert to model
  total_model = total_reg.export_model()
  # Evaluate on validation set
  evaluation = total_reg.evaluate(val)
  # Write loss and error to a file
  with open('./Dump Data/bert_combined_eval_val.txt', 'a') as f:
      f.write('bert_combined_'+model_names[i]+' -> ')
      f.write(str(evaluation))
      f.write('\n')
  # Save current model
  total_model.save(combined_path+'/'+model_names[i])

print("Combined training done!!")

#%% md

# Age training

#%%

"""# Train
Currently:  
Validation split: 0.15  
Epochs: 1000  
Trials: 100  
## Age models
"""
print("Young TRAINING BEGINS")
for i in range(5):
  train = young_train_set[i]
  val = young_validation_set[i]
  print(model_names[i])
  # Define a regressor
  total_reg = ak.StructuredDataRegressor(max_trials=100, overwrite=True,project_name = 'bert_young_'+model_names[i], directory = './Dump Data')
  # Feed the tensorflow Dataset to the regressor.
  total_reg.fit(train, epochs=1000, validation_split=0.15)
  # Convert to model
  total_model = total_reg.export_model()
  # Evaluate on validation set
  evaluation = total_reg.evaluate(val)
  # Write loss and error to a file
  with open('./Dump Data/bert_young_eval_val.txt', 'a') as f:
      f.write('bert_young_'+model_names[i]+' -> ')
      f.write(str(evaluation))
      f.write('\n')
  # Save current model
  total_model.save(young_path+'/'+model_names[i])

# """## Old models"""
print("Old TRAINING BEGINS")

for i in range(5):
  train = old_train_set[i]
  val = old_validation_set[i]
  # Define a regressor
  total_reg = ak.StructuredDataRegressor(max_trials=100, overwrite=True,project_name = 'bert_old_'+model_names[i], directory = './Dump Data')
  # Feed the tensorflow Dataset to the regressor.
  total_reg.fit(train, epochs=1000, validation_split=0.15)
  # Convert to model
  total_model = total_reg.export_model()
  # Evaluate on validation set
  evaluation = total_reg.evaluate(val)
  # Write loss and error to a file
  with open('./Dump Data/bert_old_eval_val.txt', 'a') as f:
      f.write('bert_old_'+model_names[i]+' -> ')
      f.write(str(evaluation))
      f.write('\n')
  # Save current model
  total_model.save(old_path+'/'+model_names[i])
print("Age TRAINING DONE")

#%% md

# Gender training

#%%

"""# Train
Currently:  
Validation split: 0.15  
Epochs: 1000  
Trials: 100  
## Age models
"""
print("Male TRAINING BEGINS")
for i in range(5):
  train = male_train_set[i]
  val = male_validation_set[i]
  print(model_names[i])
  # Define a regressor
  total_reg = ak.StructuredDataRegressor(max_trials=100, overwrite=True,project_name = 'bert_male_'+model_names[i], directory = './Dump Data')
  # Feed the tensorflow Dataset to the regressor.
  total_reg.fit(train, epochs=1000, validation_split=0.15)
  # Convert to model
  total_model = total_reg.export_model()
  # Evaluate on validation set
  evaluation = total_reg.evaluate(val)
  # Write loss and error to a file
  with open('./Dump Data/bert_male_eval_val.txt', 'a') as f:
      f.write('bert_male_'+model_names[i]+' -> ')
      f.write(str(evaluation))
      f.write('\n')
  # Save current model
  total_model.save(male_path+'/'+model_names[i])

# """## Female models"""
print("Female TRAINING BEGINS")

for i in range(5):
  train = female_train_set[i]
  val = female_validation_set[i]
  # Define a regressor
  total_reg = ak.StructuredDataRegressor(max_trials=100, overwrite=True,project_name = 'bert_female_'+model_names[i], directory = './Dump Data')
  # Feed the tensorflow Dataset to the regressor.
  total_reg.fit(train, epochs=1000, validation_split=0.15)
  # Convert to model
  total_model = total_reg.export_model()
  # Evaluate on validation set
  evaluation = total_reg.evaluate(val)
  # Write loss and error to a file
  with open('./Dump Data/bert_female_eval_val.txt', 'a') as f:
      f.write('bert_female_'+model_names[i]+' -> ')
      f.write(str(evaluation))
      f.write('\n')
  # Save current model
  total_model.save(female_path+'/'+model_names[i])
print("Gender TRAINING DONE")
