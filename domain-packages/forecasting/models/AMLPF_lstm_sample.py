
# coding: utf-8

# ## Integrating LSTM model with Azure Machine Learning Package for Forecasting 
# 
# In this notebook, learn how to integrate LSTM model in the framework provided by Azure Machine Learning Package for Forecasting (AMLPF) to quickly build a forecasting model. 
# We will use dow jones dataset to build a model that forecasts quarterly revenue for these 30 dow jones listed companies.
# 
# #### Disclaimer: 
# This notebook is based on the ongoing development work as part of the future release of AMLPF. Therefore, please consider this as a preview of what might become available in future as part of AMLPF. 
# Further, please note that this work has currently been tested only on Windows platform.
# 
# ### Prerequisites:
# If you don't have an Azure subscription, create a free account before you begin. The following accounts and application must be set up and installed:<br/>
# * Azure Machine Learning Experimentation account.
# 
# If these three are not yet created or installed, follow the Azure Machine Learning Quickstart and Workbench installation article.
# 

# In[18]:


import warnings
warnings.filterwarnings('ignore') # comment out this statement if you do not want to suppress the warnings.

import sys, os, inspect
import numpy as np
import pandas as pd
from datetime import datetime
import json
import requests

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import ftk
ftk_root_path = (ftk.__path__)[0] # This is the path where ftk package is installed.

from ftk.pipeline import AzureMLForecastPipeline

from ftk.operationalization.dnnscorecontext import DnnScoreContext
from ftk.operationalization.dnn_score_script_helper import score_run

from ftk.dnn_utils import create_lag_lead_features
from ftk.dnn_utils import pickle_keras_models

from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense
from keras.models import load_model

print('imports done')


# In[2]:


np.random.seed(1000) # Set random seed for reproducibility.


# In[3]:


data_file_path = ftk_root_path + "\\data\\dow_jones\\dow_jones_data.tsv" # Change it depending upon where this file is stored.
num_lag_feats = 16 # Number of lag features to be used while training the model.
num_leads = 0 # Lead zero indicates current-time's value. forecast only one step at a time. 
# Note: MAPE error computation is done considering num_leads = 0. It may need to be updated to take into account num_leads > 0. It has not been done yet.
num_test_records = 4 # Keep last four records for each company in the test data.
num_lstm_au = 50 # Number of units in single lstm layer.
num_epochs = 150 # Number of epochs to fit the model. 
dj_series_freq = 'Q'


# In[4]:


# Read the dow_jones_data.
dj_df = pd.read_table(data_file_path)
print(dj_df.head())
print(dj_df.info())


# In[5]:


# Revenue has null values for some company. 'V' has been such identified company.
# In this experiment, we remove the company from the dataset instead of interpolating.
dj_df = dj_df[dj_df['company_ticker'] != 'V'] 
# Convert quarter_start field to datetime.
dj_df['quarter_start'] = pd.to_datetime(dj_df['quarter_start'])
print(dj_df.info())


# In[6]:


# Group data by company to normalize it accordingly.
grouped_data = dj_df.groupby(by='company_ticker')
cmp_to_scaler = {}
norm_dj_df = pd.DataFrame(columns=dj_df.columns) # Dataframe with quarter_start, company_ticker, normalized-revenue information.


# In[7]:


# Normalize each company's data individually and save the scaler into a dictionary to be used later.
for grp_name, grp_data in grouped_data:
    cur_grp_data = grp_data.sort_values(by=['quarter_start'])
    cur_grp_data = cur_grp_data.drop(['company_ticker', 'quarter_start'], axis=1)
    scaler = MinMaxScaler(feature_range=(0.000001, 1)) 
    norm_grp_data = scaler.fit_transform(cur_grp_data)    
    cmp_to_scaler[grp_name] = scaler
    norm_grp_df = pd.DataFrame(norm_grp_data, columns=['revenue'])
    aux_data_df = grp_data.loc[:,('quarter_start', 'company_ticker')]
    aux_data_df.reset_index(drop=True, inplace=True)
    cur_grp_norm_df = pd.concat((aux_data_df, norm_grp_df), axis=1)
    norm_dj_df = norm_dj_df.append(cur_grp_norm_df)


# In[8]:


# Create 16 lags as features for each quarterly data point (normalized revenue in previous step).
dj_reg = pd.DataFrame()
norm_grp_data = norm_dj_df.groupby(by='company_ticker')
for grp_name, grp_data in norm_grp_data:
    cur_grp_data = grp_data.sort_values(by=['quarter_start'])
    dj_reg_grp = create_lag_lead_features(cur_grp_data, ts_col='revenue', 
                    aux_cols=['company_ticker', 'quarter_start'], num_lags=num_lag_feats)
    dj_reg = dj_reg.append(dj_reg_grp)


# In[9]:


# Create list of feature column-names.
feat_cols = []
feat_tgt_cols = []
for i in range(num_lag_feats, 0, -1) :
    feat_cols.append('revenueLag' + str(i))
feat_tgt_cols.extend(feat_cols)

# Create list of target column-names. 
target_cols = ['revenueLead0']
for i in range(1, num_leads+1) :
    target_cols.append('revenueLead' + str(i))
feat_tgt_cols.extend(target_cols)


# In[10]:


# Divide the data into taining and test dataset for each company.
dj_reg_grp_data = dj_reg.groupby(by='company_ticker')
train_data = pd.DataFrame(columns=dj_reg.columns)
test_data = pd.DataFrame(columns=dj_reg.columns)

for grp_name, grp_data in dj_reg_grp_data:
    cur_grp_data = grp_data.sort_values(by=['quarter_start'])
    num_records = cur_grp_data.shape[0]
    train_data = train_data.append(pd.DataFrame(cur_grp_data.iloc[:(num_records - num_test_records),:]))
    test_data = test_data.append(pd.DataFrame(cur_grp_data.iloc[(num_records - num_test_records):,:]))


# In[11]:


# Extract features and target values for training data.
train_X = train_data[feat_cols] 
train_Y = train_data[target_cols]
"""
Formatting the input to be of the shape (number_of_samples, timesteps, number_of_features). 
For detail explanation refer to https://keras.io/layers/recurrent/.
Note: I am considering here single timestep (set to 1) and number of features to be 16. It could be specified in 
a different way (I mean, 16 timesteps instead of 1) and I plan to experiment that in future.
"""
train_X = train_X.values.reshape((train_X.shape[0], 1, train_X.shape[1]))
train_Y = train_Y.values.reshape((train_Y.shape[0], train_Y.shape[1]))
print(train_X.shape)
print(train_Y.shape)


# In[12]:


# Create a LSTM network.
model = Sequential()
model.add(LSTM(num_lstm_au, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1)) #dimension of the output vector 
model.compile(loss='mean_squared_error', optimizer='adam')


# In[13]:


# Fit network. Currently set the batch_size=1; will add more relevant information on this later.
history = model.fit(train_X, train_Y, epochs=num_epochs, batch_size=1, verbose=2, shuffle=False) 


# In[14]:


# Print model.summary.
print(model.summary())


# In[21]:


pickle_keras_models()


# In[36]:


# Initialize dataframe with column-names to hold forecasts and other relevant information.
final_test_forecasts = pd.DataFrame(columns=['company_ticker', 'quarter_start', 'actual', 'forecast'])

# Initialize dataframe with column-names to hold MAPE (Mean Absolute Percentage Error) for each company.
final_mapes = pd.DataFrame(columns=['company_ticker', 'mape'])

"""
Compute prediction of test data one company at a time. 
This is to simplify the process of scaling it back to original scale for that company.
"""
test_grp_data = test_data.groupby(by='company_ticker')

for grp_name, grp_data in test_grp_data:
    cur_grp_data = grp_data.reset_index(drop=True)
    cur_grp_data['quarter_start'] = pd.to_datetime(cur_grp_data['quarter_start'])
    cur_grp_data = cur_grp_data.sort_values(by=['quarter_start'])
    cur_final_test_fcasts = cur_grp_data[['company_ticker', 'quarter_start']]
    scaler = cmp_to_scaler[grp_name]

    test_X = cur_grp_data[feat_cols]
    test_Y = cur_grp_data[target_cols]
    test_X_reshape = test_X.values.reshape((test_X.shape[0], 1, test_X.shape[1]))
    
    dnnscoreobject = DnnScoreContext(input_scoring_data=test_X_reshape, 
                                     pipeline_execution_type='predict') # construct a context object to be used for scoring purpose.
    pipeline_lstm = AzureMLForecastPipeline([('lstm_model', model)])
    #yhat = service.score(score_context=dnnscoreobject) # invoke the web service to get predictions on the test data.
    yhat = json.loads(score_run(dnn_score_context=dnnscoreobject, pipeline=pipeline_lstm))
    print(yhat)
    inv_x_yhat = pd.concat((test_X, pd.DataFrame(yhat)), axis=1)   
    inv_x_yhat = scaler.inverse_transform(inv_x_yhat)    
    inv_x_yhat_df = pd.DataFrame(inv_x_yhat, columns=feat_tgt_cols)
    inv_yhat = inv_x_yhat_df[target_cols] 
    cur_final_test_fcasts['forecast'] = inv_yhat
        
    inv_x_y = pd.concat((test_X, pd.DataFrame(test_Y)), axis=1)
    inv_x_y = scaler.inverse_transform(inv_x_y)
    inv_x_y_df = pd.DataFrame(inv_x_y, columns=feat_tgt_cols)
    inv_y = inv_x_y_df[target_cols]
    cur_final_test_fcasts['actual'] = inv_y

    final_test_forecasts = final_test_forecasts.append(cur_final_test_fcasts)
    mape = (np.mean(np.abs((inv_y - inv_yhat)/inv_y)))*100
    print('Company: ' + grp_name + ' Test MAPE: %.3f' % mape)
    final_mapes = final_mapes.append({'company_ticker' : grp_name, 'mape' : mape}, ignore_index=True)
    

