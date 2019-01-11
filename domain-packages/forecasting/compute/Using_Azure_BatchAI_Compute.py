
# coding: utf-8

# # Compute backends in AMLPF
# 
# This notebook showcases the `compute` sub-package in AMLPF. The `compute` modules abstract several local or distributed backends
# available in Python such as [Joblib](https://joblib.readthedocs.io/en/latest/), [concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html) or [Azure Batch AI](https://docs.microsoft.com/en-us/azure/batch-ai/overview). 

# ### Prerequisites: Setup and configure AML environment
# This notebook requires that a AML Python SDK is setup. Make sure you go through the [00. Installation and Configuration](https://github.com/Azure/ViennaDocs/blob/master/PrivatePreview/notebooks/00.configuration.ipynb) to do so if none is present. 

# ### Import FTK 
# **NOTE**: If Pandas or other core library errors are encountered. Refresh the environment by reninstall the packages.
# This can be done by activating the kernel environment the notebook is being run under and then running the following command: `python.exe -m pip install -U --force-reinstall pandas==0.20.3`

# In[ ]:


import warnings

# Suppress warnings
warnings.filterwarnings("ignore") 

import os
import urllib
import pkg_resources
import numpy as np
import pandas as pd
import math
import time
import importlib
from datetime import timedelta
from random import randint
from scipy import stats

from sklearn.datasets import load_diabetes
from sklearn.model_selection import (TimeSeriesSplit, cross_val_score)
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

azureml_spec = importlib.util.find_spec("azureml.core")
if azureml_spec is not None:
    import azureml.core
    from azureml.core import Workspace, Run, Datastore
    from azureml.core.runconfig import RunConfiguration       
else:
    print('AzureML not found')
    raise

from ftk import TimeSeriesDataFrame, ForecastDataFrame, AzureMLForecastPipeline
from ftk.compute import ComputeBase, JoblibParallelCompute, DaskDistributedCompute, AMLBatchAICompute, Scheduler
from ftk.data import load_dow_jones_dataset
from ftk.transforms import LagLeadOperator, TimeSeriesImputer, TimeIndexFeaturizer, DropColumns
from ftk.transforms.grain_index_featurizer import GrainIndexFeaturizer
from ftk.models import Arima, SeasonalNaive, Naive, RegressionForecaster, BestOfForecaster
from ftk.models.forecaster_union import ForecasterUnion
from ftk.model_selection import TSGridSearchCV, RollingOriginValidator
from ftk.ts_utils import last_n_periods_split

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)
print("All imports done")


# ### Load dataset and engineer features
# 
# Load the `Dominicks Orange Juice` dataset and perform feature engineering using available tranformers in AMLPF.

# In[ ]:


# Load in the Dominicks OJ  dataset
csv_path = pkg_resources.resource_filename('ftk', 'data/dominicks_oj/dominicks_oj.csv')
whole_df = pd.read_csv(csv_path, low_memory = False)

# Adjust 'Quantity` to be absolute value
def expround(x):
    return math.floor(math.exp(x) + 0.5)
whole_df['Quantity'] = whole_df['logmove'].apply(expround)

# Create new datetime columns containing the start and end of each week period
weekZeroStart = pd.to_datetime('1989-09-07 00:00:00')
weekZeroEnd = pd.to_datetime('1989-09-13 23:59:59')
whole_df['WeekFirstDay'] = whole_df['week'].apply(lambda n: weekZeroStart + timedelta(weeks=n))
whole_df['WeekLastDay'] = whole_df['week'].apply(lambda n: weekZeroEnd + timedelta(weeks=n))
whole_df[['store','brand','WeekLastDay','Quantity']].head()

# Create a TimeSeriesDataFrame
# 'WeekLastDay' is the time index, 'Store' and 'brand'
# combinations label the grain 
whole_tsdf = TimeSeriesDataFrame(whole_df, 
                                 grain_colnames=['store', 'brand'],
                                 time_colname='WeekLastDay', 
                                 ts_value_colname='Quantity',
                                 group_colnames='store')

# sort and slice
whole_tsdf.sort_index(inplace=True)

# Get sales of dominick's brand orange juice from store 2 during summer 1990
whole_tsdf.loc[pd.IndexSlice['1990-06':'1990-09', 2, 'dominicks'], ['Quantity']]

train_tsdf, test_tsdf = last_n_periods_split(whole_tsdf, 40)

# Use a TimeSeriesImputer to linearly interpolate missing values
imputer = TimeSeriesImputer(input_column='Quantity', 
                            option='interpolate',
                            method='linear',
                            freq='W-WED')

train_imputed_tsdf = imputer.transform(train_tsdf)

# DropColumns: Drop columns that should not be included for modeling. `logmove` is the log of the number of 
# units sold, so providing this number would be cheating. `WeekFirstDay` would be 
# redundant since we already have a feature for the last day of the week.
columns_to_drop = ['logmove', 'WeekFirstDay', 'week']
column_dropper = DropColumns(columns_to_drop)

# TimeSeriesImputer: Fill missing values in the features
# First, we need to create a dictionary with key as column names and value as values used to fill missing 
# values for that column. We are going to use the mean to fill missing values for each column.
columns_with_missing_values = train_imputed_tsdf.columns[pd.DataFrame(train_imputed_tsdf).isnull().any()].tolist()
columns_with_missing_values = [c for c in columns_with_missing_values if c not in columns_to_drop]
missing_value_imputation_dictionary = {}
for c in columns_with_missing_values:
    missing_value_imputation_dictionary[c] = train_imputed_tsdf[c].mean()
fillna_imputer = TimeSeriesImputer(option='fillna', 
                                   input_column=columns_with_missing_values,
                                   value=missing_value_imputation_dictionary)

# TimeIndexFeaturizer: extract temporal features from timestamps
time_index_featurizer = TimeIndexFeaturizer(correlation_cutoff=0.1, overwrite_columns=True)

# GrainIndexFeaturizer: create indicator variables for stores and brands
oj_series_freq = 'W-WED'
oj_series_seasonality = 52
grain_featurizer = GrainIndexFeaturizer(overwrite_columns=True, ts_frequency=oj_series_freq)

pipeline_ml = AzureMLForecastPipeline([('drop_columns', column_dropper), 
                                       ('fillna_imputer', fillna_imputer),
                                       ('time_index_featurizer', time_index_featurizer),
                                       ('grain_featurizer', grain_featurizer)
                                      ])


train_feature_tsdf = pipeline_ml.fit_transform(train_imputed_tsdf)
test_feature_tsdf = pipeline_ml.transform(test_tsdf)

# Let's get a look at our new feature set
print(train_feature_tsdf.head())


# ### Perform Rolling Origin Cross-Validation with a Random Forest model
# Perform a Rolling Origin cross validation to fit a model. In the sample below we use ROCV to fit a Random Forest model.

# In[ ]:


# Set up the `RollingOriginValidator` to do 2 folds of rolling origin cross-validation
roll_cv = RollingOriginValidator(n_splits=2)
randomforest_model_for_cv = RegressionForecaster(estimator=RandomForestRegressor(),
                                                 make_grain_features=False)

# Set up our parameter grid and feed it to our grid search algorithm
param_grid_rf = {'estimator__n_estimators': np.array([10, 100])}
grid_cv_rf = TSGridSearchCV(randomforest_model_for_cv, param_grid_rf, cv=roll_cv)

# fit and predict
start = time.time()
randomforest_cv_fitted= grid_cv_rf.fit(train_feature_tsdf, y=train_feature_tsdf.ts_value)
print('Best parameter: {}'.format(randomforest_cv_fitted.best_params_))
end = time.time()
print('Total time taken to fit model:{}'.format(end - start))


# ### Fit a model with FTK using `JoblibParallelCompute`
# 
# Use the `JoblibParallelCompute` backend to parallelize the grid search and fit a model using ROCV.

# In[ ]:


compute_strategy_joblib = JoblibParallelCompute(job_count=16)
grid_cv_rf.compute_strategy = compute_strategy_joblib

start = time.time()
# fit and predict
randomforest_cv_fitted_joblib = grid_cv_rf.fit(train_feature_tsdf, y=train_feature_tsdf.ts_value)
print('Best parameter: {}'.format(randomforest_cv_fitted_joblib.best_params_))
end = time.time()
print('Total time:{}'.format(end - start))


# ### Fit a model with FTK using `DaskDistributedCompute`
# Use the `DaskDistributedCompute` backend to fit a model using ROCV. The default execution of this backend performs a process-based parallization of work such as the grid search in this case.

# In[ ]:


compute_strategy_dask = DaskDistributedCompute()
grid_cv_rf.compute_strategy = compute_strategy_dask

start = time.time()
# fit and predict
randomforest_cv_fitted_dask = grid_cv_rf.fit(train_feature_tsdf, y=train_feature_tsdf.ts_value)
print('Best parameter: {}'.format(randomforest_cv_fitted_dask.best_params_))
end = time.time()
print('Total time:{}'.format(end - start))


# ### Fit a model using `AzureBatchAICompute`
# 
# In the section below we show how [Azure Batch AI](https://docs.microsoft.com/en-us/azure/batch-ai/overview) can be used to distribute CV Search jobs to nodes in remote clusters leveraging the Azure Machine Learning's Python SDK.

# #### Create or initialize an AML Workspace
# 
# Initialize a workspace object from scratch or from persisted configuration. Note that you must have a valid Azure subscription for this to work.

# In[ ]:


# Create or fetch workspace

# Provide valid Azure subscription id!
subscription_id = "00000000-0000-0000-0000-000000000000"                    
resource_group = "amlpfbairg1"
workspace_name = "workspace1"
workspace_region = "eastus2" # or eastus2euap

ws = Workspace.create(name = workspace_name,
                      subscription_id = subscription_id,
                      resource_group = resource_group, 
                      location = workspace_region,
                     exist_ok=True)
ws.get_details()


# #### Create Batch AI cluster as compute target
# Let's create a new Batch AI cluster in the current workspace, if it doesn't already exist. 
# And use it to run the training script

# In[ ]:


from azureml.core.compute import BatchAiCompute
from azureml.core.compute import ComputeTarget

# choose a name for your cluster
batchai_cluster_name = 'amlpfbaicluster1'

if batchai_cluster_name in ws.compute_targets():
    compute_target = ws.compute_targets()[batchai_cluster_name]
    if compute_target and type(compute_target) is BatchAiCompute:
        print('Found compute target. Reusing: ' + batchai_cluster_name)
else:
    print('Creating new Batch AI compute target: ' + batchai_cluster_name)
    provisioning_config = BatchAiCompute.provisioning_configuration(vm_size = vm_size, # NC6 is GPU-enabled
                                                                vm_priority = 'lowpriority', # optional
                                                                autoscale_enabled = autoscale_enabled,
                                                                cluster_min_nodes = cluster_min_nodes, 
                                                                cluster_max_nodes = cluster_max_nodes)

    # create the cluster
    compute_target = ComputeTarget.create(ws, batchai_cluster_name, provisioning_config)
    
    # can poll for a minimum number of nodes and for a specific timeout. 
    # if no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    
     # For a more detailed view of current BatchAI cluster status, use the 'status' property    
    print(compute_target.status.serialize())
    print(compute_target.provisioning_errors)


# #### Create a Run Configuration
# 

# In[ ]:


# Create run config
runconfig = RunConfiguration()
runconfig.target = batchai_cluster_name
runconfig.batchai.node_count = 2
runconfig.environment.docker.enabled = True

# Set the datastore config in the runconfig
_default_datastore = Datastore(ws)
data_ref_configs = {}
data_ref = _default_datastore._get_data_reference()
data_ref_configs[data_ref.data_reference_name] = data_ref._to_config()
runconfig.data_references = data_ref_configs;


# #### Run an experiment
# 

# In[ ]:


# Set AMLBatchAI as the compute backend
compute_strategy_batchai = AMLBatchAICompute(ws, runconfig)
grid_cv_rf.compute_strategy = compute_strategy_batchai

# Fit a model with CVSearch
start = time.time()
randomforest_cv_fitted_batchai = grid_cv_rf.fit(train_feature_tsdf, y=train_feature_tsdf.ts_value)
end = time.time()

# Results
print('Best parameter: {}'.format(randomforest_cv_fitted_batchai.best_params_))
print('Total time:{}'.format(end - start))

