
# coding: utf-8

# # This notebook describes creation of a forecasting model and its deployment on ACI.

# ### Before start
# 
# Install FTK using [shell](https://azuremlftkrelease.blob.core.windows.net/latest/install_amlpf_linux.sh) or [batch](https://azuremlftkrelease.blob.core.windows.net/latest/install_amlpf_windows.bat) scripts.  
# To run this notebook please install the python SDK by running 
# ```
# activate azuremlftk_nov2018
# pip install --upgrade azureml-sdk[notebooks,automl]
# ```
# Login to Azure
# ```
# az login
# ```
# After installation is complete, select Kernel>Change Kernel>azuremlftk_nov2018.

# #### Imports

# In[ ]:


import os
import pandas as pd
import numpy as np
import json

from ftk import TimeSeriesDataFrame, ForecastDataFrame
from ftk.operationalization import ScoreContext
from ftk.transforms import TimeSeriesImputer, TimeIndexFeaturizer, DropColumns, GrainIndexFeaturizer 
from ftk.models import RegressionForecaster
from sklearn.ensemble import RandomForestRegressor
from ftk.pipeline import AzureMLForecastPipeline
from ftk.data import load_dominicks_oj_dataset


# #### Load data
# To train and test model load the Dominicks data set.

# In[ ]:


train_tsdf, test_tsdf = load_dominicks_oj_dataset()
# Use a TimeSeriesImputer to linearly interpolate missing values
imputer = TimeSeriesImputer(input_column='Quantity', 
                            option='interpolate',
                            method='linear',
                            freq='W-WED')

train_imputed_tsdf = imputer.transform(train_tsdf)


# #### Prepare the pipeline.
# Create the forecasting pipeline to be deployed.

# In[ ]:


oj_series_freq = 'W-WED'
oj_series_seasonality = 52

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
grain_featurizer = GrainIndexFeaturizer(overwrite_columns=True, ts_frequency=oj_series_freq)

random_forest_model_deploy = RegressionForecaster(estimator=RandomForestRegressor(), make_grain_features=False)

pipeline_deploy = AzureMLForecastPipeline([('drop_columns', column_dropper), 
                                           ('fillna_imputer', fillna_imputer),
                                           ('time_index_featurizer', time_index_featurizer),
                                           ('random_forest_estimator', random_forest_model_deploy)
                                          ])


# ## Deployment

# #### Create the required files.
# We will now deploy the model as a web service. That means we will create a docker image with the service logic and host it on [Azure Container Instances](https://azure.microsoft.com/en-us/services/container-instances/). The image creation of Forecasting model requires the model contained in the pickle file and dependencies file. This file is required to create the conda environment.

# Model file

# In[ ]:


import pickle
with open('pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline_deploy, f)


# Conda dependencies file

# In[ ]:


get_ipython().run_cell_magic('writefile', 'conda_dependencies.yml', '################################################################################\n#\n# Create Azure ML Forecasting Toolkit Conda environments on Linux platforms. \n# This yml is used specifically in creating containers on ACR for use \n# AML deployments.\n#\n################################################################################\n\nname: azuremlftk_nov2018\ndependencies:\n  # AzureML FTK dependencies\n  - pyodbc\n  - statsmodels\n  - pandas\n  - scikit-learn==0.19.1\n  - tensorflow\n  - keras\n  - distributed==1.23.1\n\n  - pip:\n    # AML logging\n    - https://azuremldownloads.azureedge.net/history-packages/preview/azureml.primitives-1.0.11.491405-py3-none-any.whl\n    - https://azuremldownloads.azureedge.net/history-packages/preview/azureml.logging-1.0.81-py3-none-any.whl\n    \n    #azure ml\n    - azureml-sdk[automl]\n    \n    #Dependencies from other AML packages\n    - https://azuremlftkrelease.blob.core.windows.net/azpkgdaily/azpkgcore-1.0.18309.1b1-py3-none-any.whl\n    - https://azuremlftkrelease.blob.core.windows.net/azpkgdaily/azpkgsql-1.0.18309.1b1-py3-none-any.whl\n\n    # AMLPF package  \n    - https://azuremlftkrelease.blob.core.windows.net/dailyrelease/azuremlftk-0.1.18305.1a1-py3-none-any.whl')


# #### Run the deployment.

# In[ ]:


# Check core SDK version number
import azureml.core

print("SDK version:", azureml.core.VERSION)


# Initialize a workspace object.

# The workspace is an Azure resource that holds all of your models, docker images, and services created. It can be configured using the file in json format. The example of this file is shown below.

# In[ ]:


get_ipython().run_cell_magic('writefile', 'workspace_aci.json', '{\n    "subscription_id": "<subscription id>",\n    "resource_group": "<resource group>",\n    "workspace_name": "<workspace name>",\n    "location": "<location>"\n}')


# If the workspace is not already present create it.

# In[ ]:


from azureml.core import Workspace
from azureml.exceptions import ProjectSystemException
ws = None
try:
    #Try to get the workspace if it exists.
    ws = Workspace.from_config("workspace_aci.json")
except ProjectSystemException:
    #If the workspace was not found, create it.
    with open("workspace_aci.json", 'r') as config:
        ws_data = json.load(config)
    ws = Workspace.create(name = ws_data["workspace_name"],
                          subscription_id = ws_data["subscription_id"],
                          resource_group = ws_data["resource_group"],
                          location = ws_data["location"])
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')


# #### Register Model

# You can add tags and descriptions to your models. The below call registers `pipeline.pkl` file as a model with the name `aciforecast` in the workspace.

# In[ ]:


from azureml.core.model import Model

model = Model.register(model_path = "pipeline.pkl",
                       model_name = "aciforecast",
                       workspace = ws)


# Models are versioned. If you call the register_model command many times with same model name, you will get multiple versions of the model with increasing version numbers.

# In[ ]:


regression_models = Model.list(ws)
for m in regression_models:
    print("Name:", m.name,"\tVersion:", m.version, "\tDescription:", m.description, m.tags)


# You can pick a specific model to deploy

# In[ ]:


print(model.name, model.description, model.version, sep = '\t')


# ### Create Docker Image

# Create `score.py`. Note that the `aciforecast` in the `get_model_path` call is referring to a same named model `aciforecast` registered under the workspace.

# In[ ]:


get_ipython().run_cell_magic('writefile', 'score.py', 'import pickle\nimport json\n\nfrom ftk.operationalization.score_script_helper import run_impl\nfrom azureml.core.model import Model\n\ndef init():\n    #init method will be executed once at start of the docker - load the model\n    global pipeline\n    #Get the model path.\n    pipeline_pickle_file = Model.get_model_path("aciforecast")\n    #Load the model.\n    with open(pipeline_pickle_file, \'rb\') as f:\n        pipeline = pickle.load(f)\n\n#Run method is executed once per call.\ndef run(input_data):\n    #The JSON encoded input_data will be interpreted as a TimeSeriedData frame and will \n    #be used for forecasting.\n    #Return the JSON encoded data frame with forecast.\n    return run_impl(input_data, pipeline=pipeline)')


# Note that following command can take a few minutes. An image can contain multiple models.

# In[ ]:


from azureml.core.image import Image, ContainerImage

image_config = ContainerImage.image_configuration(runtime= "python",
                                 execution_script="score.py",
                                 conda_file="conda_dependencies.yml")

image = Image.create(name = "ftkimage1",
                     # this is the model object 
                     models = [model],
                     image_config = image_config, 
                     workspace = ws)


# Monitor image creation.

# In[ ]:


image.wait_for_creation(show_output = True)


# List images and find out the detailed build log for debugging.

# In[ ]:


for i in Image.list(workspace = ws):
    print('{}(v.{} [{}]) stored at {} with build log {}'.format(i.name, i.version, i.creation_state, i.image_location, i.image_build_log_uri))


# ### Deploy image as web service on Azure Container Instance
# 
# The deployment configuration defines how much resources should be reserved for this container.

# In[ ]:


from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(cpu_cores = 1, 
                                               memory_gb = 1)


#  Start deployment using newly created configuration. Note that the service creation can take few minutes.

# In[ ]:


from azureml.core.webservice import Webservice

aci_service_name = 'ftk-service-1'
print(aci_service_name)
aci_service = Webservice.deploy_from_image(deployment_config = aciconfig,
                                           image = image,
                                           name = aci_service_name,
                                           workspace = ws)
aci_service.wait_for_deployment(True)
print(aci_service.state)


# If there was a problem during deployment it may be useful to analyze the deployment logs.

# In[ ]:


print(aci_service.get_logs())


# ### Test web service

# Create a validation data set to benchmark new service.
# You might ask why we are sending a ForecastDataFrame to the service? We do it to give it the values of the future predictor variable, like price at the future time. 

# In[ ]:


imputer = TimeSeriesImputer(input_column='Quantity', 
                            option='interpolate',
                            method='linear',
                            freq='W-WED')    
train_imputed_tsdf = imputer.transform(train_tsdf)
validate_ts = train_imputed_tsdf.assign(PointForecast=0.0, DistributionForecast=np.nan)
validate_fdf = ForecastDataFrame(validate_ts, pred_point='PointForecast', pred_dist='DistributionForecast')
sc_validate = ScoreContext(input_training_data_tsdf=train_imputed_tsdf,
                           input_scoring_data_fcdf=validate_fdf, 
                           pipeline_execution_type='train_predict')


# We are sending the training data set to train the pickled model:

# In[ ]:


train_imputed_tsdf.head()


# ForecastDataFrame for validation contains predictor values and the empty columns for predicted values. In this case it is columns DistributionForecast and PointForecast.

# In[ ]:


validate_fdf.head()


# ScoreContext contains both training and prediction(validation) data frames and helps to serialize these data to JSON format understood by the service. 

# Run the prediction and show the results

# In[ ]:


json_direct =aci_service.run(sc_validate.to_json())
fcdf_direct=ForecastDataFrame.construct_from_json(json_direct)
fcdf_direct.head()


# ### Delete ACI service and resource group

# This part of a notebook is oprtional and intended to clean up after work is complete.
# First delete the service.

# In[ ]:


aci_service.delete()


# Check if services are present in the workspace.

# In[ ]:


[svc.name for svc in Webservice.list(ws)]


# Delete the resource group.<br/>
# **Note** This operation is danger and will delete all the content of the resource group.
# To delete group the azure sdk package needs to be installed:
# ```
# pip install https://azuremlftkrelease.blob.core.windows.net/azpkgdaily/azpkgamlsdk-1.0.18309.1b1-py3-none-any.whl
# ```

# In[ ]:


from azpkgamlsdk.deployment.utils_environment import delete_resource_group

delete_resource_group(ws.resource_group, ws.subscription_id)

