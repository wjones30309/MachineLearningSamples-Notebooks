
# coding: utf-8

# # Forecasting Package Transforms, Part 1
# 
# This notebook goes over some of the dataset transformations included with the **Azure Machine Learning Package for Forecasting** (AMLPF). We recommend the [Dominick's OJ Sales Forecasting Notebook](https://aka.ms/aml-packages/forecasting/notebooks/sales_forecasting) as a prerequisite to this notebook since it gives an overview of the general package features.
# 
# The following transforms are demonstrated:
# * DropColumns - drop named columns
# * TimeIndexFeaturizer - create new columns derived from the time index
# * GrainIndexFeaturizer - create new categorical columns derived from the grain index
# * CategoryBinarizer - create binary type columns from categorical columns
# * SklearnTransformWrapper - wrapper for a general sklearn transform

# In[80]:


import warnings

# Suppress warnings
warnings.filterwarnings("ignore") 

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from ftk import TimeSeriesDataFrame, AzureMLForecastPipeline
from ftk.transforms import (TimeIndexFeaturizer, GrainIndexFeaturizer,
                            CategoryBinarizer, DropColumns,
                           SklearnTransformerWrapper)
from ftk.data import load_dominicks_oj_dataset
print('imports done.')


# ## Load and Prepare Data
# 
# We begin by loading a subset of the [University of Chicago's Dominick's Finer Foods dataset](https://research.chicagobooth.edu/kilts/marketing-databases/dominicks) that contains orange juice sales from the Dominick's grocery chain. The `load_dominicks_oj_dataset` utility function from the `ftk.data` subpackage reads the data from CSV, loads it into a [TimeSeriesDataFrame](https://docs.microsoft.com/en-us/python/api/ftk.dataframe_ts.timeseriesdataframe?view=azure-ml-py-latest), and splits the data into train and test sets. By default, the test set holds the last 40 rows from each time series.  
# 
# The dataset holds sales time series for each grocery store and each orange juice brand; there are dozens of stores and three brands. To limit the amount of processing, we restrict the training set to just three stores: 2, 5, and 8. Since `store` is part of the time series grain, it is contained in the grain segment of the TimeSeriesDataFrame index for `train_df`. We can select data frame rows corresponding to the desired three stores by generating a boolean array with [pandas.Index.isin](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Index.isin.html).

# In[76]:


# Load time series data frames
train_df, test_df = load_dominicks_oj_dataset()

# Select a small number of stores
train_store_list = [2, 5, 8]
train_stores = train_df.grain_index.get_level_values('store')
train_df = train_df[train_stores.isin(train_store_list)]
train_df.head()


# ## DropColumns Transform
# 
# We note that the `logmove` column in the dataset is the logarithm of the `Quantity` column. Since `Quantity` is the target forecasting quantity, including `logmove` as a feature amounts to a leakage. We hence want to drop it from the data frame. This can be accomplished with pandas functions, of course, but in the interest of having pipeline composable operations, columns can also be dropped via the `DropColumns` transform. We also drop the `week` and `WeekFirstDay` columns here since we will not be using them.    

# In[77]:


# Drop unnecessary columns
to_drop = ['logmove', 'week', 'WeekFirstDay']
drop_columns = DropColumns(to_drop)
train_df = drop_columns.fit_transform(train_df)
train_df.head()


# ## Time Index Features
# 
# To prepare for a machine learning task, we often generate features that may be useful for prediction. The [TimeIndexFeaturizer](https://docs.microsoft.com/en-us/python/api/ftk.transforms.time_index_featurizer.timeindexfeaturizer?view=azure-ml-py-latest) automates the creation of features derived from the time index, like the day of the month and the quarter of the year. To see the full set of time features the transform can generate, we initialize it with the `prune_features=False` option. 

# In[40]:


time_featurizer = TimeIndexFeaturizer(prune_features=False)
train_with_time_feats = time_featurizer.fit_transform(train_df)

# View the full set of time index features
time_features = time_featurizer.FEATURE_COLUMN_NAMES
train_with_time_feats[time_features].head()


# By default, `TimeIndexFeaturizer` will exclude time features with zero variance or very high correlation with other features. We can restore the default by re-initializing the transform. Notice that only a small subset of the "unpruned" time features remain. See the [transform documentation](https://docs.microsoft.com/en-us/python/api/ftk.transforms.time_index_featurizer.timeindexfeaturizer?view=azure-ml-py-latest) for more information on `TimeIndexFeaturizer` options.

# In[78]:


time_featurizer = TimeIndexFeaturizer()
train_with_time_feats = time_featurizer.fit_transform(train_df)

# View the pruned set of time index features
pruned_time_features = [ft for ft in train_with_time_feats.columns 
                        if ft in time_features]
train_with_time_feats[pruned_time_features].head()


# ## Grain Index Features
# 
# Machine learning estimators in the Forecasting Package can train single models over multiple time series, so it can be helpful to generate features that identify the individual series. The `GrainIndexFeaturizer` will generate such features by copying the grain indices into regular data frame columns. By default, the grain feature column names will have 'grain_' appended to their index level names. Note that grain features are not useful machine learning features when the `TimeSeriesDataFrame` has `group_colnames == grain_colnames` since the grain features will then be constant within each group. See the [package documentation](https://docs.microsoft.com/en-us/python/api/ftk.transforms.grain_index_featurizer.grainindexfeaturizer?view=azure-ml-py-latest) for more detail on the `GrainIndexFeaturizer` options.

# In[61]:


grain_featurizer = GrainIndexFeaturizer()
grain_features = grain_featurizer.preview_grain_feature_names(train_df)
train_with_grain_feats = grain_featurizer.fit_transform(train_df)

# Show the first row of grain features for each time series
first_by_grain = train_with_grain_feats.groupby_grain().head(1)
first_by_grain[grain_features]


# Importantly, the grain features have a data type of [pandas.Categorical](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Categorical.html), indicating that they are categorical features.    

# In[62]:


train_with_grain_feats.dtypes


# ## Binary Encoding of Categorical Features
# 
# The scikit-learn estimators supported in the Forecasting Package currently require that categorical features be encoded to numeric values. There are many ways to do such an encoding, but one of the most common is to transform a categorical feature with `n` categories into `n` binary, indicator columns. This is also known as "dummy coding."
# 
# The `CategoryBinarizer` transform generates indicator columns from a desired set of categorical features. In the following example, we initialize a `CategoryBinarizer` and instruct it to dummy code the grain features introduced in the previous section.

# In[49]:


cat_encoder = CategoryBinarizer(columns=grain_features)
train_with_encoded_grain_feats = cat_encoder.fit_transform(train_with_grain_feats)

# Show the first row of grain features for each time series
first_by_grain = train_with_encoded_grain_feats.groupby_grain().head(1)
first_by_grain.filter(regex='grain_')


# In the previous example, the categorical columns were designated by the `columns=<>` option of `CategoryBinarizer` constructor. If this option isn't specified, the transform will encode all data frame columns with `dtype=category` or `dtype=object`. The former consists of `pandas.Categorical` columns while the latter includes non-numeric types like character strings. 
# 
# We also note that it is common practice to drop the indicator column corresponding to the first category in a categorical feature to avoid having linearly dependent features (i.e. creating a singular design matrix). We enable this behavior with the `drop_first=True` option to the `CategoryBinarizer`. By default, it is False. The following example uses the `drop_first` option and does not specify `columns`, allowing the transform to determine which features are categorical. See the [package documentation](https://docs.microsoft.com/en-us/python/api/ftk.transforms.category_binarizer.categorybinarizer?view=azure-ml-py-latest) for more `CategoryBinarizer` options.

# In[50]:


# Re-transform and drop the first category of the encoding 
cat_encoder = CategoryBinarizer(drop_first=True)
train_with_encoded_grain_feats = cat_encoder.fit_transform(train_with_grain_feats)

# Show the first row of grain features for each time series
first_by_grain = train_with_encoded_grain_feats.groupby_grain().head(1)
first_by_grain.filter(regex='grain_')


# ## Wrapping Scikit-learn Transforms
# 
# For the sake of flexibility, we have a wrapper class, `SklearnTransformerWrapper`, for using scikit-learn transforms from the [sklearn.preprocessing](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) package. This class enables sklearn transformations on `TimeSeriesDataFrame` columns. We note that the sklearn transforms **should be used with caution** as they are simply applied to an input data frame with no regard to `grain` or `group` structure of the `TimeSeriesDataFrame`.  
# 
# We illustrate the wrapper class with a `StandardScaler` sklearn transform. For each column in its input, this transform subtracts the column mean and scales the column to unit variance. See the [package documentation](https://docs.microsoft.com/en-us/python/api/ftk.transforms.sklearn_transformer_wrapper.sklearntransformerwrapper?view=azure-ml-py-latest) for more information on `SklearnTransformerWrapper`.

# In[72]:


# Make a list of numeric features
numeric_features = list(set(train_df.columns) - set(['feat', 'Quantity']))

# Center and scale the training set using an sklearn StandardScaler
scaler = StandardScaler()
scaler_wrapper = SklearnTransformerWrapper(scaler, input_column=numeric_features, drop_original=True)
train_scaled = scaler_wrapper.fit_transform(train_df)
train_scaled.head()


# ## Chaining Transforms in a Pipeline
# 
# Finally, we can chain the transforms together inside an [AzureMLForecastPipeline](https://docs.microsoft.com/en-us/python/api/ftk.pipeline.azuremlforecastpipeline?view=azure-ml-py-latest). The following featurization pipeline sequentially drops unneeded columns, scales the original set of numeric features, generates time and grain features, and dummy codes categorical grain features. 

# In[79]:


pipeline = AzureMLForecastPipeline(steps=[('dropper', drop_columns),
                                          ('scaler', scaler_wrapper),
                                          ('time_feats', time_featurizer),
                                          ('grain_feats', grain_featurizer),
                                          ('cat_encoder', cat_encoder)])

train_with_encoded_feats = pipeline.fit_transform(train_df)
first_by_grain = train_with_encoded_feats.groupby_grain().head(1)
first_by_grain


# An important property of a pipeline is that it can be uniformly applied to both the train and test sets. The only difference is that we generally *fit and transform* the pipeline on the training data but only *transform* the test data. 
# 
# The fit operation on the pipeline may store information for stateful transforms, as well as fit a machine learning estimator in the last pipeline step. We illustrate the former by constructing test data with a limited set of stores, as in the training data, but here we leave out one store that is present in the training data. Ideally, the featurized test set would still have an indicator column for the missing store since it was present in the training data. This is often necessary for scoring the test data against a fitted estimator. By design, `CategoryBinarizer.fit` saves the set of categories present in the training data and introduces them to the test set when it is featurized.

# In[75]:


# Select a small number of stores
# Purposely leave store 8 out of the test set
test_store_list = [2, 5]
test_stores = test_df.grain_index.get_level_values('store')
test_df = test_df[test_stores.isin(test_store_list)]

# Create features and dummy code them using previously fit pipeline
test_with_encoded_feats = pipeline.transform(test_df)

# Show the first row of grain features for each time series
first_by_grain = test_with_encoded_feats.groupby_grain().head(1)
first_by_grain.filter(regex='grain_')

