#!/usr/bin/env python
# coding: utf-8

# In[86]:


import pandas as pd   
import matplotlib.pyplot as plt
import time

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor

import seaborn as sns

from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import HuberRegressor

from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression

from numpy import mean
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint, uniform as sp_uniform

import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from interpret.blackbox import LimeTabular
from interpret import show
from lime.lime_tabular import LimeTabularExplainer
from sklearn.datasets import load_boston

from sklearn.metrics import mean_absolute_error, explained_variance_score

import shap
# Run initjs() function
shap.initjs()
from sklearn.datasets import fetch_california_housing
import dice_ml
import joblib

import streamlit as st
import pickle
from PIL import Image


# In[87]:


df = pd.read_csv("F:/Urooj/Masters/IBA/ML/assign 3/WTI Price.csv")


# In[88]:


print(df.shape)
print(df.columns)


# ### Preprocessing

# In[89]:


df.drop('DATE', axis=1, inplace=True)


# In[90]:


df.isnull().sum()


# In[91]:


df=df.fillna(0)
df.isnull().sum()


# In[92]:


df.dtypes


# ### Splitting

# In[93]:


y = df['WTI_Spot'].copy()
X = df.drop('WTI_Spot', axis=1).copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# ### Feature Selection

# #### Forward

# In[9]:


ols_reg = LinearRegression()

# Create an instance of SequentialFeatureSelector
sfs = SequentialFeatureSelector(ols_reg, direction='forward', n_features_to_select=10)

# Fit the feature selector on your data
sfs.fit(X_train, y_train)
print(sfs.get_feature_names_out())

# Get the selected feature names
selected_features = X_train.columns[sfs.get_support()]

# Fit the linear regression model on the selected features
ols_reg.fit(X_train[selected_features], y_train)

# Make predictions on the test set
y_pred = ols_reg.predict(X_test[selected_features])

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: %.2f" % mse)

# Calculate coefficient of determination (R-squared)
r2 = r2_score(y_test, y_pred)
print("Coefficient of determination: %.3f" % r2)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

ev = explained_variance_score(y_test, y_pred)
print("Explained Variance Score:", ev)


# #### Backward

# In[10]:


ols_reg = LinearRegression()
sfs1 = SequentialFeatureSelector(ols_reg, direction='backward',n_features_to_select=5)
sfs1.fit(X_train, y_train)
print(sfs1.get_feature_names_out())

selected_features1 = X_train.columns[sfs1.get_support()]

# Fit the linear regression model on the selected features
ols_reg.fit(X_train[selected_features1], y_train)

# Make predictions on the test set
y_pred1 = ols_reg.predict(X_test[selected_features1])

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred1)
print("Mean squared error: %.2f" % mse)

# Calculate coefficient of determination (R-squared)
r2 = r2_score(y_test, y_pred1)
print("Coefficient of determination: %.3f" % r2)
mae = mean_absolute_error(y_test, y_pred1)
print("Mean Absolute Error:", mae)

ev = explained_variance_score(y_test, y_pred1)
print("Explained Variance Score:", ev)


# ### Linear Regression

# In[11]:


reg = LinearRegression().fit(X_train, y_train)
print(reg.score(X, y))
print(reg.coef_)
print(reg.intercept_)
y_pred_test = reg.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred_test))
print("Coefficient of determination: %.3f" % r2_score(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)
print("Mean Absolute Error:", mae)
ev = explained_variance_score(y_test, y_pred_test)
print("Explained Variance Score:", ev)


# ### KNN Regression

# In[12]:


neigh = KNeighborsRegressor(n_neighbors=5)
knn_reg = neigh.fit(X_train, y_train)
print(knn_reg.score(X_train, y_train))
#print(reg.coef_)
#print(reg.intercept_)
y_pred3 = knn_reg.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred3))
print("Coefficient of determination: %.3f" % r2_score(y_test, y_pred3))
mae = mean_absolute_error(y_test, y_pred3)
print("Mean Absolute Error:", mae)
ev = explained_variance_score(y_test, y_pred3)
print("Explained Variance Score:", ev)


# ### Decision Tree

# In[13]:


start_time = time.time()
regTree = DecisionTreeRegressor(max_depth=5,random_state=0)
regTree.fit(X_train, y_train)
y_predDF = regTree.predict(X_test)
print("DT: R2 = %.4f and MSE = %.2f" % (regTree.score(X_test,y_test), mean_squared_error(y_test, y_predDF)))
print("Coefficient of determination: %.3f" % r2_score(y_test, y_predDF))
mae = mean_absolute_error(y_test, y_predDF)
print("Mean Absolute Error:", mae)
ev = explained_variance_score(y_test, y_predDF)
print("Explained Variance Score:", ev)
end_time = time.time()
#calculate the total time
total_time = end_time - start_time
print("Total time DT: ", total_time)


# In[14]:


start_time = time.time()
regTree = DecisionTreeRegressor(max_depth=6,random_state=0)
regTree.fit(X_train, y_train)
y_predDF1 = regTree.predict(X_test)
print("DT: R2 = %.4f and MSE = %.2f" % (regTree.score(X_test,y_test), mean_squared_error(y_test, y_predDF1)))
print("Coefficient of determination: %.3f" % r2_score(y_test, y_predDF1))
mae = mean_absolute_error(y_test, y_predDF1)
print("Mean Absolute Error:", mae)
ev = explained_variance_score(y_test, y_predDF1)
print("Explained Variance Score:", ev)
end_time = time.time()
#calculate the total time
total_time = end_time - start_time
print("Total time DT: ", total_time)


# In[15]:


start_time = time.time()
regTree = DecisionTreeRegressor(max_depth=10,random_state=0)
regTree.fit(X_train, y_train)
y_predDF2 = regTree.predict(X_test)
print("DT: R2 = %.4f and MSE = %.2f" % (regTree.score(X_test,y_test), mean_squared_error(y_test, y_predDF2)))
print("Coefficient of determination: %.3f" % r2_score(y_test, y_predDF2))
mae = mean_absolute_error(y_test, y_predDF2)
print("Mean Absolute Error:", mae)
ev = explained_variance_score(y_test, y_predDF2)
print("Explained Variance Score:", ev)
end_time = time.time()
#calculate the total time
total_time = end_time - start_time
print("Total time DT: ", total_time)


# ### Random Forest

# In[16]:


regRF = RandomForestRegressor(max_depth=10, max_features=4, min_samples_split=8,
                      n_estimators=300, random_state=0)
regRF.fit(X_train, y_train)
y_predRF = regRF.predict(X_test)
print("RF: R2 = %.4f and MSE = %.2f" % (regRF.score(X_test,y_test), mean_squared_error(y_test, y_predRF)))
print("Coefficient of determination: %.3f" % r2_score(y_test, y_predRF))
mae = mean_absolute_error(y_test, y_predRF)
print("Mean Absolute Error:", mae)
ev = explained_variance_score(y_test, y_predRF)
print("Explained Variance Score:", ev)


# In[17]:


rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_predRF1 = rf.predict(X_test)
print(f"Mean squared error: {mean_squared_error(y_test, y_predRF1)}")
print(f"Coefficient of determination: {r2_score(y_test, y_predRF1)}")
mae = mean_absolute_error(y_test, y_predRF1)
print("Mean Absolute Error:", mae)
ev = explained_variance_score(y_test, y_predRF1)
print("Explained Variance Score:", ev)


# ### Gradient Boosting

# In[18]:


regGB = GradientBoostingRegressor(max_depth=6, max_features=4, min_samples_split=8,
                      n_estimators=300, random_state=0)
regGB.fit(X_train, y_train)
y_predgb = regGB.predict(X_test)
print("GB: R2 = %.4f and MSE = %.2f" % (regGB.score(X_test,y_test), mean_squared_error(y_test, y_predgb)))
print(f"Coefficient of determination: {r2_score(y_test, y_predgb)}")
mae = mean_absolute_error(y_test, y_predgb)
print("Mean Absolute Error:", mae)
ev = explained_variance_score(y_test, y_predgb)
print("Explained Variance Score:", ev)


# In[20]:


start_time = time.time()
regGB1 = GradientBoostingRegressor(max_depth=4, max_features=3, min_samples_split=6,
                      n_estimators=290, random_state=0)
regGB1.fit(X_train, y_train)
y_predgb1 = regGB1.predict(X_test)
print("GB: R2 = %.4f and MSE = %.2f" % (regGB.score(X_test,y_test), mean_squared_error(y_test, y_predgb1)))
print(f"Coefficient of determination: {r2_score(y_test, y_predgb1)}")
mae = mean_absolute_error(y_test, y_predgb1)
print("Mean Absolute Error:", mae)
ev = explained_variance_score(y_test, y_predgb1)
print("Explained Variance Score:", ev)
end_time = time.time()
#calculate the total time
total_time = end_time - start_time
print("Total time RF: ", total_time)


# In[21]:


gb = GradientBoostingRegressor()

# Fit the model to your training data
gb.fit(X_train, y_train)
y_predgb2 = gb.predict(X_test)
print(f"Mean squared error: {mean_squared_error(y_test, y_predgb2)}")
print(f"Coefficient of determination: {r2_score(y_test, y_predgb2)}")
mae = mean_absolute_error(y_test, y_predgb2)
print("Mean Absolute Error:", mae)
ev = explained_variance_score(y_test, y_predgb2)
print("Explained Variance Score:", ev)


# ### XG Boosting

# In[22]:


# Create an XGBRegressor model
xgb_model = xgb.XGBRegressor(max_depth=3, n_estimators=300, learning_rate=0.1)

# Start the timer
start_time = time.time()

# Fit the xgb_model on the training data
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_predxg = xgb_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_predxg)
r2 = r2_score(y_test, y_predxg)

# Print the evaluation metrics
print("Mean squared error: %.2f" % mse)
print("Coefficient of determination: %.3f" % r2)
mae = mean_absolute_error(y_test, y_predxg)
print("Mean Absolute Error:", mae)
ev = explained_variance_score(y_test, y_predxg)
print("Explained Variance Score:", ev)

# End the timer and calculate the total time
end_time = time.time()
total_time = end_time - start_time
print("Total time: ", total_time)


# In[23]:


xgb_model1 = xgb.XGBRegressor(max_depth=5, n_estimators=300, learning_rate=0.1)
# Start the timer
start_time = time.time()

# Fit the xgb_model on the training data
xgb_model1.fit(X_train, y_train)

# Predict on the test set
y_predxg1 = xgb_model1.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_predxg1)
r2 = r2_score(y_test, y_predxg1)

# Print the evaluation metrics
print("Mean squared error: %.2f" % mse)
print("Coefficient of determination: %.3f" % r2)
mae = mean_absolute_error(y_test, y_predxg1)
print("Mean Absolute Error:", mae)
ev = explained_variance_score(y_test, y_predxg1)
print("Explained Variance Score:", ev)

# End the timer and calculate the total time
end_time = time.time()
total_time = end_time - start_time
print("Total time: ", total_time)


# In[24]:


xgb_model2 = xgb.XGBRegressor(max_depth=7, n_estimators=1000, learning_rate=0.1)
# Start the timer
start_time = time.time()

# Fit the xgb_model on the training data
xgb_model2.fit(X_train, y_train)

# Predict on the test set
y_predxg2 = xgb_model2.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_predxg2)
r2 = r2_score(y_test, y_predxg2)

# Print the evaluation metrics
print("Mean squared error: %.2f" % mse)
print("Coefficient of determination: %.3f" % r2)
mae = mean_absolute_error(y_test, y_predxg2)
print("Mean Absolute Error:", mae)
ev = explained_variance_score(y_test, y_predxg2)
print("Explained Variance Score:", ev)

# End the timer and calculate the total time
end_time = time.time()
total_time = end_time - start_time
print("Total time: ", total_time)


# In[25]:


xgb = XGBRegressor()
# Fit the model to your training data
xgb.fit(X_train, y_train)
y_predxg3 = xgb.predict(X_test)
print(f"Mean squared error: {mean_squared_error(y_test, y_predxg3)}")
print(f"Coefficient of determination: {r2_score(y_test, y_predxg3)}")
mae = mean_absolute_error(y_test, y_predxg3)
print("Mean Absolute Error:", mae)
ev = explained_variance_score(y_test, y_predxg3)
print("Explained Variance Score:", ev)


# ### LG Boosting

# In[26]:


# Create an LGBMRegressor model
lgb_model = lgb.LGBMRegressor(max_depth=5, n_estimators=300, learning_rate=0.1)

# Start the timer
start_time = time.time()

# Fit the lgb_model on the training data
lg1=lgb_model.fit(X_train, y_train)

# Predict on the test set
y_predlg = lgb_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_predlg)
r2 = r2_score(y_test, y_predlg)

# Print the evaluation metrics
print("Mean squared error: %.2f" % mse)
print("Coefficient of determination: %.3f" % r2)
mae = mean_absolute_error(y_test, y_predlg)
print("Mean Absolute Error:", mae)
ev = explained_variance_score(y_test, y_predlg)
print("Explained Variance Score:", ev)

# End the timer and calculate the total time
end_time = time.time()
total_time = end_time - start_time
print("Total time: ", total_time)


# In[30]:


# Create an LGBMRegressor model
lgb_model1 = lgb.LGBMRegressor(max_depth=15, n_estimators=500, learning_rate=0.1)

# Start the timer
start_time = time.time()

# Fit the lgb_model on the training data
lgb_model1.fit(X_train, y_train)

# Predict on the test set
y_predlg1 = lgb_model1.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_predlg1)
r2 = r2_score(y_test, y_predlg1)

# Print the evaluation metrics
print("Mean squared error: %.2f" % mse)
print("Coefficient of determination: %.3f" % r2)
mae = mean_absolute_error(y_test, y_predlg1)
print("Mean Absolute Error:", mae)
ev = explained_variance_score(y_test, y_predlg1)
print("Explained Variance Score:", ev)

# End the timer and calculate the total time
end_time = time.time()
total_time = end_time - start_time
print("Total time: ", total_time)


# In[31]:


# Create an LGBMRegressor model
lgb_model2 = lgb.LGBMRegressor(max_depth=10, n_estimators=1000, learning_rate=0.1)

# Start the timer
start_time = time.time()

# Fit the lgb_model on the training data
lgb_model2.fit(X_train, y_train)

# Predict on the test set
y_predlg2 = lgb_model2.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_predlg2)
r2 = r2_score(y_test, y_predlg2)

# Print the evaluation metrics
print("Mean squared error: %.2f" % mse)
print("Coefficient of determination: %.3f" % r2)
mae = mean_absolute_error(y_test, y_predlg2)
print("Mean Absolute Error:", mae)
ev = explained_variance_score(y_test, y_predlg2)
print("Explained Variance Score:", ev)

# End the timer and calculate the total time
end_time = time.time()
total_time = end_time - start_time
print("Total time: ", total_time)


# In[32]:


lgb_model3 = lgb.LGBMRegressor(max_depth=10, n_estimators=150, learning_rate=0.1)

# Start the timer
start_time = time.time()

# Fit the lgb_model on the training data
lgb_model3.fit(X_train, y_train)

# Predict on the test set
y_predlg3 = lgb_model3.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_predlg3)
r2 = r2_score(y_test, y_predlg3)

# Print the evaluation metrics
print("Mean squared error: %.2f" % mse)
print("Coefficient of determination: %.3f" % r2)
mae = mean_absolute_error(y_test, y_predlg3)
print("Mean Absolute Error:", mae)
ev = explained_variance_score(y_test, y_predlg3)
print("Explained Variance Score:", ev)

# End the timer and calculate the total time
end_time = time.time()
total_time = end_time - start_time
print("Total time: ", total_time)


# In[33]:


lgbm = LGBMRegressor()

# Fit the model to your training data
lgbm1=lgbm.fit(X_train, y_train)

# Make predictions on the test data
y_predlg4 = lgbm1.predict(X_test)
print(f"Mean squared error: {mean_squared_error(y_test, y_predlg4)}")
print(f"Coefficient of determination: {r2_score(y_test, y_predlg4)}")
mae = mean_absolute_error(y_test, y_predlg4)
print("Mean Absolute Error:", mae)
ev = explained_variance_score(y_test, y_predlg4)
print("Explained Variance Score:", ev)


# ### Random Search

# In[34]:


# Start the timer
start_time = time.time()

# Define the parameter distributions to search over
param_dist = {
    'num_leaves': sp_randint(31, 128),
    'max_depth': [-1, 5, 10, 15, 20],
    'min_child_samples': sp_randint(1, 21),
    'subsample': sp_uniform(loc=0.6, scale=0.4),
    'colsample_bytree': sp_uniform(loc=0.6, scale=0.4),
    'learning_rate': [0.01, 0.04, 0.1, 0.2, 0.3, 0.5],
    'n_estimators': sp_randint(300, 1000),
    'reg_alpha': [0, 0.1, 1, 5, 10],
    'reg_lambda': [0, 1, 10, 50, 100]
}

# Create a LightGBM regressor object
lgbm = lgb.LGBMRegressor(random_state=42)

# Create a RandomizedSearchCV object and fit it to the data
random_search = RandomizedSearchCV(lgbm, param_distributions=param_dist, n_iter=100, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding score
print("Best hyperparameters: ", random_search.best_params_)
print("Best score: ", -random_search.best_score_)

# Use the best model to predict on the test data
best_lgbm = random_search.best_estimator_
y_predrs = best_lgbm.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_predrs)
r2 = r2_score(y_test, y_predrs)

# Print the evaluation metrics
print("Mean squared error: %.2f" % mse)
print("Coefficient of determination: %.3f" % r2)
mae = mean_absolute_error(y_test, y_predrs)
print("Mean Absolute Error:", mae)
ev = explained_variance_score(y_test, y_predrs)
print("Explained Variance Score:", ev)

# End the timer and calculate the total time
end_time = time.time()
total_time = end_time - start_time
print("Total time: ", total_time)


# ### Stacking

# In[36]:


from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

# Define your base models
base_models = [
    ('random_forest', RandomForestRegressor()),
    ('gradient_boosting', GradientBoostingRegressor()),
    ('lg_boosting', LGBMRegressor())
]

# Define the meta model
meta_model = LinearRegression()

# Create the stacking regressor
stacking_reg = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Fit the stacking regressor on your training data
stackingreg=stacking_reg.fit(X_train, y_train)

# Make predictions on the test data
y_pred_stacking = stacking_reg.predict(X_test)

# Evaluate the performance
mse_stacking = mean_squared_error(y_test, y_pred_stacking)
r2_stacking = r2_score(y_test, y_pred_stacking)
mae_stacking = mean_absolute_error(y_test, y_pred_stacking)
ev_stacking = explained_variance_score(y_test, y_pred_stacking)

print("Stacking Regressor:")
print(f"Mean squared error: {mse_stacking}")
print(f"Coefficient of determination: {r2_stacking}")
print("Mean Absolute Error:", mae_stacking)
print("Explained Variance Score:", ev_stacking)


# ### LIME

# In[38]:


#stacking
feature_names = ['GSCI', 'SP500', 'NASDAQ', 'DJIA', 'DXY', 'Federal_Funds_Effective', 'Economic_Uncertainty_Index', 'Inflation_5Y_BE', 'US_3M_Treasury', 'US_1Y_Treasury', '10Y_Less_2Y', 'US_Oil_Demand', 'US_Oil_Supply', 'Demand_Less_Supply']
X_train_array = X_train.values  # Convert X_train to a NumPy array
#X_test_instance = X_test[-20:]  # Select a single instance from X_test
X_test_instance = X_test.iloc[-20]  # Select a single instance from X_test

explainer = LimeTabularExplainer(training_data=X_train_array, mode='regression', feature_names=feature_names)
explanation = explainer.explain_instance(X_test_instance, stackingreg.predict, num_features=14)
explanation.show_in_notebook(show_table=True)


# In[39]:


# Define the feature names
feature_names = ['GSCI', 'SP500', 'NASDAQ', 'DJIA', 'DXY', 'Federal_Funds_Effective', 'Economic_Uncertainty_Index', 'Inflation_5Y_BE', 'US_3M_Treasury', 'US_1Y_Treasury', '10Y_Less_2Y', 'US_Oil_Demand', 'US_Oil_Supply', 'Demand_Less_Supply']

X_train_array = X_train.values  # Convert X_train to a NumPy array
X_test_instance = X_test.iloc[-20]  # Select a single instance from X_test

explainer = LimeTabularExplainer(training_data=X_train_array, mode='regression', feature_names=feature_names)
explanation = explainer.explain_instance(X_test_instance, rf.predict, num_features=14)
explanation.show_in_notebook(show_table=True)


# In[103]:


# Define the feature names
feature_names = ['GSCI', 'SP500', 'NASDAQ', 'DJIA', 'DXY', 'Federal_Funds_Effective', 'Economic_Uncertainty_Index', 'Inflation_5Y_BE', 'US_3M_Treasury', 'US_1Y_Treasury', '10Y_Less_2Y', 'US_Oil_Demand', 'US_Oil_Supply', 'Demand_Less_Supply']

X_train_array = X_train.values  # Convert X_train to a NumPy array
X_test_instance = X_test.iloc[400]  # Select a single instance from X_test

explainer = LimeTabularExplainer(training_data=X_train_array, mode='regression', feature_names=feature_names)
explanation = explainer.explain_instance(X_test_instance, lgbm1.predict, num_features=14)
explanation.show_in_notebook(show_table=True)

predicted_value = lgbm1.predict(X_test_instance.values.reshape(1, -1))
actual_value = y_test.iloc[400]  # Retrieve the actual value for the selected instance
print("Instance Index:", 400)
print("Predicted Value:", predicted_value)
print("Actual Value:", actual_value)


# In[96]:


# Define the feature names
feature_names = ['GSCI', 'SP500', 'NASDAQ', 'DJIA', 'DXY', 'Federal_Funds_Effective', 'Economic_Uncertainty_Index', 'Inflation_5Y_BE', 'US_3M_Treasury', 'US_1Y_Treasury', '10Y_Less_2Y', 'US_Oil_Demand', 'US_Oil_Supply', 'Demand_Less_Supply']

X_train_array = X_train.values  # Convert X_train to a NumPy array
X_test_instance = X_test.iloc[20]  # Select a single instance from X_test

explainer = LimeTabularExplainer(training_data=X_train_array, mode='regression', feature_names=feature_names)
explanation = explainer.explain_instance(X_test_instance, lgbm1.predict, num_features=14)
explanation.show_in_notebook(show_table=True)

predicted_value = lgbm1.predict(X_test_instance.values.reshape(1, -1))
actual_value = y_test.iloc[20]  # Retrieve the actual value for the selected instance
print("Instance Index:", index)
print("Predicted Value:", predicted_value)
print("Actual Value:", actual_value)


# In[102]:


# Define the feature names
feature_names = ['GSCI', 'SP500', 'NASDAQ', 'DJIA', 'DXY', 'Federal_Funds_Effective', 'Economic_Uncertainty_Index', 'Inflation_5Y_BE', 'US_3M_Treasury', 'US_1Y_Treasury', '10Y_Less_2Y', 'US_Oil_Demand', 'US_Oil_Supply', 'Demand_Less_Supply']

X_train_array = X_train.values  # Convert X_train to a NumPy array
X_test_instance = X_test.iloc[100]  # Select a single instance from X_test

explainer = LimeTabularExplainer(training_data=X_train_array, mode='regression', feature_names=feature_names)
explanation = explainer.explain_instance(X_test_instance, lgbm1.predict, num_features=14)
explanation.show_in_notebook(show_table=True)

predicted_value = lgbm1.predict(X_test_instance.values.reshape(1, -1))
actual_value = y_test.iloc[100]  # Retrieve the actual value for the selected instance
print("Instance Index:", 100)
print("Predicted Value:", predicted_value)
print("Actual Value:", actual_value)


# ### Shapley

# In[40]:


#LGBoostingRegressor
explainer = shap.TreeExplainer(rf)
#calculate shapley values for test data
start_index = 1
end_index = 2
shap_values = explainer.shap_values(X_test[start_index:end_index])
X_test[start_index:end_index]


# In[106]:


#LGBoostingRegressor
explainer = shap.TreeExplainer(lgbm1)
#calculate shapley values for test data
start_index = 1
end_index = 2
shap_values = explainer.shap_values(X_test[start_index:end_index])
X_test[start_index:end_index]


# #### Instance 1

# In[111]:


import shap

explainer = shap.TreeExplainer(lgbm1)
# Calculate SHAP values for test data
start_index = 20
end_index = 21
shap_values1 = explainer.shap_values(X_test[start_index:end_index])
instance = X_test[start_index:end_index]

print("Instance Index:", start_index)
print("Predicted Value:", lgbm1.predict(instance)[0])
print("Actual Value:", y_test.iloc[start_index])
X_test[start_index:end_index]


# #### Instance 2

# In[112]:


import shap

explainer = shap.TreeExplainer(lgbm1)
# Calculate SHAP values for test data
start_index = 100
end_index = 101
shap_values2 = explainer.shap_values(X_test[start_index:end_index])
instance = X_test[start_index:end_index]

print("Instance Index:", start_index)
print("Predicted Value:", lgbm1.predict(instance)[0])
print("Actual Value:", y_test.iloc[start_index])
X_test[start_index:end_index]


# #### Instance 3

# In[113]:


explainer = shap.TreeExplainer(lgbm1)
# Calculate SHAP values for test data
start_index = 400
end_index = 401
shap_values3 = explainer.shap_values(X_test[start_index:end_index])
instance = X_test[start_index:end_index]

print("Instance Index:", start_index)
print("Predicted Value:", lgbm1.predict(instance)[0])
print("Actual Value:", y_test.iloc[start_index])
X_test[start_index:end_index]


# In[41]:


print(shap_values[0].shape)
shap_values


# In[56]:


model = lgbm1
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

shap.waterfall_plot(shap_values[0])


# In[114]:


# Create instance index 20
instance_index = 20
X_instance = X_test.iloc[instance_index]

# Get the predicted value
predicted_value = model.predict(X_instance.values.reshape(1, -1))[0]

# Get the actual value
actual_value = y_test.iloc[instance_index]

print("Instance Index:", instance_index)
print("Predicted Value:", predicted_value)
print("Actual Value:", actual_value)

# Generate the waterfall plot
shap_values_instance = shap_values[instance_index]
shap.waterfall_plot(shap_values_instance)


# In[57]:


shap.plots.force(shap_values[0])


# In[121]:


# Create instance index 100
instance_index = 100
X_instance = X_test.iloc[instance_index]

# Get the predicted value
predicted_value = model.predict(X_instance.values.reshape(1, -1))[0]

# Get the actual value
actual_value = y_test.iloc[instance_index]

print("Instance Index:", instance_index)
print("Predicted Value:", predicted_value)
print("Actual Value:", actual_value)

# Generate the force plot with feature names
shap.force_plot(explainer.expected_value, shap_values[0], feature_names=feature_names)


# In[120]:


# Create instance index 20
instance_index = 20
X_instance = X_test.iloc[instance_index]

# Get the predicted value
predicted_value = model.predict(X_instance.values.reshape(1, -1))[0]

# Get the actual value
actual_value = y_test.iloc[instance_index]

print("Instance Index:", instance_index)
print("Predicted Value:", predicted_value)
print("Actual Value:", actual_value)

# Generate the force plot with feature names
shap.force_plot(explainer.expected_value, shap_values[0], feature_names=feature_names)


# In[122]:


# Create instance index 100
instance_index = 400
X_instance = X_test.iloc[instance_index]

# Get the predicted value
predicted_value = model.predict(X_instance.values.reshape(1, -1))[0]

# Get the actual value
actual_value = y_test.iloc[instance_index]

print("Instance Index:", instance_index)
print("Predicted Value:", predicted_value)
print("Actual Value:", actual_value)

# Generate the force plot with feature names
shap.force_plot(explainer.expected_value, shap_values[0], feature_names=feature_names)


# In[58]:


shap.plots.force(shap_values)


# In[130]:


# Create instance index 20
instance_index = 20
X_instance = X_test.iloc[instance_index]

# Get the predicted value
predicted_value = model.predict(X_instance.values.reshape(1, -1))[0]

# Get the actual value
actual_value = y_test.iloc[instance_index]

print("Instance Index:", instance_index)
print("Predicted Value:", predicted_value)
print("Actual Value:", actual_value)

# Generate the force plot with feature names
shap.plots.force(shap_values)


# ### Counterfactual

# In[71]:


# Create a DiceML data object
data = dice_ml.Data(dataframe=df, continuous_features=['GSCI', 'SP500', 'NASDAQ', 'DJIA', 'DXY', 'Federal_Funds_Effective', 'Economic_Uncertainty_Index', 'Inflation_5Y_BE', 'US_3M_Treasury', 'US_1Y_Treasury', '10Y_Less_2Y', 'US_Oil_Demand', 'US_Oil_Supply', 'Demand_Less_Supply'], outcome_name='WTI_Spot')

# Create a DiceML model object
model = dice_ml.Model(model=lgbm1, backend='sklearn', model_type='regressor')

# Create a DiceML expander object
expander = dice_ml.Dice(data, model)

# Generate counterfactual examples
counterfactuals = expander.generate_counterfactuals(X_test.iloc[0:1], total_CFs=3, desired_range=(0, 80))

# Visualize the counterfactual examples
counterfactuals.visualize_as_dataframe()


# In[72]:


features_to_vary=['GSCI', 'NASDAQ', 'DJIA']
permitted_range={'GSCI': [300,450], 'NASDAQ': [11000,15000], 'DJIA': [20000,25000]}
counterfactuals = expander.generate_counterfactuals(X_test.iloc[0:1], total_CFs=3, desired_range=(0, 80), permitted_range=permitted_range,
                                                   features_to_vary=features_to_vary)
counterfactuals.visualize_as_dataframe()


# In[136]:


# Create a DiceML data object
data = dice_ml.Data(dataframe=df, continuous_features=['GSCI', 'SP500', 'NASDAQ', 'DJIA', 'DXY', 'Federal_Funds_Effective', 'Economic_Uncertainty_Index', 'Inflation_5Y_BE', 'US_3M_Treasury', 'US_1Y_Treasury', '10Y_Less_2Y', 'US_Oil_Demand', 'US_Oil_Supply', 'Demand_Less_Supply'], outcome_name='WTI_Spot')

# Create a DiceML model object
model = dice_ml.Model(model=lgbm1, backend='sklearn', model_type='regressor')

# Create a DiceML expander object
expander = dice_ml.Dice(data, model)

# Generate counterfactual examples for instance index 20
counterfactuals = expander.generate_counterfactuals(X_test.iloc[400:401], total_CFs=3, desired_range=(0, 80))

# Visualize the counterfactual examples
counterfactuals.visualize_as_dataframe()


# ### Streamlit sav

# In[77]:


df = pd.read_csv("F:/Urooj/Masters/IBA/ML/assign 3/WTI Price.csv")
df=df.fillna(0)
image = Image.open(r"F:\Urooj\Masters\IBA\ML\assign 3\10-Applications-of-Machine-Learning-in-Oil-Gas1.jpg")
y = df['WTI_Spot'].copy()
X = df.drop('WTI_Spot', axis=1).copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
filename = 'WTI_OilPriceP.sav'
pickle.dump(lgbm1, open(filename, 'wb'))


# In[ ]:




