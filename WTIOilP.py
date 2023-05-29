#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer

import shap
# Run initjs() function
shap.initjs()

import dice_ml
import joblib

import streamlit as st
import pickle
import datetime
import os
from PIL import Image
import streamlit.components.v1 as components
import io


# In[ ]:


file_path = "WTI Price.csv"
if os.path.isfile(file_path):
    df = pd.read_csv(file_path)
else:
    st.error("File not found. Please check the file path.")

df.drop('DATE', axis=1, inplace=True)
df.fillna(df.mean(), inplace=True)

y = df['WTI_Spot'].copy()
X = df.drop('WTI_Spot', axis=1).copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

st.set_page_config(layout="wide")
image = Image.open(r"10-Applications-of-Machine-Learning-in-Oil-Gas1.jpg")

# Display the image above the title
st.image(image)

filename = 'WTI_OILPrice.sav'
loaded_model = pickle.load(open(filename, 'rb'))

st.title('West Texas Intermediate (WTI) Oil Price Prediction Web App')
st.write('This is a web app to predict the price of Crude Oil using several features that you can see in the sidebar. Please adjust the value of each feature. After that, click on the Predict button at the bottom to see the prediction of the regressor.')

import streamlit as st

# Sidebar styling
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #F0F5F9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("App Information")
st.sidebar.markdown("**Created by:** Urooj Mumtaz Joyo")
st.sidebar.markdown("**Project:** Machine Learning ")

with st.sidebar:
    ffe = st.slider(label='Effective Federal Funds Rate', min_value=0.01, max_value=4.00, value=0.04, step=0.02)
    EUI = st.slider(label='Metric for Economic Policy Uncertainty', min_value=0.0, max_value=810.0, value=3.32, step=10.0)
    Inf = st.slider(label='5-year Break-Even Inflation Rate', min_value=0.0, max_value=4.0, value=0.14, step=0.05)
    DXY = st.slider(label='Closing price of DXY', min_value=79.0, max_value=115.0, value=79.14, step=1.0)

   # Define the range options for Monthly US Oil Demand - Monthly US Oil Supply
    range_options = {
        2000: (2000, 2999),
        3000: (3000, 3999),
        4000: (4000, 4999),
        5000: (5000, 5999),
        6000: (6000, 6999),
        7000: (7000, 7999),
        8000: (8000, 8999),
        9000: (9000, 9999),
        10000: (10000, 10999),
        11000: (11000, 11999),
    }

    # Display the range options as tabs for Monthly US Oil Demand - Monthly US Oil Supply
    range_selected = st.radio('Monthly US Oil Demand - Monthly US Oil Supply', list(range_options.keys()))

    # Get the selected range for Monthly US Oil Demand - Monthly US Oil Supply
    selected_range = range_options[range_selected]

    # Extract the lower and upper limits of the selected range for Monthly US Oil Demand - Monthly US Oil Supply
    lower_limit, upper_limit = selected_range

    # Example usage of the selected range for Monthly US Oil Demand - Monthly US Oil Supply
    st.write(f"Monthly US Oil Demand - Monthly US Oil Supply: {lower_limit} - {upper_limit}")
    
    # Define the range options for NASDAQ
    range_options1 = {
        3000: (3000, 3999),
        4000: (4000, 4999),
        5000: (5000, 5999),
        6000: (6000, 6999),
        7000: (7000, 7999),
        8000: (8000, 8999),
        9000: (9000, 9999),
        10000: (10000, 10999),
        11000: (11000, 11999),
        12000: (12000, 12999),
        13000: (13000, 13999),
        14000: (14000, 14999),
        15000: (15000, 15999),
        16000: (16000, 16999)
    }
    # Display the range options as tabs for NASDAQ
    range_selected1 = st.radio('Closing price of NASDAQ', list(range_options1.keys()))

    # Get the selected range for NASDAQ
    selected_range1 = range_options1[range_selected1]

    # Extract the lower and upper limits of the selected range for NASDAQ
    lower_limit1, upper_limit1 = selected_range1

    # Example usage of the selected range for NASDAQ
    st.write(f"Closing price of NASDAQ: {lower_limit1} - {upper_limit1}")
    
    # Define the range options for NASDAQ
    range_options2 = {
        1000: (1000, 1999),
        2000: (2000, 2999),
        3000: (3000, 3999),
        4000: (4000, 4999),
        5000: (5000, 5999),
    }
    # Display the range options as tabs for NASDAQ
    range_selected2 = st.radio('Closing price of S&P 500', list(range_options2.keys()))

    # Get the selected range for NASDAQ
    selected_range2 = range_options2[range_selected2]

    # Extract the lower and upper limits of the selected range for NASDAQ
    lower_limit2, upper_limit2 = selected_range2

    # Example usage of the selected range for NASDAQ
    st.write(f"Closing price of S&P 500: {lower_limit2} - {upper_limit2}")
    
    range_options3 = {
        200: (200, 299),
        300: (300, 399),
        400: (400, 499),
        500: (500, 599),
        600: (600, 699),
        700: (700, 799),
        800: (800, 899)
    }
    # Display the range options as tabs for NASDAQ
    range_selected3 = st.radio('Closing price of GSCI Commodities Index', list(range_options3.keys()))

    # Get the selected range for NASDAQ
    selected_range3 = range_options3[range_selected3]

    # Extract the lower and upper limits of the selected range for NASDAQ
    lower_limit3, upper_limit3 = selected_range3

    # Example usage of the selected range for NASDAQ
    st.write(f"Closing price of S&P 500: {lower_limit3} - {upper_limit3}")

  # Date selector
    start_date = st.date_input("Start Date", value=datetime.date(2013, 1, 1))
    end_date = st.date_input("End Date", value=datetime.date(2023, 12, 31))

# Convert the date range to Unix timestamps
    start_datetime = datetime.datetime.strptime(str(start_date), "%Y-%m-%d")
    end_datetime = datetime.datetime.strptime(str(end_date), "%Y-%m-%d")
    start_timestamp = int(start_datetime.timestamp())
    end_timestamp = int(end_datetime.timestamp())

# Create the features dictionary
features = {
    'ffe': ffe,
    'EUI': EUI,
    'Inf': Inf,
    'DXY': DXY,
    'lower_limit': lower_limit,
    'upper_limit': upper_limit,
    'lower_limit1': lower_limit1,
    'upper_limit1': upper_limit1,
    'lower_limit2': lower_limit2,
    'upper_limit2': upper_limit2,
    'lower_limit3': lower_limit3,
    'upper_limit3': upper_limit3,
    'start_date': start_timestamp,
    'end_date': end_timestamp
}

features_df = pd.DataFrame([features])
st.table(features_df)
col1, col2 = st.columns((1,2))

#XAI

lgbm = LGBMRegressor()
# Fit the model to your training data
lgbm1=lgbm.fit(X_train, y_train)

#LIME
explainer = LimeTabularExplainer(training_data=X_train.values, mode='regression', feature_names=X_train.columns.tolist())
# Select a single instance from X_test
X_test_instance = X_test.iloc[-20]
# Explain the instance using Lime
explanation = explainer.explain_instance(X_test_instance.values, lgbm.predict, num_features=10)
# Display the explanation
explanation.show_in_notebook(show_table=True)

#SHAP
model = lgbm1
explainer = shap.Explainer(model)
# Calculate the SHAP values
shap_values = explainer(X_test)
# Generate the waterfall plot
shap.plots.waterfall(shap_values[0])

# Define a function to display SHAPLEY explanations in Streamlit
def st_shap(plot, height=None, width=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Perform the prediction when the Predict button is clicked
with col1:
    prButton = st.button('Predict')
with col2: 
    if prButton:    
        prediction = loaded_model.predict(features_df)    
        st.write(' Based on the selected features, the predicted price of Crude Oil is:'+ str(int(prediction)))  
        
# Display Lime explanation
st.subheader("Lime Explanation")
if st.button('LIME'):
    st.write('''It shows a table that ranks the features based on their importance for the prediction and provides the 
             corresponding weight or contribution of each feature.''')
    html = explanation.as_html()
# Display the Lime plot in Streamlit
    components.html(html, height=800)
    
# Display SHAPLEY 
st.subheader("SHAPLEY Explanation")
if st.button("SHAPLEY"):
    explanation = shap_values[0]
    st.subheader('SHAP Force Plot (Horizontal Bar Chart):')
    st.write('''It shows the contribution of each feature towards the final prediction by displaying the impact of each 
             feature value on the predicted outcome. The force plot typically consists of a horizontal bar chart 
             with positive and negative values, representing the positive and negative contributions of the features. 
             The length of each bar indicates the magnitude of the feature's contribution. The bars are arranged in 
             descending order based on their contribution, with the most significant features at the top.''')
    st.write('')
    st_shap(shap.force_plot(shap_values[0]))
    st.subheader('SHAP Force Plot (Graph):')
    st.write('''The force plot visualizes the Shapley values of the features for multiple instances or predictions. It provides 
             an interactive and hierarchical representation of the feature contributions. Each instance is represented by a 
             vertical column of bars, where each bar corresponds to a feature's contribution to the prediction. The color of 
             the bar represents the feature's value, with blue indicating lower values and red indicating higher values. The 
             length of the bar represents the magnitude of the feature's contribution.''')
    st_shap(shap.plots.force(shap_values), height=800, width=1200)


# In[ ]:



