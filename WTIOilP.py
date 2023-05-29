#!/usr/bin/env python
# coding: utf-8

# In[13]:

pip install --upgrade streamlit

import pandas as pd   
from utils import DataLoader
from numpy import mean
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from pycaret.regression import load_model, predict_model
import lightgbm as lgb
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


# In[14]:

data_loader = DataLoader()
data_loader.load_dataset()
# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()

st.set_page_config(layout="wide")
image = Image.open(r"10-Applications-of-Machine-Learning-in-Oil-Gas1.jpg")

# Display the image above the title
st.image(image)

filename = 'WTI_OilPriceP.sav'
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
    st.write(f"Closing price of GSCI Commodities Index: {lower_limit3} - {upper_limit3}")

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
#Fit the model to your training data
lgbm1=lgbm.fit(X_train, y_train)
model = lgbm1

#LIME
explainer = LimeTabularExplainer(training_data=X_train.values, mode='regression', feature_names=X_train.columns.tolist())
# Select a single instance from X_test
X_test_instance = X_test.iloc[-20]
# Explain the instance using Lime
explanation = explainer.explain_instance(X_test_instance.values, model.predict, num_features=10)
# Display the explanation
explanation.show_in_notebook(show_table=True)

#SHAP

explainer = shap.Explainer(model)
# Calculate the SHAP values
shap_values = explainer(X_test)
# Generate the waterfall plot
shap.plots.waterfall(shap_values[0])

# Define a function to display SHAPLEY explanations in Streamlit
def st_shap(plot, height=None, width=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

#CounterFactual
def main():
    # Create a DiceML data object
    data = dice_ml.Data(dataframe=df, continuous_features=['GSCI', 'SP500', 'NASDAQ', 'DJIA', 'DXY', 'Federal_Funds_Effective', 'Economic_Uncertainty_Index', 'Inflation_5Y_BE', 'US_3M_Treasury', 'US_1Y_Treasury', '10Y_Less_2Y', 'US_Oil_Demand', 'US_Oil_Supply', 'Demand_Less_Supply'], outcome_name='WTI_Spot')

    # Create a DiceML model object
    model1 = dice_ml.Model(model=model, backend='sklearn', model_type='regressor')

    # Create a DiceML expander object
    expander = dice_ml.Dice(data, model1)

    # Generate counterfactual examples
    counterfactuals = expander.generate_counterfactuals(X_test.iloc[0:1], total_CFs=3, desired_range=(0, 80))

    # Visualize the counterfactual examples
    st.write(counterfactuals.visualize_as_dataframe())

    features_to_vary = ['GSCI', 'NASDAQ', 'DJIA']
    permitted_range = {'GSCI': [300, 450], 'NASDAQ': [11000, 15000], 'DJIA': [20000, 25000]}
    counterfactuals = expander.generate_counterfactuals(X_test.iloc[0:1], total_CFs=3, desired_range=(0, 80), permitted_range=permitted_range,
                                                       features_to_vary=features_to_vary)
    # Visualize the counterfactual examples
    st.write(counterfactuals.visualize_as_dataframe())

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
    
st.subheader("CounterFactual Explanation")
if st.button("CounterFactual"):
    if __name__ == '__main__':
        main()


# In[ ]:




