#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader():
    def __init__(self):
        self.data = None

    def load_dataset(self, path="WTI Price.csv"):
        self.data = pd.read_csv(path)

        # Standardization
        # Usually we would standardize here and convert it back later
        # But for simplification we will not standardize / normalize the features

    def preprocess_data(self):
        # Drop 'DATE' column
        self.data.drop('DATE', axis=1, inplace=True)
        # Fill null values with column means
        self.data.fillna(self.data.mean(), inplace=True)

    def get_data_split(self):
        self.preprocess_data()
        X = self.data.drop('WTI_Spot', axis=1).copy()
        y = self.data['WTI_Spot'].copy()
        return train_test_split(X, y, test_size=0.3)


# In[ ]:





# In[ ]:




