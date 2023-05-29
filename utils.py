#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd 
# Makes sure we see all columns
from sklearn.model_selection import train_test_split

class DataLoader():
    def __init__(self):
        self.data = None

    def load_dataset(self, path="WTI Price.csv"):
        self.data = pd.read_csv(path)

        # Standardization 
        # Usually we would standardize here and convert it back later
        # But for simplification we will not standardize / normalize the features

    def get_data_split(self):
        X = self.data.iloc[:,:-1]
        y = self.data.iloc[:,-1]
        return train_test_split(X, y, test_size=0.30)


# In[ ]:





# In[ ]:




