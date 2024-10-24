import pandas as pd
import numpy as np
import random
import statistics
import math
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
    
def remove_outliers(df, column):
    # Normalize the column
    mean = df[column].mean()
    std = df[column].std()
    normalized = (df[column] - mean) / std
    
    Q1 = normalized.quantile(0.25)
    Q3 = normalized.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Create a boolean mask for rows to keep
    mask = (normalized >= lower_bound) & (normalized <= upper_bound)
    
    # Apply the mask to the dataframe
    df_filtered = df[mask]
    
    return df_filtered

    
class Preprocess:
    # load data (in the format of dataframe)
    # user provides dimensions (these dimensions will be pre-aggregated)
    # name should be unique across datasets
    def load(self, data, join_key, divide_by_max=True):
        self.data = data
        self.join_key = join_key
        self.X = []
        self.find_features()
        self.to_numeric_and_impute_all(divide_by_max)
    
    def agg_by_jk(self):
        self.data = self.data.groupby(
            self.join_key)[self.X].mean().reset_index()
    
    # iterate all attributes and find candidate features
    def find_features(self):
        atts = []
        for att in self.data.columns:
            if att != self.join_key and self.is_feature(att, 0.6, 10):
                atts.append(att)
                
        self.X = atts
        
    def is_feature(self, att, pct, unique_val):
        self.to_numeric(att)
        col = self.data[att]
        missing = sum(np.isnan(col))/len(self.data)
        distinct = len(col.unique())
        mean_value = col.mean()
        
        if missing < pct and distinct > unique_val and not np.isinf(
            mean_value):
            return True
        else:
            return False
        
    # this is the function to transform an attribute to number
    def to_numeric(self, att):
        # parse attribute to numeric
        self.data[att] = pd.to_numeric(self.data[att],errors="coerce")
    
    def impute_mean(self, att):
        mean_value=self.data[att].mean()
        self.data[att].fillna(value=mean_value, inplace=True)
        
    def to_numeric_and_impute_all(self, divide_by_max):
        new_X = []
        for att in self.X:
            self.to_numeric(att)
            self.impute_mean(att)
            if divide_by_max:
                self.data[att] /= np.abs(self.data[att].values).max()
                # self.data[att] /= self.data[att].mean()
            if self.data[att].std() > 0.1:
                new_X.append(att)
        self.X = new_X
        
