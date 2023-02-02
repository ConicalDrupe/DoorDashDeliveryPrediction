# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 12:58:36 2023

@author: Christopher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# read data

historical_data = pd.read_csv(r"C:\Users\Christopher\Documents\DataProjects2022\DoorDash\historical_data.csv")
historical_data.head()

#%%
# Simple EDA and transformation

historical_data.info() #we see created_at and actual_delivery_time
#are objects and not datetime, so lets change that

historical_data["created_at"] = pd.to_datetime(historical_data["created_at"])
historical_data["actual_delivery_time"] = pd.to_datetime(historical_data["actual_delivery_time"])

#%% Feature Creation: 
#Create target variable: Delivery time and % workers
#Create Busy Dashers ratio % = Total busy dashers/Total onshift dashers
#Create Estimated time it takes to deliver order, without restraunt prep

historical_data["actual_total_delivery_duration"] = (historical_data["actual_delivery_time"] - historical_data["created_at"])

historical_data["busy_dashers_ratio"] = historical_data["total_busy_dashers"] / historical_data["total_onshift_dashers"]

historical_data["estimated_non_prep_duration"] = historical_data["estimated_store_to_consumer_driving_duration"] + historical_data["estimated_order_place_duration"]

#%% Check ids and decide wheather to encode or not
historical_data["market_id"].nunique() #returns 6
historical_data["store_id"].nunique() #returns 6743
historical_data["order_protocol"].nunique() #returns 7

#%% Create dummies for order protocol and market id

order_protocol_dummies = pd.get_dummies(historical_data.order_protocol)
order_protocol_dummies = order_protocol_dummies.add_prefix('order_protocol_')

market_id_dummies = pd.get_dummies(historical_data.market_id)
market_id_dummies = market_id_dummies.add_prefix('market_id_')

#%% Both market_id and order_protocol have null entries, we will map each store_id
# to the most frequent cuisine_category they have - to fill null values when possible

store_id_unique = historical_data["store_id"].unique().tolist()
store_id_and_category = {store_id: historical_data[historical_data.store_id == store_id].store_primary_category.mode()
                         for store_id in store_id_unique}


def fill(store_id):
    #return primary store category from the dictionary
    try:
        return store_id_and_category[store_id].values[0]
    except:
        return np.nan
    
#fill null values
historical_data["nan_free_store_primary_category"] = historical_data.store_id.apply(fill)

#%% One-Hot Encoding store_primary_category

store_primary_category_dummies = pd.get_dummies(historical_data.nan_free_store_primary_category)
store_primary_category_dummies = store_primary_category_dummies.add_prefix('category_')

#%% Cleaning unused columns, and putting in One-hot encoding columns
train_df = historical_data.drop(columns =
 ["created_at","market_id","store_id","store_primary_category","actual_delivery_time","nan_free_store_primary_category","order_protocol"])

train_df.head()

#%% Concat all
train_df = pd.concat([train_df, order_protocol_dummies, market_id_dummies, store_primary_category_dummies], axis=1)

#align dtype over dataset
train_df = train_df.apply(np.float32)
train_df.head()

#%% Inspect Busy Dashers ratio

train_df["busy_dashers_ratio"].describe()

#check infinity values wih using numpy isfinite() function

np.where(np.any(~np.isfinite(train_df),axis=0) == True)

#replace inf values with nan to drop all nans

train_df.replace([np.inf, -np.inf], np.nan, inplace=True)

#drop all nans
train_df.dropna(inplace=True)

train_df.shape
