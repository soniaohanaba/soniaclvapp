# import libraries
# RECENCY, FREQUENCY AND MONETRY VALUE

import pandas as pd

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

global users
global retailData




def order_cluster(cluster_field_name, target_field_name, data_frame, ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    dataNew = data_frame.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    dataNew = dataNew.sort_values(by=target_field_name, ascending=ascending).reset_index(drop=True)
    dataNew['index'] = dataNew.index
    df_final = pd.merge(data_frame, dataNew[[cluster_field_name, 'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name], axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    
    return df_final

def cluster_values(data_frame):
	# # specifically selects the customerID
	retailData = data_frame
	users = pd.DataFrame(retailData['customerid'].unique())
	users = retailData.drop_duplicates(['customerid','country'])[['customerid','country']]
	
	# Recency
	#  date where the last purchase took place per customerID
	max_purchase = data_frame.groupby('customerid').invoicedate.max().reset_index()
	max_purchase.columns = ['customerid', 'MaxPurchaseDate']

	max_purchase['Recency'] = (max_purchase['MaxPurchaseDate'].max() - max_purchase['MaxPurchaseDate']).dt.days

	# refers to the last time that a customer made a purchase.
	users = pd.merge(users, max_purchase[['customerid', 'Recency']], on='customerid')


	kmeans = KMeans(n_clusters=4)
	kmeans.fit(users[['Recency']])
	users['RecencyCluster'] = kmeans.predict(users[['Recency']])

	#order the recency cluster
	users = order_cluster('RecencyCluster', 'Recency', users, True)


	# FREQUENCY
	#get order counts for each user and create a dataframe with it
	frequency = data_frame.groupby('customerid').invoicedate.count().reset_index()
	frequency.columns = ['customerid','Frequency']

	# frequency
	#add this data to our main dataframe
	users = pd.merge(users, frequency, on='customerid')

	#k-means
	kmeans = KMeans(n_clusters=4)
	kmeans.fit(users[['Frequency']])
	users['FrequencyCluster'] = kmeans.predict(users[['Frequency']])

	#order the frequency cluster
	users = order_cluster('FrequencyCluster', 'Frequency',users,True)


	# MONETARY
	#calculate monetary value for each customer
	
	data_frame['Monetary'] = pd.to_numeric(data_frame['unitprice']) * pd.to_numeric(data_frame['quantity'])
	monetary = data_frame.groupby('customerid').Monetary.sum().reset_index()

	# #merge it with our main dataframe
	users = pd.merge(users, monetary, on='customerid')


	#k-means
	kmeans = KMeans(n_clusters=4)
	kmeans.fit(users[['Monetary']])
	users['MonetaryCluster'] = kmeans.predict(users[['Monetary']])

	#order the frequency cluster
	users = order_cluster('MonetaryCluster', 'Monetary',users,True)



	# TO MERGE RECENCY, FREQUENCY AND MONETARY
	users['OverallScore'] = pd.to_numeric(users['RecencyCluster'] + users['FrequencyCluster'] + users['MonetaryCluster'])
	
	#  knowing the overall scores, it is time to divide them into segments: low, mid and high. Where 0-2 = low, 3-4 = mid, 5-6 = high
	users['Segment'] = 'Low-Value'
	users.loc[users['OverallScore']>2,'Segment'] = 'Mid-Value' 
	users.loc[users['OverallScore']>4,'Segment'] = 'High-Value' 

	return users






