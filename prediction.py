import pandas as pd
from datetime import datetime, date
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.cluster import KMeans
import numpy as np 
# import plotly.offline as pyoff
# import plotly.graph_objs as go


#initate plotly
# pyoff.init_notebook_mode()

#read data from csv and redo the data work we done before

# retailData = pd.read_csv('/Users/sharonokeakwalam/Desktop/my-project/new_Online_Retail.csv')
# retailData['invoicedate'] = pd.to_datetime(retailData['InvoiceDate'])

# using France data
# france_data = retailData.query("Country=='France'").reset_index(drop=True)

# #  using 3 months data for RFM
# data_3M = france_data[(france_data.InvoiceDate < datetime(2011,6,1)) & (france_data.InvoiceDate >= datetime(2011,3,1))].reset_index(drop=True)

# # 6 months data
# data_6M = france_data[(france_data.InvoiceDate >= datetime(2011,6,1)) & (france_data.InvoiceDate < datetime(2011,12,1))].reset_index(drop=True)

# #create users for assigning clustering
# users = pd.DataFrame(data_3M['CustomerID'].unique())
# users.columns = ['CustomerID']

#order cluster method
def order_cluster(cluster_field_name, target_field_name, df, ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    dataNew = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    dataNew = dataNew.sort_values(by=target_field_name, ascending=ascending).reset_index(drop=True)
    dataNew['index'] = dataNew.index
    df_final = pd.merge(df, dataNew[[cluster_field_name, 'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name], axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    
    return df_final

def predict_values(data_frame, request_query):
	""" returns a data frame of predicted values

	"""
	response = {'data':{}, 'error':False, 'message':'Successfully predicted values'}
	#calculate recency score
	retailData = data_frame
	#  using 3 months data for RFM
	# {
	# 			"country": selected_country,
	# 			"rfm_start_date": rfm_start_date,
	# 			"rfm_end_date": rfm_end_date,
	# 			"prediction_start_date": prediction_start_date,
	# 			"prediction_end_date": prediction_end_date,
	# 		}

	rfm_start = request_query['rfm_start_date'].split('-')
	rfm_end = request_query['rfm_end_date'].split('-')
	prediction_start = request_query['prediction_start_date'].split('-')
	prediction_end = request_query['prediction_end_date'].split('-')
	
	data_3M = retailData[(retailData.invoicedate < datetime(int(rfm_end[0]),int(rfm_end[1]),int(rfm_end[2]))) &
	(retailData.invoicedate >= datetime(int(rfm_start[0]),int(rfm_start[1]),int(rfm_start[2])))].reset_index(drop=True)
	
	# 6 months data
	data_6M = retailData[(retailData.invoicedate >= datetime(int(prediction_start[0]),int(prediction_start[1]),int(prediction_start[2]))) & 
	(retailData.invoicedate < datetime(int(prediction_end[0]), int(prediction_end[1]), int(prediction_end[2])))].reset_index(drop=True)
	
	#create users for assigning clustering
	users = pd.DataFrame(data_3M['customerid'].unique())
	users.columns = ['customerid']


	max_purchase = data_3M.groupby('customerid').invoicedate.max().reset_index()
	max_purchase.columns = ['customerid','MaxPurchaseDate']
	max_purchase['Recency'] = (max_purchase['MaxPurchaseDate'].max() - max_purchase['MaxPurchaseDate']).dt.days
	users = pd.merge(users, max_purchase[['customerid','Recency']], on='customerid')

	


	kmeans = KMeans(n_clusters=4)
	kmeans.fit(users[['Recency']])
	users['RecencyCluster'] = kmeans.predict(users[['Recency']])
	users = order_cluster('RecencyCluster', 'Recency',users,False)



	#calcuate frequency score
	frequency = data_3M.groupby('customerid').invoicedate.count().reset_index()
	frequency.columns = ['customerid','Frequency']
	users = pd.merge(users, frequency, on='customerid')

	
	kmeans = KMeans(n_clusters=4)
	kmeans.fit(users[['Frequency']])
	users['FrequencyCluster'] = kmeans.predict(users[['Frequency']])
	users = order_cluster('FrequencyCluster', 'Frequency',users,True)

	
	#calcuate revenue/monetary value score
	data_3M['Monetary'] = pd.to_numeric(data_3M['unitprice']) * pd.to_numeric(data_3M['quantity'])
	revenue = data_3M.groupby('customerid').Monetary.sum().reset_index()
	users = pd.merge(users, revenue, on='customerid')

	

	kmeans = KMeans(n_clusters=4)
	kmeans.fit(users[['Monetary']])
	users['MonetaryCluster'] = kmeans.predict(users[['Monetary']])
	users = order_cluster('MonetaryCluster', 'Monetary',users,True)


	#overall scoring
	users['OverallScore'] = users['RecencyCluster'] + users['FrequencyCluster'] + users['MonetaryCluster']
	users.groupby('OverallScore')['Recency','Frequency','Monetary'].mean()

	users['Segment'] = 'Low-Value'
	users.loc[users['OverallScore']>2,'Segment'] = 'Mid-Value' 
	users.loc[users['OverallScore']>5,'Segment'] = 'High-Value' 

	
	response['data']['rfm_customer_count'] = users[['customerid']].count().to_dict()
	
	response['data']['rfm_customer_segment_count'] = users.groupby('Segment')['customerid'].count().to_dict()
	
	response['data']['rfm_customer_segment_monetary_value'] = users.groupby('Segment')['Monetary'].sum().to_dict()
	

	#  calculate 6 months LTV for each customer which will be used for training our model

	#calculate revenue/ monetary value and create a new dataframe for it
	data_6M['Revenue'] = pd.to_numeric(data_6M['unitprice']) * pd.to_numeric(data_6M['quantity'])
	users_data_6M = data_6M.groupby('customerid')['Revenue'].sum().reset_index()
	users_data_6M.columns = ['customerid','Revenue_6Mon']


	data_merge = pd.merge(users, users_data_6M, on='customerid', how='left')

	data_merge = data_merge.fillna(0)
	


	#remove outliers
	data_merge = data_merge[data_merge['Revenue_6Mon']<data_merge['Revenue_6Mon'].quantile(0.99)]




	#creating 3 clusters
	kmeans = KMeans(n_clusters=3)
	kmeans.fit(data_merge[['Revenue_6Mon']])
	data_merge['LTVCluster'] = kmeans.predict(data_merge[['Revenue_6Mon']])


	#order cluster number based on LTV
	data_merge = order_cluster('LTVCluster', 'Revenue_6Mon',data_merge,True)

	
	#creatinga new cluster dataframe
	data_cluster = data_merge.copy()

	data_cluster = data_cluster.fillna(0)

	
	
	response['data']['prediction_customer_count'] = data_cluster.groupby('LTVCluster')['customerid'].count().to_dict()
	
	response['data']['prediction_ltv_revenue'] = data_cluster.groupby('LTVCluster')['Revenue_6Mon'].sum().to_dict()
	
	response['data']['prediction_ltv_description'] = data_cluster.groupby('LTVCluster')['Revenue_6Mon'].describe()
	response['data']['prediction_ltv_description'] = response['data']['prediction_ltv_description'].fillna(0)
	response['data']['prediction_ltv_description'] = response['data']['prediction_ltv_description'].to_dict()

	return response