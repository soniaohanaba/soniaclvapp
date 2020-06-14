import csv
import pandas as pd
import json

# customer_id, quantity, invoicedate, unitprice

def is_any_field_empty(row):
	for field in row:
		if row[field] is None:
			return True
	return False


def clean_file(file):
	""" Reads the file, converts it to a dataframe and returns a dataframe object
	"""
	response = {'message': "Successfully cleaned file", 'error':True, 'data':None}
	file_extension = file[file.rfind('.'):]
	try:
		if file_extension == '.json':
			df = pd.read_json(file, index_col=0, header='infer', squeeze=True)
		elif file_extension == '.csv':
			df = pd.read_csv(file, index_col=0, header='infer', squeeze=True)
		elif file_extension == '.xlsx':
			df = pd.read_excel(file, index_col=0, header='infer', squeeze=True)
	except Exception as e:
		response['message'] = "There was an error reading file"
		return response
		
	df = df.dropna(how='any', axis=0)
	df = df.apply(lambda x: x.astype(str).str.lower())
	df.columns = df.columns.str.replace(' ', '')
	df.columns = [x.lower() for x in df.columns]

	necessary_columns = ['unitprice', 'invoicedate', 'customerid', 'quantity']
	for column in necessary_columns:
		if column not in df.columns:
			response['message'] = 'You must provided {} column in your dataset. Please upload another file'.format(column)
			return response

	df['invoicedate'] = pd.to_datetime(df['invoicedate'])
	
	response['data'] = df 
	response['error'] = False
	return response

