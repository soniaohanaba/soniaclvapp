from flask import Flask, g,  Response, request, render_template, redirect, session, flash, url_for
import uuid
from werkzeug.utils import secure_filename
import os
from cleaning import clean_file
from utilities import allowed_file
import rfm
import prediction
import json
from datetime import timedelta

UPLOAD_FOLDER = './uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = uuid.uuid4().hex
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024




@app.route('/')
def home():
	return render_template("home.html")

@app.route('/index')
def index():
	return render_template("index.html")

@app.route('/about')
def about():
	return render_template("about.html")

@app.route('/upload', methods=['POST','GET'])
def upload_data():
	if request.method == 'POST':
		if 'file' not in request.files:
			print("inexistent file")
			return redirect(request.url)
		file = request.files['file']
		# if user didn't submit any file
		if file.filename == '':
			print("no file name")
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			file.save(file_path)
			
			session['file_path'] = file_path
			session.permanent = True
			return redirect("/upload_response")

	return render_template("upload.html")

@app.route('/upload_response', methods=['GET'])
def upload_response():

	return render_template("upload_response.html")

@app.route('/describe')
def describe():
	""" A route handler for preparing uploaded file for basic description

	"""
	file_path = session.get('file_path')
	
	if file_path is None:
		return redirect('/upload')
	data_frame = clean_file(file_path)
	if data_frame['error'] == True:
		flash(data_frame['message'])
		return redirect('/upload')

	dictionary = data_frame['data'].to_dict('records')
	list_of_rows = list(dictionary)
	for row in list_of_rows:
		row['invoicedate'] = str(row['invoicedate'])
	return render_template("describe.html", data=json.dumps(list_of_rows))


@app.route('/rfm')
def rfm_description():
	print("session file is ",session.get('file_path'))
	file_path = session.get('file_path')
	
	if file_path is None:
		return redirect('/upload')

	data_frame = clean_file(file_path)
	if data_frame['error'] == True:
		flash(data_frame['message'])
		return redirect('/upload')

	clustered_frame = rfm.cluster_values(data_frame['data'])
	cluster_dictionary = clustered_frame.to_dict('records')
	list_of_rows = list(cluster_dictionary)
	
	return render_template("rfm_model.html", data=json.dumps(list_of_rows))

@app.route('/predict')
def predict():
	if request.args.get('country'):
		print("how far na")
		return {'message': "hello world", 'error': False}
	print("session file is ", session.get('file_path'))
	file_path = session.get('file_path')

	if file_path is None:
		flash("Please upload a file before trying to predict")
		return redirect('/upload')

	data_frame = clean_file(file_path)
	if data_frame['error'] == True:
		flash(data_frame['message'])
		return redirect('/upload')

	predicted_data_frame = prediction.predict_values(data_frame['data'])

	return render_template("prediction.html", data=json.dumps([]))

if __name__ == '__main__':
	app.run(debug=True)
