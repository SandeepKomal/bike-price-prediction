import numpy as np
from flask import Flask, render_template,url_for, request, redirect
import pickle
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



app = Flask(__name__)
# model=pickle.load(open('olxpick.pkl', 	'rb'))
model =joblib.load('simifinal.pkl')



@app.route('/')
@app.route('/main')
def main():
  return render_template('main.html')


@app.route('/predict',methods=['POST'])
def predict():
	int_features =[x for x in request.form.values()]
	print("Hello")
	print(int_features)
	check=[int_features]
	check=pd.DataFrame(check,columns=["km_driven","make_year","bike_name","bike_model","state","city"])
	ohot =joblib.load('ohe.joblib')
	dino = pd.DataFrame(ohot.transform(check.iloc[:,2:6]))
	dino.columns =ohot.get_feature_names_out()
	check =pd.concat([check.iloc[:,0:2],dino],axis=1)
	output = model.predict(check)
	print(output)
	
	# # x=pd.get_dummies(int_features,drop_first=True)
	# print(x)
	# final_features = [np.array(x).ravel()]
	# print(final_features)
	# prediction = model.predict(final_features)
	# print("Hello2")
	return render_template('main.html',prediction_text="Your Bike Estimated Cost is INR : {}".format(output))
	

if __name__ == "__main__":
	app.debug = True
	app.run(host = '0.0.0.0', port =7000)

	
