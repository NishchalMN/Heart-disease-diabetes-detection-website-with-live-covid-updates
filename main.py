import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.simplefilter("ignore", category=UserWarning)
import requests
import time
import smtplib
from bs4 import BeautifulSoup

global TO
TO = ["nishchalmn619@gmail.com"]

from sklearn.externals import joblib 

from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
	return render_template("index.html")

@app.route("/doctor")
def doctor():
    return render_template("doctor.html")

@app.route("/about_heart")
def about_heart():
    return render_template("heart.html")

@app.route("/get_chunk", methods=["GET", "POST"])
def get_chunk():
	count = request.form['count']
	count = int(count)
	chunk = 108;
	pos = count*chunk;
	f = open("content.txt", "r", errors='ignore');
	f.seek(pos)
	data = f.read(chunk)
	return data

@app.route("/about_diabetes")
def about_diabetes():
    return render_template("diabetes.html")

@app.route("/get_dia", methods=["GET", "POST"])
def get_dia():
	count = request.form['count']
	count = int(count)
	chunk = 108;
	pos = count*chunk;
	f = open("contentdiabetes.txt", "r", errors='ignore');
	f.seek(pos)
	data = f.read(chunk)
	return data

@app.route("/heart_process", methods=["GET", "POST"])
def heart_process():
	if request.method == "POST":
		print('came in')
		name = request.form["name"]
		age = request.form["age"]
		sex = request.form["sex"]
		print(type(sex), sex)
		if sex=='M':
		    sex = int(1)
		else:
		    sex=int(0)

		cp = int(request.form["cp"])
		trestbps = int(request.form["trestbps"])
		chol = int(request.form["chol"])
		fbs = int(request.form["fbs"])
		restecg = int(request.form["restecg"])
		thalac = int(request.form["thalac"])
		exang = int(request.form["exang"])
		oldpeak = float(request.form["oldpeak"])
		slope = int(request.form["slope"])
		ca = int(request.form["ca"])
		thal = int(request.form["thal"])

		test = np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalac,exang,oldpeak,slope,ca,thal])
		test = test.reshape(1,-1)
		
		dataset = pd.read_csv("heart.csv")

		dataset.head()
		X = dataset.iloc[:, :-1].values
		y = dataset.iloc[:,-1].values

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 101)

		sc = StandardScaler()
		X_train = sc.fit_transform(X_train)
		test = sc.transform(test)

		print('text processed')
		print('results are:')

		classifierSVM = joblib.load('models/h_svm.pkl')
		classifierDT = joblib.load('models/h_dt.pkl')
		classifierRF = joblib.load('models/h_rf.pkl')
		classifierNB = joblib.load('models/h_nb.pkl')
		classifierKNN = joblib.load('models/h_knn.pkl')


		y_predSVM = classifierSVM.predict(test)
		y_predRF = classifierRF.predict(test)
		y_predDT = classifierDT.predict(test) 
		y_predNB = classifierNB.predict(test) 
		y_predKNN = classifierKNN.predict(test)

		results = [int(y_predSVM),int(y_predRF),int(y_predDT),int(y_predNB),int(y_predKNN)]

		if(results.count(1) > results.count(0)):
		    return "Positive"
		else:
		    return "Negative"


@app.route("/pima_process", methods=["GET", "POST"])
def pima_process():
	if request.method == "POST":
		print('came in2')
		name = request.form["name"]
		age = int(request.form["age"])
		sex = request.form["sex"]
		preg = int(request.form["preg"])
		glucose = int(request.form["glucose"])
		bp = int(request.form["bp"])
		st = int(request.form["st"])
		insulin = int(request.form["insulin"])
		bmi = float(request.form["bmi"])
		dpf = float(request.form["dpf"])

		test = np.array([preg,glucose,bp,st,insulin,bmi,dpf,age])
		test = test.reshape(1,-1)
		
		dataset = pd.read_csv("diabetes.csv")

		dataset.head()
		X = dataset.iloc[:, :-1].values
		y = dataset.iloc[:,-1].values

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 101)

		sc = StandardScaler()
		X_train = sc.fit_transform(X_train)
		test = sc.transform(test)
		
		print('text processed')
		print('results are:')

		classifierSVM = joblib.load('models/d_svm.pkl')
		classifierDT = joblib.load('models/d_dt.pkl')
		classifierRF = joblib.load('models/d_rf.pkl')
		classifierNB = joblib.load('models/d_nb.pkl')
		classifierKNN = joblib.load('models/d_knn.pkl')


		y_predSVM = classifierSVM.predict(test)
		y_predRF = classifierRF.predict(test)
		y_predDT = classifierDT.predict(test) 
		y_predNB = classifierNB.predict(test) 
		y_predKNN = classifierKNN.predict(test)

		results = [int(y_predSVM),int(y_predRF),int(y_predDT),int(y_predNB),int(y_predKNN)]
		
		if(results.count(1) > results.count(0)):
		    return "Positive"
		else:
		    return "Negative"

@app.route("/covid_process", methods=["GET", "POST"])
def covid_process():
	if request.method == "POST":
		print('came in3')
		email = request.form["email"]
		subs = request.form["subscribe"]

		global TO
		if(subs == 'Y'):
			TO.append(email)
		else:
			try:
				TO.remove(email)
				return("Your email is removed from subscriptions")
			except:
				return("You have not subscribed yet!")

		# email
		def send_email(TEXT):
		  global TO
		  print(TO)
		  FROM = "nishchalmn619files@gmail.com"
		  # TO = ["nishchalmn619backup@gmail.com", "rohan.m.rohan7@gmail.com", "sumananjunda76@gmail.com"]
		  # T = ["nishchalmn619@gmail.com", "nishchalmn619backup@gmail.com"]
		  SUBJECT = "COVID-19 UPDATE"
		  TO = list(set(TO))
		  message = """From: %s\nTo: %s\nSubject: %s\n\n%s
		  """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
		  try:
		      server = smtplib.SMTP("smtp.gmail.com", 587)
		      server.ehlo()
		      server.starttls()
		      server.login("nishchalmn619files@gmail.com", "Ritheshmn@13")
		      server.sendmail(FROM, TO, message)
		      server.close()
		      print('successfully sent the mail')
		  except:
		      print("failed to send mail")
		
		states = {'Andhra Pradesh': 348, 'Andaman and Nicobar Islands': 11, 'Bihar': 39, 'Chandigarh': 18, 'Chhattisgarh': 10, 'Delhi': 720, 'Goa': 7, 'Gujarat': 241, 'Haryana': 169, 'Himachal Pradesh': 18, 'Jammu and Kashmir': 158, 'Karnataka': 181, 'Kerala': 357, 'Ladakh': 15, 'Madhya Pradesh': 259, 'Maharashtra': 1364, 'Manipur': 2, 'Mizoram': 1, 'Odisha': 44, 'Puducherry': 5, 'Punjab': 101, 'Rajasthan': 463, 'Tamil Nadu': 834, 'Telengana': 442, 'Uttarakhand': 35, 'Uttar Pradesh': 410, 'West Bengal': 116, 'total': 6412, 'Assam': 29, 'Jharkhand': 13, 'Arunachal Pradesh': 1, 'Tripura': 1}
		print(states)

		# states['Karnataka'] = 10
		# states['Kerala'] = 0
		print('started')
		# periodic refresh and scanning start
		for interval in range(1):
		  try:
		    url = 'https://www.mohfw.gov.in/'
		    r = requests.get(url, headers={'Cache-Control': 'no-cache'})

		    soup = BeautifulSoup(r.content, 'html5lib')
		    table1 = soup.find_all('tbody')

		    flag = 0
		    msg = ['Covid 19 Update as per the govt\n']
		    current = 0
		    # print(states['Karnataka'])

		    for tbody in table1:
		      try:
		        table2 = tbody.find_all('tr')
		        for row in table2:
		          cells = row.find_all('td')
		          if(len(cells[0].text) <= 2):
		            current += int(cells[2].text)
		            if(cells[1].text in states):
		              if(int(cells[2].text) > states[cells[1].text]):
		                flag = 1
		                msg.append('Increase in ' + cells[1].text + ' from ' + str(states[cells[1].text]) + ' to ' + cells[2].text)
		                states[cells[1].text] = int(cells[2].text)
		            else:
		              flag = 1
		              states[cells[1].text] = int(cells[2].text)
		              msg.append('New state added which is ' + cells[1].text + ' with ' + cells[2].text + ' cases')
		      except Exception as e:
		        print(e)

		    if(flag == 1):
		      # msg.append('No new cases found in last 30 minutes :)')
		      msg.append('\nTotal number of cases have reached ' + 'from ' + str(states['total']) + ' to ' + str(current) + ' !')

		      msg.append('Be safe, stay indoors and take care.')
		      msg.append('\n--This is an automated message.')
		      states['total'] = current

		      new_msg = ''
		      for i in msg:
		        new_msg += i
		        new_msg += '\n'

		      # print(new_msg)
		      send_email(new_msg)
		    else:
		      print('flag not asserted')
		  except Exception as e:
		    print(e)

		  # print(states)
		  

		print("Finished")
		return("Your Email has been successfully registered for updates :)")


if __name__ == "__main__":
	app.jinja_env.auto_reload = True
	app.config['TEMPLATES_AUTO_RELOAD'] = True
	app.run(debug = True)