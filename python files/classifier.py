import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.externals import joblib 

dataset = pd.read_csv("heart.csv")

dataset.head()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 101)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''
#SVM
classifierSVM = SVC(kernel = 'rbf', random_state = 0)
classifierSVM.fit(X_train, y_train)
joblib.dump(classifierSVM, 'SVM.pkl')

#ANN
warnings.filterwarnings("ignore")
np.seterr(divide = 'ignore') 

classifierANN = Sequential()

# Adding the input layer and the first hidden layer
classifierANN.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))

classifierANN.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))

classifierANN.add(Dense(units = 35, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifierANN.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifierANN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifierANN.fit(X_train, y_train,validation_data=(X_test,y_test), batch_size = 20, epochs = 150)

classifierANN.save('ANN.h5')

#RANDOM FOREST 
classifierRF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 101)
classifierRF.fit(X_train, y_train)
joblib.dump(classifierRF, 'RF.pkl')
'''
'''
#INPUT

print("start entering")
age = int(input())
sex = input()
if sex=='M':
    sex = int(1)
else:
    sex=int(0)

cp = int(input())
trestbps = int(input())
chol = int(input())
fbs = int(input())
restecg = int(input())
thalac = int(input())
exang = int(input())
oldpeak = float(input())
slope = int(input())
ca = int(input())
thal = int(input())


"""
ageage in years
sex(1 = male; 0 = female)
cpchest pain type
trestbpsresting blood pressure (in mm Hg on admission to the hospital)
cholserum cholestoral in mg/dl
fbs(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
restecgresting electrocardiographic results
thalachmaximum heart rate achieved
exangexercise induced angina (1 = yes; 0 = no)
oldpeakST depression induced by exercise relative to rest
slopethe slope of the peak exercise ST segment
canumber of major vessels (0-3) colored by flourosopy
thal3 = normal; 6 = fixed defect; 7 = reversable defect

"""
'''

# test = np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalac,exang,oldpeak,slope,ca,thal])
# test = test.reshape(1,-1)
# print(type(test))
# print(test)
# test = sc.transform(test)
# print(test)
test = X_test[14].reshape(1,-1)
test = sc.transform(test)

#loading the models

# Predicting the Test set results

print('results are:')
classifierSVM = joblib.load('SVM.pkl')
classifierRF = joblib.load('RF.pkl')
classifierANN = load_model('ANN.h5')
y_predSVM = classifierSVM.predict(test)
print(int(y_predSVM))


y_predANN = classifierANN.predict(test)
y_predANN = (y_predANN > 0.5)
print(int(y_predANN))


y_predRF = classifierRF.predict(test)
print(int(y_predRF))

results = [int(y_predANN),int(y_predSVM),int(y_predRF)]
if(results.count(1)>results.count(0)):
    result = 1
else:
    result = 0




#print(result)
# Making the Confusion Matrix
'''from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("Accuracy = {}".format((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]) * 100))
'''

