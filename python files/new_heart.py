import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv("heart.csv")
from sklearn.externals import joblib 


dataset.head()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 101)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#SVM

from sklearn.svm import SVC
classifierSVM = SVC(kernel = 'rbf', random_state = 0)
classifierSVM.fit(X_train, y_train)
joblib.dump(classifierSVM, 'h_svm.pkl') 


#ANN

import warnings
warnings.filterwarnings("ignore")
np.seterr(divide = 'ignore') 

# import keras
# from keras.models import Sequential
# from keras.layers import Dense

# classifierANN = Sequential()


# classifierANN.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))
# classifierANN.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
# classifierANN.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
# classifierANN.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# classifierANN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# classifierANN.fit(X_train, y_train, batch_size = 20, epochs = 50)



#RANDOM FOREST 

from sklearn.ensemble import RandomForestClassifier
classifierRF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 101)
classifierRF.fit(X_train, y_train)
joblib.dump(classifierRF, 'h_rf.pkl') 




#DECISION TREES

from sklearn.tree import DecisionTreeClassifier
classifierDT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifierDT.fit(X_train, y_train)
joblib.dump(classifierDT, 'h_dt.pkl') 


#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
classifierLR = LogisticRegression(random_state = 0)
classifierLR.fit(X_train, y_train)


#NAIVE BAYES

from sklearn.naive_bayes import GaussianNB
classifierNB = GaussianNB()
classifierNB.fit(X_train, y_train)
joblib.dump(classifierNB, 'h_nb.pkl') 


#KNN

from sklearn.neighbors import KNeighborsClassifier
classifierKNN = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
classifierKNN.fit(X_train, y_train)
joblib.dump(classifierKNN, 'h_knn.pkl') 






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


test = np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalac,exang,oldpeak,slope,ca,thal])
test = test.reshape(1,-1)
print(type(test))
print(test)
test = sc.transform(test)
print(test)




y_predSVM = classifierSVM.predict(test)
y_predRF  = classifierRF.predict(test)
y_predDT  = classifierDT.predict(test) 
y_predLR  = classifierLR.predict(test) 
y_predNB  = classifierNB.predict(test) 
y_predKNN = classifierKNN.predict(test)
y_predANN = classifierANN.predict(test)
y_predANN = (y_predANN > 0.5)

 

results = [int(y_predLR),int(y_predSVM),int(y_predRF),int(y_predDT),int(y_predANN),int(y_predNB),int(y_predKNN)]

#print(result)
# Making the Confusion Matrix
'''from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("Accuracy = {}".format((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]) * 100))
'''

print(int(y_predLR),int(y_predSVM),int(y_predRF),int(y_predDT),int(y_predANN),int(y_predNB),int(y_predKNN))

confidence = results.count(1)*100/len(results)

print(confidence)
