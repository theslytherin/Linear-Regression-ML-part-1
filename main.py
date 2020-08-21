import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3" #lables

X=np.array(data.drop([predict], 1))
Y = np.array(data[predict])

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,Y, test_size=0.1)

linear= linear_model.LinearRegression()

linear.fit(X_train, Y_train)
acc =linear.score(X_test, Y_test)

print(acc)
print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(X_test)

for x in range(len(predictions)):
    print(round(predictions[x]), X_test[x], Y_test[x])
