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

X_train, Y_train, X_test, Y_test = sklearn.model_selection.train_test_split((X,Y, test_size = 0.1))