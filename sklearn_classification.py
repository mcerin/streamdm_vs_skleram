import numpy as np
import time

from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron

from sklearn.metrics import recall_score, precision_score, f1_score

data = np.genfromtxt('data.txt', delimiter = ',')

X = data[:, 0:13]
y = data[:, 13]


regressors = {
    "log": LogisticRegression(solver = 'liblinear'),
    "tree": DecisionTreeClassifier(),
    "mlpc": MLPClassifier(),
    "forest": RandomForestClassifier(n_estimators = 10),
    "knc": KNeighborsClassifier(),
    "percep": Perceptron()
}

regressor = regressors["tree"]

start = time.time() 
regressor.fit(X,y)
end = time.time()

print('time: ', end - start)

#predict = regressor.predict(X[m:])
#print('F1=', f1_score(y[m:], predict), 'Recall= ', recall_score(y[m:], predict),'Precision= ',precision_score(y[m:], predict))