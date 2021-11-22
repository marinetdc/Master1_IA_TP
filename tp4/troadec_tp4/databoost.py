import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from statistics import mean, stdev 
from time import time

##############################################################################

# DATASET A
XHA = np.array([[1,1],[2,1],[1,2],[2,2], [3,3], [3,4], [4,3], [4,4],
               [1,3], [1,4], [2,3], [2,4], [3,1], [3,2], [4,1], [4,2]])
YHA = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])

# DATASET B
XHB = np.array([[1,1],[2,1],[1,2],[2,2], [3,3], [3,4], [4,3], [4,4],
               [1,3], [1,4], [2,3], [2,4], [3,1], [3,2], [4,1], [4,2],[5,1]])
YHB = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1])

# DATAWET C
XHC = np.array([[1,1],[2,1],[1,2],[2,2], [3,3], [3,4], [4,3], [4,4],
               [1,3], [1,4], [2,3], [2,4], [3,1], [3,2], [4,1], [4,2],[5,1]])
YHC = np.array([0,0,1,0,0,0,0,0,1,1,1,1,1,1,0,1,1])

##############################################################################

def couleur(y) : 
    c = []
    color = ['red', 'blue']
    for i in y :
        c.append(color[i])
    return c


def performances(X , Y, i ):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3) 
    scores = []
    clf = AdaBoostClassifier(n_estimators=i)
    model = clf.fit(X_train,y_train)
    scores.append(model.score(X_test, y_test))
    y_pred = model.predict(X_test)
    x = [X_test[i][0] for i in range(len(y_pred))]
    y = [X_test[i][1] for i in range(len(y_pred))]
    c = couleur(y_pred)
    plt.scatter(x,y, c = c)  
    plt.title("Classification pour {0} itérations".format(i))

for i in  [20, 50, 70, 90, 150, 300] :       
    performances(X=XHA,Y=YHA,i=i)

performances(X=XHA,Y=YHA,i=20)
performances(X=XHA,Y=YHA,i=50)
performances(X=XHA,Y=YHA,i=70)
performances(X=XHA,Y=YHA,i=90)
performances(X=XHA,Y=YHA,i=150)
performances(X=XHA,Y=YHA,i=300)