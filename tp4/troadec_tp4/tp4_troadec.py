
from databoost import *
from boostutil import *
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from statistics import mean, stdev 
from time import time
import random
import math
from sklearn.ensemble import GradientBoostingClassifier

##############################################################################

# Question 1.1.1 

# def generateZones(clf,limitsx, limitsy, T, process_all=1):
    # ...
    # r = Rectangle(TX[0][xt], TX[0][xt+1], TX[1][yt], TX[1][yt+1])
    # La lige ci-dessus appelle la classe Rectangle pour le rectangle 
    # (xt,xt+1,yt,yt+1) associé à une classe 
    
    ## class Rectangle:
        ##def set_class(self, c):
           ##self.class_ = c
         # Cette fonction défnie la classe à c 
        ##def __str__(self):
            ##return str(self.center_)+' -- '+str(self.class_)
        # Pour le centre de ce rectangle la classe c est associée 
   
##############################################################################

# Question 1.1.2

# def generateZones(clf,limitsx, limitsy, T, process_all=1):
#     # A list of two lists (thresholds on first then second component)
#     TX =[]                    #Création d'une liste
#     TX.append([limitsx[0]])   #TX[0] prend la valeur minimale de x 
#     TX.append([limitsy[0]])   #TX[1] prend la valeur minimale de y 
#     # getting the weak separators given by stumps
#     if process_all:
#         for ite in range(T): #Pour chaque itération
#             stump = getStump(clf,ite)  #On défni le stump, le coin d'un rectangle 
#             TX[stump[0]].append(stump[1])  #Le coin stump[1] est rentré dans la liste TX pour la classe stump[0]
#     else: # pareil que if mais on le fait qu'une seule fois 
#         stump = getStump(clf,T)
#         TX[stump[0]].append(stump[1])
#     TX[0].append(limitsx[1]) # TX[0] prend la valeur maximale de x 
#     TX[1].append(limitsy[1]) # TX[1] prend la valeur maximale de y 
#     # sorting
#     for i in [0,1]:  # les données des sous listes de TX sont triés par ordre croissant 
#         TX[i] = np.array(TX[i]) 
#         TX[i].sort()
#     # list of rectangles to be colored
#     R = []                # Liste qui va contenir les coordonnées des rectangles 
#     for yt in range(TX[1].shape[0]-1): # Pour chaque coordonnées y des rectangles possibles
#         for xt in range(TX[0].shape[0]-1): # pour chaque coordonnées x des rectangles possibles
#             r = Rectangle(TX[0][xt], TX[0][xt+1], TX[1][yt], TX[1][yt+1])  #Création d'un rectangle 
#             R.append(r)
#     return R                      # retourne la liste de tous les rectangles 

##############################################################################

# Question 1.1.3

##############################################################################

# Question 1.2.1

def couleur(y) : 
    c = []
    color = ['red', 'blue']
    for i in y :
        c.append(color[i])
    return c

X,Y = make_classification(n_samples=100, n_features=2, n_informative=2,
                          n_redundant=0, n_repeated=0, class_sep=0.5,random_state=72)

x = [X[i][0] for i in range(100)]
y = [X[i][1] for i in range(100)]

c = couleur(Y)

plt.scatter(x,y, c = c)            
               
# Question 1.2.3 

# class_sep défini la séparation des données 

# Question 1.2.4 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3) 

clf = AdaBoostClassifier()
model = clf.fit(X_train,y_train)
print("Accuracy:", model.score(X_test, y_test))

# Question 1.2.5 

def performances(nb_iter = [20, 50, 70, 90, 150, 300], X = X, Y = Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3) 
    scores = []
    for i in nb_iter : 
        start = time()
        clf = AdaBoostClassifier(n_estimators=i)
        model = clf.fit(X_train,y_train)
        end = time()
        scores.append(model.score(X_test, y_test))
        print("Training time took for nb_iter = {0}".format(i)+" is ",end - start)
        print("and the score is ", scores[-1])
    return [mean(scores), stdev(scores)]
        
perf =  performances()           
print("La moyenne des scores est {0} et l'écart-type est {1}".format(perf[0], perf[1]))            
       
performances([i for i in range(2,500, 25)])      
               

##############################################################################
               
# Partie 1.3

X,Y = make_classification(n_samples=100, n_features=2, n_informative=2,
                          n_redundant=0, n_repeated=0, class_sep=0.5,random_state=72)
             
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3) 

def error(attendu, pred) : 
    err = 0 
    for i,j in zip(attendu,pred) : 
        if i != j : 
            err += 1 
    err /= len(pred)
    return err

err = []
for i in range(10,55,5) : 
    y = list(y_train[:i])
    random.shuffle(y)
    y = np.array(y)
    y_tr = y_train[i:]
    y_train_2 = np.concatenate((y, y_tr))
    clf = AdaBoostClassifier()
    model = clf.fit(X_train,y_train_2)
    err.append(error(y_test, model.predict(X_test)))
         
                
plt.plot([i for i in range(10,55,5)], err)   
plt.title("Erreur en fonctoin du pourcentage de bruit")         
                
             
# Question 4              

X,Y = make_classification(n_samples=100, n_features=2, n_informative=2,
                          n_redundant=0, n_repeated=0, class_sep=0.5,random_state=72)

num_zeros = (Y ==0).sum()
num_un = (Y == 1 ).sum()

lx0 = []
ly0 = np.array([0 for i in range(num_zeros)])
lx1 = []
ly1 = np.array([1 for i in range(num_un)])

for i in range(len(Y)) : 
    if Y[i] == 0 :
        lx0.append(X[i])
    else : 
        lx1.append(X[i])

new_x = np.concatenate((np.array(lx0),np.array(lx1)))
new_y = np.concatenate((ly0,ly1))

clf = AdaBoostClassifier()
model = clf.fit(new_x,new_y)

from sklearn.utils import shuffle
new_x_test, new_y_test = shuffle(new_x,new_y)

print(error(new_y_test, model.predict(new_x_test)))

##############################################################################
                
# Partie 2            
                
def f(xmin = -5, xmax = 5, pas = 0.1) : 
    vy = []
    vx = []
    i = xmin
    while i < xmax : 
        vx.append(i)
        vy.append(i*i*math.sin(2*i-1))
        i += pas 
        
    return np.array(vx), np.array(vy)

x,y = f()

plt.plot(x,y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3) 

clf = GradientBoostingClassifier(n_estimators = 5)
model = clf.fit(X_train,y_train)                
             

plt.plot(X_test,y_test, c = 'pink')
plt.plot(X_test, model.predict(X_test), c= 'green')               
             
                
             
                
             
                
             
                
             
               
               
               
