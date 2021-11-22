#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:30:26 2021

@author: marinetroadec
"""

import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import pprint 
from sklearn import tree

###############################################################################

def distance_au_centre(X) : 
    
    d = X.shape[1]
    centre = 0.5*np.ones(d)
    moy = 0
    for i in X : 
        moy += calcul_distance(centre,i)
    moy /= d 
    return moy
    
    
def calcul_distance(x,y) : 
    
    maxim = 0
    for i in range(len(x)) :
        if maxim < abs(x[i]-y[i]) : 
            maxim = abs(x[i]-y[i])
    return maxim 

def voisin_le_plus_proche_du_centre(X) : 
    
    d = X.shape[1]
    centre = 0.5*np.ones(d)
    minimum = calcul_distance(centre,X[0])
    for i in X :
        if minimum > calcul_distance(centre,i) :
            minimum = calcul_distance(centre,i)
    return minimum

###############################################################################

def damier(dimension, grid_size, nb_examples, noise = 0):
    data = np.random.rand(nb_examples,dimension)
    labels = np.ones(nb_examples)
    for i in range(nb_examples):
        x = data[i,:];
        for j in range(dimension):
            if int(np.floor(x[j]*grid_size)) % 2 != 0:
                labels[i]=labels[i]*(-1)
        if np.random.rand()<noise:
            labels[i]=labels[i]*(-1)
    return data, labels





# Ex 1 
dict_tab = {
            "dim":[2,3,4,5,6,7,8,9,10],
            "size":[2,3,4,5,6,7,8],
            "nb_ex":[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000],
            "noise":[0, 0.05, 0.1, 0.15, 0.2],
            "n_neig":[1,2,3,4,5],
            "score":0,
        }

tab_dict = []

for i in dict_tab["dim"]:
    
    nb_data_test = dict_tab["nb_ex"][0] * 0.3
    nb_data_train = dict_tab["nb_ex"][0] * 0.7

    data, labels = damier(i, dict_tab["size"][0], dict_tab["nb_ex"][0], dict_tab["noise"][0])
    
    x_train = data[:int(nb_data_train)]
    y_train = labels[:int(nb_data_train)]
    
    x_test = data[int(nb_data_test):]
    y_test = labels[int(nb_data_test):]
    
    knn=KNeighborsClassifier(n_neighbors=dict_tab["n_neig"][0])
    knn.fit(x_train, y_train) 
    pred_test=knn.predict(x_test) 
        
    #Evaluer le modèle en utilisant le score :
    score = knn.score(x_test, y_test)
    
    tab_dict.append({"dim" :i, 
                     "size" : dict_tab["size"][0],
                     "nb_ex" : dict_tab["nb_ex"][0],
                     "noise" : dict_tab["noise"][0],
                     "n_neig" : dict_tab["n_neig"][0],
                     "score" : score,
                     })
    
for i in dict_tab["size"]:
    
    nb_data_test = dict_tab["nb_ex"][0] * 0.3
    nb_data_train = dict_tab["nb_ex"][0] * 0.7

    data, labels = damier(dict_tab["dim"][0], i, dict_tab["nb_ex"][0], dict_tab["noise"][0])
    
    x_train = data[:int(nb_data_train)]
    y_train = labels[:int(nb_data_train)]
    
    x_test = data[int(nb_data_test):]
    y_test = labels[int(nb_data_test):]
    
    knn=KNeighborsClassifier(n_neighbors=dict_tab["n_neig"][0])
    knn.fit(x_train, y_train) 
    pred_test=knn.predict(x_test) 
        
    #Evaluer le modèle en utilisant le score :
    score = knn.score(x_test, y_test)
    
    tab_dict.append({"dim" :dict_tab["dim"][0], 
                     "size" : i,
                     "nb_ex" : dict_tab["nb_ex"][0],
                     "noise" : dict_tab["noise"][0],
                     "n_neig" : dict_tab["n_neig"][0],
                     "score" : score,
                     })

for i in dict_tab["nb_ex"]:
    
    nb_data_test = i * 0.3
    nb_data_train = i * 0.7

    data, labels = damier(dict_tab["dim"][0], dict_tab["size"][0], i, dict_tab["noise"][0])
    
    x_train = data[:int(nb_data_train)]
    y_train = labels[:int(nb_data_train)]
    
    x_test = data[int(nb_data_test):]
    y_test = labels[int(nb_data_test):]
    
    knn=KNeighborsClassifier(n_neighbors=dict_tab["n_neig"][0])
    knn.fit(x_train, y_train) 
    pred_test=knn.predict(x_test) 
        
    #Evaluer le modèle en utilisant le score :
    score = knn.score(x_test, y_test)
    
    tab_dict.append({"dim" :dict_tab["dim"][0], 
                     "size" : dict_tab["size"][0],
                     "nb_ex" : i,
                     "noise" : dict_tab["noise"][0],
                     "n_neig" : dict_tab["n_neig"][0],
                     "score" : score,
                     })

for i in dict_tab["noise"]:
    
    nb_data_test = dict_tab["nb_ex"][0] * 0.3
    nb_data_train = dict_tab["nb_ex"][0] * 0.7

    data, labels = damier(dict_tab["dim"][0], dict_tab["size"][0], dict_tab["nb_ex"][0], i)
    
    x_train = data[:int(nb_data_train)]
    y_train = labels[:int(nb_data_train)]
    
    x_test = data[int(nb_data_test):]
    y_test = labels[int(nb_data_test):]
    
    knn=KNeighborsClassifier(n_neighbors=dict_tab["n_neig"][0])
    knn.fit(x_train, y_train) 
    pred_test=knn.predict(x_test) 
        
    #Evaluer le modèle en utilisant le score :
    score = knn.score(x_test, y_test)
    
    tab_dict.append({"dim" :dict_tab["dim"][0], 
                     "size" : dict_tab["size"][0],
                     "nb_ex" : dict_tab["nb_ex"][0],
                     "noise" : i,
                     "n_neig" : dict_tab["n_neig"][0],
                     "score" : score,
                     })
    
for i in dict_tab["n_neig"]:
    
    nb_data_test = dict_tab["nb_ex"][0] * 0.3
    nb_data_train = dict_tab["nb_ex"][0] * 0.7

    data, labels = damier(dict_tab["dim"][0], dict_tab["size"][0], dict_tab["nb_ex"][0], dict_tab["noise"][0])
    
    x_train = data[:int(nb_data_train)]
    y_train = labels[:int(nb_data_train)]
    
    x_test = data[int(nb_data_test):]
    y_test = labels[int(nb_data_test):]
    
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train) 
    pred_test=knn.predict(x_test) 
        
    #Evaluer le modèle en utilisant le score :
    score = knn.score(x_test, y_test)
    
    tab_dict.append({"dim" :dict_tab["dim"][0], 
                     "size" : dict_tab["size"][0],
                     "nb_ex" : dict_tab["nb_ex"][0],
                     "noise" : dict_tab["noise"][0],
                     "n_neig" : i,
                     "score" : score,
                     })

    
pprint.pprint(tab_dict)   

# EXERCICE 2 

dict_tab_2 = { 
            "dim":2,
            "size":[2,3,4,5,6,7,8,9,10],
            #"size":[2],
            "nb_ex":1000,
            "noise":0,
            "n_neig":[1,2,3,4,5],
            "score":0,
        }


tab_dict_res = []

for i in dict_tab_2["size"]:
    
    for j in dict_tab_2["n_neig"] :
        
        moy = 0 
        
        for i in range(10) : 
    
            nb_data_test = dict_tab_2["nb_ex"] * 0.3
            nb_data_train = dict_tab_2["nb_ex"] * 0.7
        
            data, labels = damier(dict_tab_2["dim"], i, dict_tab_2["nb_ex"], dict_tab_2["noise"])
            
            x_train = data[:int(nb_data_train)]
            y_train = labels[:int(nb_data_train)]
            
            x_test = data[int(nb_data_test):]
            y_test = labels[int(nb_data_test):]
            
            knn=KNeighborsClassifier(n_neighbors= j )
            knn.fit(x_train, y_train) 
            pred_test=knn.predict(x_test) 
                
            #Evaluer le modèle en utilisant le score :
            score = knn.score(x_test, y_test)
            
            moy += score 
        
        tab_dict_res.append({"dim" :2, 
                        "size" : i,
                        "nb_ex" : 1000,
                        "noise" : 0,
                        "n_neig" : j,
                        "score" : moy/10,
                        })
        

X = [i["score"] for i in tab_dict_res]
Y = [i["n_neig"] for i in tab_dict_res]

cmap_bold = ['darkorange', 'c', 'darkblue']

sns.scatterplot(x=X, y=Y, 
                    palette=cmap_bold, alpha=1.0, edgecolor="black")
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.title("3-Class classification (k = %i, weights = '%s')"
#               % (n_neighbors, weights))
plt.xlabel("Score")
plt.ylabel("Neig")






#### Modif noise 




dict_tab_2_noise = { 
            "dim":2,
            "size":[2,3,4,5,6,7,8,9,10],
            "nb_ex":1000,
            "noise":0.2,
            "n_neig":[1,2,3,4,5],
            "score":0,
        }
        
dic_res_noise = []

for i in dict_tab_2_noise["size"]:
    
    for j in dict_tab_2_noise["n_neig"] : 
        
        moy = 0
        
        for i in range(10) : 
    
            nb_data_test = dict_tab_2_noise["nb_ex"] * 0.3
            nb_data_train = dict_tab_2_noise["nb_ex"] * 0.7
        
            data, labels = damier(dict_tab_2_noise["dim"], i, dict_tab_2_noise["nb_ex"], dict_tab_2_noise["noise"])
            
            x_train = data[:int(nb_data_train)]
            y_train = labels[:int(nb_data_train)]
            
            x_test = data[int(nb_data_test):]
            y_test = labels[int(nb_data_test):]
            
            knn=KNeighborsClassifier(n_neighbors= j )
            knn.fit(x_train, y_train) 
            pred_test=knn.predict(x_test) 
                
            #Evaluer le modèle en utilisant le score :
            score = knn.score(x_test, y_test)
            
            moy += score
        
        dic_res_noise.append({"dim" :2, 
                        "size" : i,
                        "nb_ex" : 1000,
                        "noise" : 0.2,
                        "n_neig" : j,
                        "score" : moy/10,
                        })

#print("dic_res_noise")
#pprint.pprint(dic_res_noise)

X = [i["score"] for i in dic_res_noise]
Y = [i["n_neig"] for i in dic_res_noise]


cmap_bold = ['darkorange', 'c', 'darkblue']

sns.scatterplot(x=X, y=Y, 
                    palette=cmap_bold, alpha=1.0, edgecolor="black")
plt.xlabel("Score")
plt.ylabel("Neig")

plt.show()


###############################################################################
######## Arbre de décision 


dic_res_tree = []


for i in dict_tab_2["size"]:
    
    for j in dict_tab_2["n_neig"] :
        
        moy = 0 
        
        for i in range(10) : 
    
            nb_data_test = dict_tab_2["nb_ex"] * 0.3
            nb_data_train = dict_tab_2["nb_ex"] * 0.7
        
            data, labels = damier(dict_tab_2["dim"], i, dict_tab_2["nb_ex"], dict_tab_2["noise"])
            
            x_train = data[:int(nb_data_train)]
            y_train = labels[:int(nb_data_train)]
            
            x_test = data[int(nb_data_test):]
            y_test = labels[int(nb_data_test):]
            
            clf = tree.DecisionTreeClassifier(max_depth=j)
            clf = clf.fit(x_train, y_train)
            
            pred_test=clf.predict(x_test) 
                
            #Evaluer le modèle en utilisant le score :
            score = clf.score(x_test, y_test)
            
            moy += score 
        
        dic_res_tree.append({"dim" :2, 
                        "size" : i,
                        "nb_ex" : 1000,
                        "noise" : 0,
                        "n_neig" : j,
                        "score" : moy/10,
                        })

tree.plot_tree(clf) 

X = [i["score"] for i in dic_res_tree]
Y = [i["n_neig"] for i in dic_res_tree]


cmap_bold = ['darkorange', 'c', 'darkblue']

sns.scatterplot(x=X, y=Y, 
                    palette=cmap_bold, alpha=1.0, edgecolor="black")
plt.xlabel("Score")
plt.ylabel("Neig")

plt.show()

#pprint.pprint(dic_res_tree)



dic_res_noise_tree = []


for i in dict_tab_2_noise["size"]:
    
    for j in dict_tab_2_noise["n_neig"] :
        
        moy = 0 
        
        for i in range(10) : 
    
            nb_data_test = dict_tab_2_noise["nb_ex"] * 0.3
            nb_data_train = dict_tab_2_noise["nb_ex"] * 0.7
        
            data, labels = damier(dict_tab_2_noise["dim"], i, dict_tab_2_noise["nb_ex"], dict_tab_2_noise["noise"])
            
            x_train = data[:int(nb_data_train)]
            y_train = labels[:int(nb_data_train)]
            
            x_test = data[int(nb_data_test):]
            y_test = labels[int(nb_data_test):]
            
            clf = tree.DecisionTreeClassifier(max_depth = j)
            clf = clf.fit(x_train, y_train)
            
            pred_test=clf.predict(x_test) 
                
            #Evaluer le modèle en utilisant le score :
            score = clf.score(x_test, y_test)
            
            moy += score 
        
        dic_res_noise_tree.append({"dim" :2, 
                        "size" : i,
                        "nb_ex" : 1000,
                        "noise" : 0.2,
                        "n_neig" : j,
                        "score" : moy/10,
                        })

#tree.plot_tree(clf) 

X = [i["score"] for i in dic_res_noise_tree]
Y = [i["n_neig"] for i in dic_res_noise_tree]


cmap_bold = ['darkorange', 'c', 'darkblue']

sns.scatterplot(x=X, y=Y, 
                    palette=cmap_bold, alpha=1.0, edgecolor="black")

plt.xlabel("Score")
plt.ylabel("Neig")
#pprint.pprint(dic_res_tree)


# if __name__ == "__main__" : 
    
#     de = np.random.rand(10,6)
    
#     print(distance_au_centre(de))
    
#     print(voisin_le_plus_proche_du_centre(de))
    
#     ##############
    
#     for d in range(1,21):
#         dist = []
#         v = []
#         for i in range(10): # creation de 10 tableau 
#             X = np.random.rand(100,d)  # de dimansion 100 x d (d de 1 à 20)
#             dist.append(distance_au_centre(X))  # liste contenant les distances au centre de chaque tb
#             v.append( voisin_le_plus_proche_du_centre(X))  # liste contenant les points les plus proche de chaque tb
#     print(np.mean(dist), np.mean(v)) # retourne la moy sur 10 tb des distance au centre et min par rapport à la dimension du dé 
    
    
    
