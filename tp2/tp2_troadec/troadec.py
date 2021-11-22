
############################### TP2 ######################################

import numpy as np 
from pylab import rand
import matplotlib.pyplot as plt
from perceptron_data import iris
from perceptron_data import bias
from collections import Counter
from sklearn.metrics import confusion_matrix
from time import time

## Algorithme du perceptron pour la classification binaire : 

def classifieur(s,n) : 
    """ Creer le vecteur W """   
    d = len(s[0][0])    
    w = np.zeros(d)
    y_pred = 0
    
    for j in range(n) :    
        for i in s :             
            y_pred = np.vdot(i[0],w)            
            if y_pred >= 0 and (i[1] <= 0 or i[1] == False) :                
                w = w - i[0]    
            elif y_pred <= 0 and (i[1] >= 0 or i[1] == True) :                 
                w = w + i[0]    
    return w 


def classifieur_biais(s,n,b) : 
    """ Creer le vecteur W """   
    d = len(s[0][0])    
    w = np.zeros(d)
    y_pred = 0
    
    for j in range(n) :   
        index = 0
        for i in s :  
            y_pred =  np.vdot(i[0],w) + b[index][0] if b[index][1] else (np.vdot(i[0],w)  - b[index][0] ) 
            if y_pred >= 0 and (i[1] <= 0 or i[1] == False) :                
                w = w - i[0]    
            elif y_pred <= 0 and (i[1] >= 0 or i[1] == True) :                 
                w = w + i[0]    
            index += 1 
    return w




def genererDonnees(n):
#g ́en ́erer un jeu de donn ́ees 2D lin ́eairement s ́eparable de taille n. 
    x1b = (rand(n)*2-1)/2-0.5
    x2b = (rand(n)*2-1)/2+0.5
    x1r = (rand(n)*2-1)/2+0.5
    x2r = (rand(n)*2-1)/2-0.5
    donnees = []
    for i in range(len(x1b)):
        donnees.append(((x1b[i],x2b[i]),False))
        donnees.append(((x1r[i],x2r[i]),True))
    return donnees

def err_prediction(w,s) : 
    
    y = np.array([])
    err = 0
    for i in s :
       
        if  np.vdot(i[0],w) > 0 : 
            y = np.append(y,True)
        else : 
            y = np.append(y, False)
        if (i[1] == True and y[-1] == 0) or (i[1] == False and y[-1] ==1) :
            err +=1
    err /= len(s)
    return err, y

def color(s) : 
    couleur = []
    col = ["r", "g", "b", "y"]
    for i in s : 
        couleur.append(col[int(i)])
    return couleur

S_train = genererDonnees(300)

S_train = np.asarray(S_train)

W = classifieur(S_train,5)

W_biais = classifieur_biais(S_train,5,bias)

S_test = np.asarray(genererDonnees(100))


error_train, pred_train = err_prediction(W,S_train)

error_train_biais, pred_train_biais = err_prediction(W_biais,S_train)

print('L\'erreur sur le jeu d\'entrainement est de : ', error_train)

error_test, pred_test = err_prediction(W,S_test)

error_test_biais, pred_test_biais = err_prediction(W_biais,S_test)

print('L\'erreur sur le jeu de test est de : ', error_test)

x = [i[0][0] for i in S_test]
y = [i[0][1] for i in S_test]
plt.scatter(x,y, c = color(pred_test))
plt.title("Sans biais")
plt.show()

x = [i[0][0] for i in S_test]
y = [i[0][1] for i in S_test]
plt.scatter(x,y, c = color(pred_test_biais))
plt.title("Avec biais")
plt.show()


## Perceptron multi-classes

def classifieur_multiclasse(s,n) : 
    y = [i[1] for i in s]
    l = len(np.unique(y))
    c = len(s[0][0])
    w = np.zeros((l,c))
    y_pred = 0
    
    for j in range(n) :
        for i in s : 
            y_pred = np.argmax(w.dot(i[0]))
            if y_pred != i[1] :
                w[i[1]] += i[0]
                w[y_pred] -= i[0]
    return w 

def predict_example(x_ts,w) :
    
    y_pred = []
    for i in x_ts : 
        y_pred.append(np.argmax(w.dot(i)))
    return y_pred 
    
def traitement_donnees(donne) : 
    
    classes = []
    nouv_donne = np.array([[[0,0,0,0],0]])
    
    classes = list(Counter([i[1] for i in donne]).keys())
    for i in donne : 
        add = np.array([[i[0], classes.index(i[1])]])
        nouv_donne = np.concatenate((nouv_donne,add),axis=0)
    nouv_donne = np.delete(nouv_donne,(0), axis = 0)
    return nouv_donne                                    

def error(s, pred) : 
    err = 0 
    j = 0
    for i in s : 
        if i[1] != pred[j] : 
            err += 1 
        j += 1
    err /= len(pred)
    return err

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
df_iris = traitement_donnees(iris)
iris_train = np.array([i for i in df_iris[:100]])
iris_test = np.array([i for i in df_iris[50:]])

start = time()
W_iris = classifieur_multiclasse(iris_train, 5)
end = time()
print("Training time took",end - start)

y_iris = predict_example([i[0] for i in iris_test], W_iris)

err_iris_test = error(iris_test, y_iris)

print("L'erreur sur le jeu de données de test est de ", err_iris_test)


x = [i[0][3] for i in iris_test]
y = [i[0][1] for i in iris_test]
plt.scatter(x,y, c = color(y_iris))
plt.show()

cm = confusion_matrix(y_iris, [i for i in y_iris])

plot_confusion_matrix(cm)

