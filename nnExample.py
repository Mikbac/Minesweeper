from neuralnet import NeuralNet
import matplotlib.pyplot as plt
import os
import struct
import numpy as np
import sys

X_train = np.genfromtxt('data/img_lear.csv', dtype=int, delimiter=',')
y_train = np.genfromtxt('data/le_lear.csv', dtype=int, delimiter=',')
X_test = np.genfromtxt('data/img_test.csv', dtype=int, delimiter=',')
y_test = np.genfromtxt('data/le_test.csv', dtype=int, delimiter=',')

print('\nRzedy: %d, kolumny: %d' % (X_train.shape[0], X_train.shape[1]))

# n_output = 2 - warstwy wyjsciowe
# n_features=X_train.shape[1] - ilosc danych trenowanych na wejsciu - wejsciowe
# n_hidden=50 - warstwy ukryte
# l2=0.1 - parametr regularyzacji, pozwalajacy na zmniejszenie stopnia przetrenowania
# l1=0.0 - 
# epochs=1000 - liczba przebiegow algorytmu po zestawie danych uczacych 
# eta=0.001 - wspołczynnik uczenia
# alpha=0.001 - parametr dla uczenia momentowego
# decrease_const=0.00001 - stała redukcji d stanowiaca element adaptacyjnego wspołczynnika uczenia n
# shuffle=True - przed kazda epoka testujemy zestaw danycz uczacych
# minibatches=50 - rozdzielenie danycz uczacych na k podzbiorow w kazdej epoce
# random_state=1 - wartosc losowa

#--------learning--------

nn = NeuralNet(n_output = 2, n_features=X_train.shape[1], n_hidden=50, l2=0.1, l1=0.0, epochs=500, eta=0.001, alpha=0.001, decrease_const=0.00001, shuffle=True, minibatches=50, random_state=1)
nn.fit(X_train, y_train, print_progress=True)

#--------end-learning--------

print('\n')

#--------examples-type-1--------
##-1-
X_data1 = np.genfromtxt('data/1.csv', dtype=int, delimiter=',')
y_test_pred1 = nn.predict(X_data1)
miscl_lab1 = y_test_pred1
if(miscl_lab1[1] == 1):
	print('1 -> bomb')
else:
	print('1 -> nothing')
	
# nothing	
	
	
##-2-	
X_data1 = np.genfromtxt('data/2.csv', dtype=int, delimiter=',')
y_test_pred1 = nn.predict(X_data1)
miscl_lab1 = y_test_pred1
if(miscl_lab1[1] == 1):
	print('2 -> bomb')
else:
	print('2 -> nothing')
		
# nothing


##-3-	
X_data2 = np.genfromtxt('data/20.csv', dtype=int, delimiter=',')
y_test_pred2 = nn.predict(X_data2)
miscl_lab2 = y_test_pred2
if(miscl_lab2[1] == 1):
	print('20 -> bomb')
else:
	print('20 -> nothing')
	
# bomb	


##-4-
X_data2 = np.genfromtxt('data/40.csv', dtype=int, delimiter=',')
y_test_pred2 = nn.predict(X_data2)
miscl_lab2 = y_test_pred2
if(miscl_lab2[1] == 1):
	print('40 -> bomb')
else:
	print('40 -> nothing')
	
# bomb	


##-5-
X_data2 = np.genfromtxt('data/60.csv', dtype=int, delimiter=',')
y_test_pred2 = nn.predict(X_data2)
miscl_lab2 = y_test_pred2
if(miscl_lab2[1] == 1):
	print('60 -> bomb')
else:
	print('60 -> nothing')
	
# bomb	


##-6-
X_data2 = np.genfromtxt('data/155.csv', dtype=int, delimiter=',')
y_test_pred2 = nn.predict(X_data2)
miscl_lab2 = y_test_pred2
if(miscl_lab2[1] == 1):
	print('155 -> bomb')
else:
	print('155 -> nothing')
	
# nothing	


##-7-
X_data2 = np.genfromtxt('data/225.csv', dtype=int, delimiter=',')
y_test_pred2 = nn.predict(X_data2)
miscl_lab2 = y_test_pred2
if(miscl_lab2[1] == 1):
	print('225 -> bomb')
else:
	print('225 -> nothing')

# nothing

#--------end-examples-type-1--------


#--------examples-type-2--------
##-1-
print(nn.detection(nn, 33))
# return 0

##-2-
print(nn.detection(nn, 80))
# return 1

##-3-
print(nn.detection(nn, 122))
# return 0

##-4-
print(nn.detection(nn, 200))
# return 1

##-5-
print(nn.detection(nn, 199))
# return 0

#--------end-examples-type-2--------


#--------data_visualisation--------
# 1 - bomb
# 0 - nothing
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
	img = X_train[y_train == 1][i].reshape(28, 28)
	ax[i].imshow(img, cmap = 'Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
#--------end-data_visualisation--------
