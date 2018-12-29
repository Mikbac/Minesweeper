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

print('\n Rzedy: %d, kolumny: %d' % (X_train.shape[0], X_train.shape[1]))

# n_output = 2 - wyjście
# n_features=X_train.shape[1] - 
# n_hidden=50 - 
# l2=0.1 - parametr regularyzacji, pozwalający na zmniejszenie stopnia przetrenowania
# l1=0.0 - 
# epochs=1000 - liczba przebiegów algorytmu po zestawie danych uczących 
# eta=0.001 - współczynnik uczenia
# alpha=0.001 - parametr dla uczenia momentowego, określający część wartości poprzedniego gradientu
# decrease_const=0.00001 - stała redukcji d stanowiąca element adaptacyjnego współczynnika uczenia n
# shuffle=True - przek każdą epoką testujemy zestaw danycz uczących
# minibatches=50 - rozdzielenie danycz uczących na k podzbiorów w każdej epoce
# random_state=1 -

nn = NeuralNet(n_output = 2, n_features=X_train.shape[1], n_hidden=50, l2=0.1, l1=0.0, epochs=1000, eta=0.001, alpha=0.001, decrease_const=0.00001, shuffle=True, minibatches=50, random_state=1)
nn.fit(X_train, y_train, print_progress=True)



X_data1 = np.genfromtxt('data/1.csv', dtype=int, delimiter=',')
y_test_pred1 = nn.predict(X_data1)
miscl_lab1 = y_test_pred1
if(miscl_lab1[1] == 1):
	print('bomb')
else:
	print('nothing')
	
# nothing	
	
	
	
X_data1 = np.genfromtxt('data/2.csv', dtype=int, delimiter=',')
y_test_pred1 = nn.predict(X_data1)
miscl_lab1 = y_test_pred1
if(miscl_lab1[1] == 1):
	print('bomb')
else:
	print('nothing')
		
# nothing


	
X_data2 = np.genfromtxt('data/20.csv', dtype=int, delimiter=',')
y_test_pred2 = nn.predict(X_data2)
miscl_lab2 = y_test_pred2
if(miscl_lab2[1] == 1):
	print('bomb')
else:
	print('nothing')
	
# bomb	



X_data2 = np.genfromtxt('data/40.csv', dtype=int, delimiter=',')
y_test_pred2 = nn.predict(X_data2)
miscl_lab2 = y_test_pred2
if(miscl_lab2[1] == 1):
	print('bomb')
else:
	print('nothing')
	
# bomb	



X_data2 = np.genfromtxt('data/60.csv', dtype=int, delimiter=',')
y_test_pred2 = nn.predict(X_data2)
miscl_lab2 = y_test_pred2
if(miscl_lab2[1] == 1):
	print('bomb')
else:
	print('nothing')
	
# bomb	



X_data2 = np.genfromtxt('data/225.csv', dtype=int, delimiter=',')
y_test_pred2 = nn.predict(X_data2)
miscl_lab2 = y_test_pred2
if(miscl_lab2[1] == 1):
	print('bomb')
else:
	print('nothing')
	
# bomb	


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
#----------------------------------