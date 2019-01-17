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
# eta=0.001 - wspolczynnik uczenia
# alpha=0.001 - parametr dla uczenia momentowego
# decrease_const=0.00001 - stala redukcji d stanowiaca element adaptacyjnego wspolczynnika uczenia n
# shuffle=True - przed kazda epoka testujemy zestaw danycz uczacych
# minibatches=50 - rozdzielenie danycz uczacych na k podzbiorow w kazdej epoce
# random_state=1 - wartosc losowa

#--------learning--------

nn = NeuralNet(n_output = 3, n_features=X_train.shape[1], n_hidden=50, l2=0.1, l1=0.0, epochs=500, eta=0.001, alpha=0.001, decrease_const=0.00001, shuffle=True, minibatches=50, random_state=1)
nn.fit(X_train, y_train, print_progress=True)

#--------end-learning--------



print('\n')



#--------examples-type-1--------
##-1-
print(nn.detection(nn, 20))
# return 1

##-2-
print(nn.detection(nn, 30))
# return 2

##-3-
print(nn.detection(nn, 40))
# return 1

##-4-
print(nn.detection(nn, 60))
# return 2

##-5-
print(nn.detection(nn, 80))
# return 1

##-6-
print(nn.detection(nn, 90))
# return 2

##-7-
print(nn.detection(nn, 100))
# return 1

##-8-
print(nn.detection(nn, 120))
# return 2

##-9-
print(nn.detection(nn, 140))
# return 1

##-10-
print(nn.detection(nn, 150))
# return 2

##-11-
print(nn.detection(nn, 160))
# return 1

##-12-
print(nn.detection(nn, 180))
# return 2

##-13-
print(nn.detection(nn, 200))
# return 1

##-14-
print(nn.detection(nn, 210))
# return 2

##-15-
print(nn.detection(nn, 220))
# return 1

##-16-
print(nn.detection(nn, 123))
# return 0

##-17-
print(nn.detection(nn, 44))
# return 0

##-18-
print(nn.detection(nn, 199))
# return 0

#--------end-examples-type-1--------


#--------data-visualisation-bomb--------
# 2 - dynamite
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


#--------data-visualisation-dynamite--------
# 2 - dynamite
# 1 - bomb
# 0 - nothing
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
	img = X_train[y_train == 2][i].reshape(28, 28)
	ax[i].imshow(img, cmap = 'Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
#--------end-data_visualisation--------


#--------data-plot--------
plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 500])
plt.ylabel('Koszt')
plt.xlabel('Epoki')
plt.tight_layout()
plt.show()
#--------end-data_visualisation--------