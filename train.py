import numpy as np
from TFANN import ANNR
import matplotlib.pyplot as mpl
from sklearn.preprocessing import scale
import argparse

import pickle

future_n = 7
pth =  'AAPL_one_year.csv'
A = np.loadtxt(pth, delimiter=",", skiprows=1, usecols=(1, 5))
print(A.shape)
Amax=int(np.max(A))

A = scale(A)
#y is the dependent variable
y = A[:, 1].reshape(-1, 1)
#A contains the independent variable
A = A[:, 0].reshape(-1, 1)
#A = scale(A)
#Plot the high value of the stock price
mpl.plot(A[:, 0], y[:, 0])
mpl.show()

#Number of neurons in the input layer
i =1
#Number of neurons in the output layer
o =1
#Number of neurons in the hidden layers
h = len(A)/2

nDays = 7
n = len(A)

#3 Fully-connected layers with tanh followed by linear output layer
layers=[]
k=1
while int(h/k) > 1:
    layers.append(('F',int(h/k)))
    layers.append(('AF','relu6'))
    k*=2
layers.append(('F',o))
#"""
layers = [('F', int(h)), ('AF', 'tanh'), ('F', int(h/2)), ('AF', 'tanh'), ('F', int(h/4)), ('AF', 'tanh'),
    ('F', int(h/8)), ('AF', 'tanh'), ('F', int(h/16)), ('AF', 'tanh'), ('F', int(h/16)), ('AF', 'tanh'), ('F', int(h/32)),
    ('AF', 'tanh'),  ('F', int(h/64)), ('AF', 'tanh'), ('F', o)]
#    """
mlpr = ANNR([i], layers, batchSize = 256, maxIter = 100000, tol = 0.05, reg = 1e-4, verbose = True, name='Stocker')

#Learn the data
mlpr.fit(A[0:(n-nDays)], y[0:(n-nDays)])

#save the model
mlpr.SaveModel('model/'+mlpr.name)
#Begin prediction
yHat = mlpr.predict(A)
#Plot the results
mpl.plot(A[-20:], y[-20:], c='#b0403f')
mpl.plot(A[-20:], yHat[-20:], c='#5aa9ab')
mpl.show()

mpl.plot(A, y, c='#b0403f')
mpl.plot(A, yHat, c='#5aa9ab')
mpl.show()
