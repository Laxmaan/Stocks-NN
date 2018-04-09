from TFANN import ANNR
import numpy as np
from TFANN import ANNR
import matplotlib.pyplot as mpl
from sklearn.preprocessing import scale,StandardScaler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("days", help="number of days into the future")
args = parser.parse_args()


future_n = int(args.days)
pth =  'AAPL_one_year.csv'
A = np.loadtxt(pth, delimiter=",", skiprows=1, usecols=(1, 5))
Amax=int(np.max(A))
y = A[:, 1].reshape(-1, 1)
B=[]
for x in range(1,future_n+1):
    B.append(float(Amax+x))

B=np.array(B)
A = A[:, 0]
A=np.concatenate((A,B))
scaler = StandardScaler()

A=A.reshape((-1, 1))
A=scaler.fit_transform(A)
y=scaler.fit_transform(y)


#Number of neurons in the input layer
i =1
#Number of neurons in the output layer
o =1
#Number of neurons in the hidden layers
h = (len(A)-len(B))/2

nDays = 7
n = len(A)-len(B)

#3 Fully-connected layers with tanh followed by linear output layer
layers=[]
k=1
while int(h/k) > 1:
    layers.append(('F',h/k))
    layers.append(('AF','relu6'))
    k*=2
layers.append(('F',o))
#"""
layers = [('F', int(h)), ('AF', 'tanh'), ('F', int(h/2)), ('AF', 'tanh'), ('F', int(h/4)), ('AF', 'tanh'),
    ('F', int(h/8)), ('AF', 'tanh'), ('F', int(h/16)), ('AF', 'tanh'), ('F', int(h/16)), ('AF', 'tanh'), ('F', int(h/32)),
    ('AF', 'tanh'),  ('F', int(h/64)), ('AF', 'tanh'), ('F', o)]
#    """
mlpr = ANNR([i], layers, batchSize = 256, maxIter = 100000, tol = 0.05, reg = 1e-4, verbose = True, name='Stocker')

mlpr.RestoreModel('model/',mlpr.name)
#Begin prediction
yHat = mlpr.predict(A)
y=scaler.inverse_transform(y)
A=scaler.inverse_transform(A)
yHat=scaler.inverse_transform(yHat)
#Plot the results
mpl.plot(A[-20:-future_n], y[-(20-future_n):], c='#b0403f',label='Stock value')
mpl.plot(A[-20:], yHat[-20:], c='#5aa9ab',label='Stock Estimate')
mpl.xlabel('Days (20 days) scaled')
mpl.ylabel('Value')
mpl.title('Apple Stocks vs Estimates')
mpl.legend()
mpl.show()

mpl.plot(A[:-future_n], y, c='#b0403f',label='Stock value')
mpl.plot(A, yHat, c='#5aa9ab',label='Stock Estimate')
mpl.xlabel('Days (12/12/1980-02/2/2018) scaled')
mpl.ylabel('Value ')
mpl.title('Apple Stocks vs Estimates')
mpl.legend()
mpl.show()

print("Estimated value is{}".format(yHat[-1]))
