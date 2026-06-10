import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
def relu(x):
    return (x > 0) * x
def relu2deriv(out):
    return out > 0
data = np.array([[0, 0, 0, 1,
                  0, 0, 1, 1,
                  0, 0, 0, 1,
                  0, 0, 0, 1],
                 [0, 0, 0, 1,
                  0, 0, 0, 1,
                  0, 0, 0, 1,
                  0, 0, 0, 1],
                 [0, 0, 1, 0,
                  0, 1, 1, 0,
                  0, 0, 1, 0,
                  0, 0, 1, 0],
                 [0, 0, 1, 0,
                  0, 0, 1, 0,
                  0, 0, 1, 0,
                  0, 0, 1, 0],
                 [0, 1, 0, 0,
                  0, 1, 0, 0,
                  0, 1, 0, 0,
                  0, 1, 0, 0],
                 [0, 1, 0, 0,
                  1, 1, 0, 0,
                  0, 1, 0, 0,
                  0, 1, 0, 0],
                 [0, 0, 1, 0,
                  0, 1, 0, 1,
                  0, 1, 0, 1,
                  0, 0, 1, 0],
                 [0, 1, 0, 0,
                  1, 0, 1, 0,
                  1, 0, 1, 0,
                  0, 1, 0, 0],
                 [1, 0, 0, 0,
                  1, 0, 0, 0,
                  1, 0, 0, 0,
                  1, 0, 0, 0],
                 [0, 1, 0, 0,
                  1, 1, 0, 0,
                  0, 1, 0, 0,
                  1, 1, 1, 0],
                 [0, 0, 1, 0,
                  0, 1, 1, 0,
                  0, 0, 1, 0,
                  0, 1, 1, 1],
                 [0, 0, 0, 0,
                  0, 0, 1, 0,
                  0, 1, 0, 1,
                  0, 0, 1, 0],
                 [0, 1, 0, 0,
                  1, 0, 1, 0,
                  0, 1, 0, 0,
                  0, 0, 0, 0],
                 [0, 0, 0, 0,
                  0, 1, 0, 0,
                  1, 0, 1, 0,
                  0, 1, 0, 0],
                 [0, 0, 1, 0,
                  0, 1, 0, 1,
                  0, 0, 1, 0,
                  0, 0, 0, 0]])
tr = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0]])
a = 0.005
h1 = 35
h2 = 35
w01 = 2 * np.random.random((16, h1)) - 1
w12 = 2 * np.random.random((h1, h2)) - 1
w23 = 2 * np.random.random((h2, 2)) - 1
for i in range(100):
    for j in range(len(data)):
        l0 = data[j:j+1]
        l1 = relu(np.dot(l0, w01))
        l2 = relu(np.dot(l1, w12))
        l3 = np.dot(l2, w23)
        l3d = (l3 - tr[j:j+1])
        l2d = l3d.dot(w23.T)
        l2d *= relu2deriv(l2)
        l1d = l2d.dot(w12.T)
        l1d *= relu2deriv(l1)
        wd23 = l2.T.dot(l3d)
        wd12 = l1.T.dot(l2d)
        wd01 = l0.T.dot(l1d)
        w23 -= a * wd23
        w12 -= a * wd12
        w01 -= a * wd01
l0 = np.array([0, 0, 1, 0,
               0, 0, 1, 0,
               0, 0, 1, 0,
               0, 0, 1, 0])
l1 = relu(np.dot(l0, w01))
l2 = relu(np.dot(l1, w12))
print(np.round(np.dot(l2, w23), decimals=2))
pred = np.dot(l2, w23)
ii = 0
ma = -10000
for i in range(len(pred)):
    if (pred[i] > ma):
        ma = pred[i]
        ii = i
print(ii)
def gener(q):
    a = []
    for i in range(q):
        a.append(random.randint(0, 1))
    b = []
    b.append(a)
    return np.array(b)
def whatnum(w):
    if (w[0][0] > w[0][1] and w[0][0] >= 0.99):
        return 0
    elif (w[0][1] > w[0][0] and w[0][1] >= 0.99):
        return 1
    else:
        return 2
c = int(input())
d = 0
fig, ax = plt.subplots(nrows = 2, ncols = 5, figsize=(20, 8))
ax = ax.flatten()
while (d < 10):
    q = gener(16)
    q = q[0:1]
    l1 = relu(np.dot(q, w01))
    l2 = relu(np.dot(l1, w12))
    w = np.round(np.dot(l2, w23), decimals=2)
    g = whatnum(w)
    if (g == c and (w[0][0] - 0 + w[0][1] - 1)**2 <= 0.00000001):
        q = q.reshape(4, 4)
        im = ax[d].imshow(q, cmap = 'viridis', aspect = 'auto')
        d += 1
plt.show()
