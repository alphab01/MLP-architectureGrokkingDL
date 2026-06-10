import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(1)
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
                  0, 0, 0, 0],
                 [0, 1, 1, 1,
                  0, 1, 0, 1,
                  0, 1, 1, 1,
                  0, 0, 0, 0],
                 [1, 1, 1, 0,
                  1, 0, 1, 0,
                  1, 1, 1, 0,
                  0, 0, 0, 0],
                 [0, 0, 0, 0,
                  0, 1, 1, 1,
                  0, 1, 0, 1,
                  0, 1, 1, 1],
                 [0, 0, 0, 0,
                  1, 1, 1, 0,
                  1, 0, 1, 0,
                  1, 1, 1, 0],
                 [0, 1, 1, 1,
                  0, 1, 0, 1,
                  0, 1, 0, 1,
                  0, 1, 1, 1],
                 [1, 1, 1, 0,
                  1, 0, 1, 0,
                  1, 0, 1, 0,
                  1, 1, 1, 0]])
tr = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]])
a = 0.005
h1 = 500
w01 = 2 * np.random.random((16, h1)) - 1
w12 = 2 * np.random.random((h1, 2)) - 1
for i in range(100):
    for j in range(len(data)):
        l0 = data[j:j+1]
        l1 = relu(np.dot(l0, w01))
        l2 = np.dot(l1, w12)
        l2d = (l2 - tr[j:j+1])
        l1d = l2d.dot(w12.T)
        l1d *= relu2deriv(l1)
        wd12 = l1.T.dot(l2d)
        wd01 = l0.T.dot(l1d)
        w12 -= a * wd12
        w01 -= a * wd01
l0 = np.array([0, 0, 1, 0,
               0, 0, 1, 0,
               0, 0, 1, 0,
               0, 0, 1, 0])
l1 = relu(np.dot(l0, w01))
print(np.round(np.dot(l1, w12), decimals=2))
pred = np.dot(l1, w12)
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
    if (w[0][0] > w[0][1] and w[0][0] >= 1):
        return 0
    elif (w[0][1] > w[0][0] and w[0][1] >= 1):
        return 1
    else:
        return 2
c = int(input())
d = 0
fig, ax = plt.subplots(nrows = 2, ncols = 5, figsize=(20, 8))
ax = ax.flatten()
def ggwp(x):
    for i in range(len(x) - 1):
        c = 0
        for j in range(len(x)):
            c += x[i][j]
        if (c >= 3):
            return 0
    return 1
def what(x):
    if (x[0][0] == x[0][1] and x[0][0] == x[0][2] and x[0][0] == x[0][3]):
        return 0
    if (x[len(x) - 1][0] == x[len(x) - 1][1] and x[len(x) - 1][0] == x[len(x) - 1][2] and x[len(x) - 1][0] == x[len(x) - 1][3]):
        return 0
    for i in range(1, len(x) - 1):
        for j in range(1, len(x) - 1):
            if (x[i][j] == x[i][j - 1] and x[i][j] == x[i][j + 1] and x[i][j] == x[i - 1][j] and x[i][j] == x[i + 1][j]):
                return 0
        if (x[i][0] == x[i][1] and x[i][0] == x[i][2] and x[i][0] == x[i][3]):
            return 0
    return 1
while (d < 10):
    q = gener(16)
    q = q[0:1]
    l1 = relu(np.dot(q, w01))
    w = np.round(np.dot(l1, w12), decimals=2)
    g = whatnum(w)
    q = q.reshape(4, 4)
    if (g == c and (w[0][0] - 0 + w[0][1] - 1)**2 <= 0.00000001 and what(q) == 1 and ggwp(q) == 1):
        im = ax[d].imshow(q, cmap = 'viridis', aspect = 'auto')
        d += 1
plt.show()
