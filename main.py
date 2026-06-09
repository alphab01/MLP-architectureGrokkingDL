import numpy as np

np.random.seed(1)

def relu(x):
  return (x > 0) * x
def relu2deriv(ou):
  return ou > 0

data = np.array([[0,0,0,0,1,
                  0,0,0,1,1,
                  0,0,0,0,1,
                  0,0,0,0,1,
                  0,0,0,0,1],
                 [0,0,0,0,1,
                  0,0,0,0,1,
                  0,0,0,0,1,
                  0,0,0,0,1,
                  0,0,0,0,1],
                 [0,0,1,0,0,
                  0,1,0,1,0,
                  0,1,0,1,0,
                  0,1,0,1,0,
                  0,0,1,0,0]])
g = np.array([[0,1], [0,1], [1,0]])
h1 = 15
h2 = 15
we01 = 2 * np.random.random((25, h1)) - 1
we12 = 2 * np.random.random((h1, h2)) - 1
we23 = 2 * np.random.random((h2, 2)) - 1
a = 0.005

for i in range(1000):
  for j in range(len(g)):
    l0 = data[j:j+1]
    l1 = relu(np.dot(l0, we01))
    l2 = relu(np.dot(l1, we12))
    l3 = np.dot(l2, we23)
    l3d = l3 - g[j:j + 1]
    l2d = l3d.dot(we23.T)
    l2d *= relu2deriv(l2)
    l1d = l2d.dot(we12.T)
    l1d *= relu2deriv(l1)
    l0d = l1d.dot(we01.T)
    l0d *= relu2deriv(l0)
    wd23 = l2.T.dot(l3d)
    wd12 = l1.T.dot(l2d)
    wd01 = l0.T.dot(l1d)
    we23 -= wd23 * a
    we12 -= wd12 * a
    we01 -= wd01 * a
l0 = np.array([0,0,1,0,0,
               0,1,0,1,0,
               0,1,0,1,0,
               0,1,0,1,0,
               0,0,1,0,0])
l1 = relu(np.dot(l0, we01))
l2 = relu(np.dot(l1, we12))
print(np.dot(l2, we23))
