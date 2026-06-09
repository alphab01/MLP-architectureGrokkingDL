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
h = 25
we01 = 2 * np.random.random((25, h)) - 1
we12 = 2 * np.random.random((h, 2)) - 1
a = 0.005

for i in range(100):
  for j in range(len(g)):
    l0 = data[j:j+1]
    l1 = relu(np.dot(l0, we01))
    l2 = np.dot(l1, we12)
    l2d = l2 - g[j:j + 1]
    l1d = l2d.dot(we12.T)
    l1d *= relu2deriv(l1)
    l0d = l1d.dot(we01.T)
    l0d *= relu2deriv(l0)
    wd12 = l1.T.dot(l2d)
    wd01 = l0.T.dot(l1d)
    we12 -= wd12 * a
    we01 -= wd01 * a
