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
g = np.array([[0,1], [0,1], [1,0])
h = 25
we01 = 2 * np.random.random((25, h)) - 1
we12 = 2 * np.random.random((h, 2)) - 1
