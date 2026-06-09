import numpy as np

def relu(x):
  return (x > 0) * x
def relu2deriv(ou):
  return ou > 0
