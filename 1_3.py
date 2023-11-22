#Problem 1.3 in assignment 1A
import numpy as np
from matplotlib import pyplot as plt
import math


def GenerateDatapoints(mu, tau, N):
  '''
    Generate Gaussian distributed datasets of size N with 
    mean mu and precision tau
  '''
  sigma = 1/np.sqrt(tau)
  X = np.random.normal(mu, sigma, N)
  return X

def PlotHistogram(X):
  plt.hist(X, bins='auto', edgecolor='black')
  plt.xlabel('Value')
  plt.ylabel('Frequency')
  plt.title('Histogram of Dataset')

  plt.show()

def ML_estimate(X):
  pass


def Q_1_3_12():
  N = [10, 100, 1000]
  mu = 1
  tau = 0.5
  for n in N:
    data = GenerateDatapoints(mu, tau, n)
    PlotHistogram(data)



if __name__ == '__main__':
  Q_1_3_12()
