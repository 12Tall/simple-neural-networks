# 重构
import numpy as np

class Activation(object):
  def __call__(self, z:np.array) -> np.array:
      pass
  def gradient(self, z:np.array) -> np.array:
    pass

class Sigmoid(Activation):
  def __call__(self, z:np.array) -> np.array:
    return 1/(1+np.exp(-z))
  def gradient(self, z:np.array) -> np.array:
    a = self.__call__(z)
    return a*(1-a)

class Tanh(Activation):
  def __call__(self, z:np.array) -> np.array:
    return 1-2/(1+np.exp(2*z))
  def gradient(self, z:np.array) -> np.array:    
    a = self.__call__(z)
    return 1-np.power(a, 2)

class ReLU(Activation):
  def __call__(self, z:np.array) -> np.array:
    return np.where(z>0, z, 0)
  def gradient(self, z:np.array) -> np.array:    
    return np.where(z>0, 1, 0)

class LeakyReLU(Activation):
  def __init__(self, alpha=0.1) -> None:
      self.alpha = alpha
  def __call__(self, z:np.array) -> np.array:
    return np.where(z>0, z, self.alpha*z)
  def gradient(self, z:np.array) -> np.array:    
    return np.where(z>0, 1, self.alpha)