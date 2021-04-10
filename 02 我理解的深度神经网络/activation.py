# activation.py

'''
梯度在函数调用时就能计算出来
这样做的好处可以避免遗漏
'''
import numpy as np

class Sigmoid():
  def __init__(self):
    pass

  def __call__(self, z):
    return 1/(1+np.exp(-z))

  def gradient(self, z):
    a = self.__call__(z)
    return a*(1-a)

class Tanh():
  def __init__(self):
    pass

  def __call__(self, z):
    ez = np.exp(z)
    e_z = np.exp(-z)
    return (ez-e_z)/(ez+e_z)

  def gradient(self,z):
    a = self.__call__(z)
    return 1- np.power(a,2)

class ReLU():
  def __init__(self):
    pass

  def __call__(self, z):
    return np.maximum(z,0)

  def gradient(self, z):
    return np.where(z>0, 1, 0)

class LeakyReLU():  # leakrelu = LeakyReLU(alpha=0.1)  ## 调用
  def __init__(self,alpha=0.01):
    self.alpha = alpha

  def __call__(self, z):
    return np.where(z>0, 1, self.alpha*z)

  def gradient(self, z):
    return np.where(z>0, 1, self.alpha)