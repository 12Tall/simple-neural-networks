import numpy as np
import activation


def main():
  np.random.seed(1)  

  alpha = 0.03  # 学习率 learning rate
  sigmoid = activation.Sigmoid()

  a0,y = loadData()                # A,B,C,D
                                   # 1,1,1,0
                                   # 0,1,1,1
                                   # 0,0,1,0
                                   # 1,0,1,1
                                   # 0,0,0,0
                                   # 0,1,0,1
                                   # 1,1,0,0
                                   # 1,0,0,1
  # 每层节点个数：l0-3, l1-4, l2-1，可以确定权重矩阵的维度
  w1 = generateWeights((3, 4))
  w2 = generateWeights((4, 1))

  for i in range(100000):  
    z1 = np.dot(a0, w1)  # 正向过程：每一级的输出都是下一级的输入
    a1 = sigmoid(z1)  #
  
    z2 = np.dot(a1, w2)
    a2 = sigmoid(z2)

    loss = Loss(a2, y)  # 反向过程，由后至前反推一遍
    grad_a2 = loss.gradient()
    grad_z2 = grad_a2 * sigmoid.gradient(z2)
    grad_w2 = np.dot(a1.T, grad_z2)/len(a1)  # 成本函数对应的是平均值
    
    grad_a1 = np.dot(grad_z2, w2.T)  # 死公式，但很有意思
    # 这一步非常重要，而且不太好理解，最好能多通过流程图在稿纸上多演算几次运算过程和矩阵乘法
    grad_z1 = grad_a1 * sigmoid.gradient(z1)
    grad_w1 = np.dot(a0.T, grad_z1)/len(a0)
    # 看反向过程的每一层都有求da, dz, dw。而且基本都是死公式

    w1 -= alpha * grad_w1
    w2 -= alpha * grad_w2
  
  # 验证
  s0 = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]]
  s1 = np.dot(s0, w1)
  s2 = sigmoid(s1)
  s3 = np.dot(s2, w2)
  s4 = sigmoid(s3)
  print(s4)



def loadData():
  org_data = np.loadtxt(
    './data.csv',
    delimiter=',',
    usecols=(0, 1, 2, 3),
    skiprows=1
  )
  exp_data = org_data[:, 0:3]
  res_data = org_data[:, 3:4]
  return exp_data, res_data

def generateWeights(shape):  # 生成权重矩阵
    weight = 2 * np.random.random(shape)-1
    return weight

class Loss():  # 损失函数（对于单个样本来说
    def __init__(self, a: np.array, y: np.array):
        self.a = a
        self.y = y

    def __call__(self):
        a = self.a
        y = self.y
        return -(y*np.log(a))-(1-y)*np.log(1-a)

    def gradient(self):
        a = self.a
        y = self.y
        return -y/a + (1-y)/(1-a)

if __name__ == "__main__":
    main()