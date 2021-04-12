import numpy as np
import activation


class Layer(object):
    def __init__(self,
                 weights: np.array,  # 权重矩阵
                 b: np.array,  # 常数项
                 learning_rate=0.01,
                 act: activation.Activation = activation.ReLU(),  # 激活函数
                 ) -> None:
        self.weights = weights
        self.b = b
        self.learning_rate = learning_rate
        self.activation = act

    def fp(self, a: np.array) -> np.array:
        self.a = a
        z = np.dot(a, self.weights) + self.b
        self.z = z
        return self.activation(self.z)

    def bp(self, dNext: np.array) -> np.array:  # 传入下一层的偏导
        dz = dNext*self.activation.gradient(self.z)
        dw = np.dot(self.a.T, dz)/len(self.a)
        self.weights -= self.learning_rate*dw
        db = np.mean(dz, axis=0)
        self.b -= self.learning_rate*db
        dPrev = np.dot(dz, self.weights.T)
        return dPrev

    def randomW_B(row, col):
        weight = 2 * np.random.random((row, col))-1
        b = 2 * np.random.random((1, col))-1
        return weight, b
