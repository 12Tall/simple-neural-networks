import numpy as np
import activation
import layer


def main():
    np.random.seed(1)
    a0, y = loadData()
    
    w0, b0 = layer.Layer.randomW_B(3, 4)
    w1, b1 = layer.Layer.randomW_B(4, 1)
    l0 = layer.Layer(
        w0,
        b0,
        learning_rate=0.3,
        act=activation.LeakyReLU()
    )
    l1 = layer.Layer(
        w1,
        b1,
        learning_rate=0.3,
        act=activation.Sigmoid()
    )

    for i in range(10000):
        a1 = l0.fp(a0)
        a2 = l1.fp(a1)
        loss = Loss(a2, y)  # 反向过程，由后至前反推一遍
        g_a2 = loss.gradient()
        g_a1 = l1.bp(g_a2)
        l0.bp(g_a1)

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
    s1 = l0.fp(s0)
    s2 = l1.fp(s1)
    print(s2)


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
