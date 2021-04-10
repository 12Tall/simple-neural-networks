
import sys # NOQA: E402
sys.path.append("..") # NOQA: E402
# 123
print(1)
import numpy as np
import my_tools.activation as act
from my_tools.loss import Loss


def loadData():
    org_data = np.loadtxt(
        './data.csv',
        delimiter=',',
        usecols=(0, 1, 2, 3),
        skiprows=1
    )

    exp_data = org_data[:, 0:3]
    res_data = org_data[:, 3:4]
    return insertConst(exp_data), res_data


def insertConst(mat):  # 为输入参数矩阵插入常数项1
    return mat
    r, c = mat.shape
    return np.insert(mat, c, 1, axis=1)


def generateWeights(shape):
    weight = 2 * np.random.random(shape)-1
    return weight


def main():
    alpha = 0.3
    np.random.seed(1)
    LeakyReLU = act.LeakyReLU()
    Sigmoid = act.Sigmoid()

    exp_data, res_data = loadData()

    a0 = exp_data
    y = res_data
    w1 = generateWeights((3, 5))
    w2 = generateWeights((5, 1))
    for i in range(100000):
        # fp
        z1 = np.dot(a0, w1)
        a1 = Sigmoid(z1)
        z2 = np.dot(a1, w2)
        a2 = Sigmoid(z2)
        loss = Loss(a2, y)
        g_a2 = loss.gradient()
        g_z2 = Sigmoid.gradient(z2)
        g_w2 = np.dot(a1.T, g_a2*g_z2)/len(a1)
        g_a1 = np.dot(g_a2*g_z2,w2.T)
        g_z1 = Sigmoid.gradient(z1)
        g_w1 = np.dot(a0.T, g_z1*g_a1)/len(a0)
        w1 -= alpha*g_w1
        w2 -= alpha*g_w2

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
    s2 = Sigmoid(s1)
    s4 = np.dot(s2, w2)
    s5 = Sigmoid(s4)
    print(s5)


if __name__ == "__main__":
    main()
