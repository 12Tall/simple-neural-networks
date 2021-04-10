import numpy as np


def loadData():
    orgData = np.loadtxt(
        './data.csv',
        delimiter=',',
        usecols=(0, 1, 2, 3),
        skiprows=1
    )

    exp_data = orgData[:, 0:3]
    res_data = orgData[:, 3:4]
    # d = np.insert(exp_data, 3, 1, axis=1)
    return exp_data, res_data


def initWeight(shape):
    rows, cols = shape
    np.random.seed(1)
    weight = 2 * np.random.random((cols, 1))-1
    return weight


def Sigmoid(z):
    return 1/(1+np.exp(-z))


def Loss(a, y):
    return -(y*np.log(a))-(1-y)*np.log(1-a)


def Cost(loss):
    return np.mean(loss, axis=0)


def Gradient(exp_data, res_data, a):
    return np.dot(exp_data.T, a - res_data)/len(exp_data)


def main():
    alpha = 0.3
    exp_data, res_data = loadData()
    weight = initWeight(exp_data.shape)

    for i in range(10000):
        a = Sigmoid(np.dot(exp_data, weight))
        # e = Loss(a, res_data)
        grad = Gradient(exp_data, res_data, a)
        weight -= grad*alpha

    print(1/(1+np.exp(-np.dot([[0, 1, 1]], weight))))
    pass


if __name__ == "__main__":
    main()
