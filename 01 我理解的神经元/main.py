import numpy as np


def loadData():
    orgData = np.loadtxt(
        './data.csv',
        delimiter=',',
        usecols=(0, 1, 2, 3),
        skiprows=1
    )

    expData = orgData[:, 0:3]
    resData = orgData[:, 3:4]
    d = np.insert(expData, 3, 1, axis=1)
    return d, resData


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


def Gradient(expData, resData, a):
    return np.dot(expData.T, a - resData)/len(expData)


def main():
    alpha = 0.3
    expData, resData = loadData()
    weight = initWeight(expData.shape)

    for i in range(10000):
        a = Sigmoid(np.dot(expData, weight))
        # e = Loss(a, resData)
        grad = Gradient(expData, resData, a)
        weight -= grad*alpha

    print(1/(1+np.exp(-np.dot([[1, 0, 0,1]], weight))))
    pass


if __name__ == "__main__":
    main()
