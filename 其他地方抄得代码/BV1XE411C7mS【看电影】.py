# -*- coding:utf-8 -*-

''' 需求：预测D 会不会去看电影
（中文表格排版没有用ABCD 好看）

已知：前几次去看电影的经历如下：
-------------------
| A | B | C | D |
-------------------
| 0 | 0 | 1 | 0 |
| 1 | 1 | 1 | 1 |
| 1 | 0 | 1 | 1 |
| 0 | 0 | 1 | 0 |
| 0 | 1 | 1 | 0 |
-------------------
| 1 | 1 | 0 | ? |

解题思路：可以根据数字电路里面的真值表去判断，
但是这是一个概率问题。所以凡是不能这么理想化
'''


from numpy import array, exp, random, dot


# 采集ABC 的数据
X = array([
    # A, B, C    # D
    [0, 0, 1],   # 0
    [1, 1, 1],   # 1
    [1, 0, 1],   # 1
    [0, 1, 1]])  # 0

# 采集D 对应的结果
Y = array([[0, 1, 1, 0]]).T  # 注意转置

""" 
每次调用random.seed(1) 都会重置随机数序列。只要种子值是一样的，
那么无论程序调用多少次，产生的随机数序列都是一样的。好像是与机器
无关。不设置的话每次产生的随机数都不一样。测试代码：

random.seed(5)
arr = random.random(3)
print(arr)
random.seed(5)
arr = random.random(3)
"""
#random.seed(1)  # 随机数种子。
# random.random((m, n)) 生成一个m行n列的随机数矩阵
# random.random(n) 生成一个n列的随机数矩阵
weights = 5 * random.random((3, 1)) - 1  # 初始化权重，-1<weight<1 的序列

for it in range(1000):
    # z = [A,B,C] · [w1, w2, w3].T = dot(X, weights)
    # dot(X, weights) 相当于对每一组数据都进行了一次点乘，得到了一个理论输出量的集合z
    # Sigmoid 函数：Phi(z) = 1/(1+exp(-z)) 用于将数值归一化、标准化
    # Sigmoid 函数的曲线是一个s 型曲线
    output = 1/(1+exp(-dot(X, weights)))
    
    # 将z 的集合与采集到的结果比较、计算误差
    error = Y - output
    # 计算在z 处的补偿量，
    # 也就是Sigmoid 函数变化的速度乘以误差，之所以取导数、斜率是为了避免超调、震荡
    # 无限逼近，永不相交
    delta = error * output * (1-output)
    print('delta:',dot(X.T, delta))
    # 更新权重，加等于步长乘以输入
    weights += dot(X.T, delta)

print(weights)
print(1/(1+exp(-dot([[1, 0, 0]], weights))))
