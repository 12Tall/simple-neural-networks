import numpy as np

z = [1, 2, 3, 4, 1, 2, 3]
exp_z = np.exp(z-np.max(z)) # 减去最大值，防止指数爆炸
print(exp_z)

sum = np.sum(exp_z)
print(sum)

softMax=exp_z/sum

print(softMax)