import numpy as np

x = [3,7,5,2,1,8,9,6]
x = np.array(x)

inds = np.argsort(x)

b = x[inds]

print(x)
print(inds)
print(b)
