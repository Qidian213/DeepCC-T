import numpy as np

arr = np.array([1, 13, 2, 24, 5,23,76,45,0])

inds = np.argsort(arr)
print(inds[:3])
