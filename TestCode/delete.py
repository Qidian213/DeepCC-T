import numpy as np

a = [0,1,2,3,4,5,6,7,8,9]
indexs = [3,4,7,8]

print(a)
for index in sorted(indexs, reverse=True):
    del a[index]

print(a)
