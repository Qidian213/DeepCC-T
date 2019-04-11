###cluster.py
import scipy
import scipy.cluster.hierarchy as sch
from scipy.optimize import linear_sum_assignment
from scipy.cluster.vq import vq,kmeans,whiten
import numpy as np
import matplotlib.pylab as plt

points=scipy.randn(20,4)  
print(points)

disMat = sch.distance.pdist(points,'euclidean') 
print(disMat)

Z=sch.linkage(disMat,method='average') 
print(Z)
#plot_dendrogram.png
P=sch.dendrogram(Z)
plt.savefig('plot_dendrogram.png')
#linkage matrix Z:
cluster= sch.fcluster(Z, t=1,criterion='inconsistent') 

print ("Original cluster by hierarchy clustering:\n",cluster)


