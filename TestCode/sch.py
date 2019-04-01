###cluster.py
#导入相应的包
import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq,kmeans,whiten
import numpy as np
import matplotlib.pylab as plt


#生成待聚类的数据点,这里生成了20个点,每个点4维:
points=scipy.randn(20,4)  
print(points)
#1. 层次聚类
#生成点与点之间的距离矩阵,这里用的欧氏距离:
disMat = sch.distance.pdist(points,'euclidean') 
print(disMat)
#进行层次聚类:
Z=sch.linkage(disMat,method='average') 
print(Z)
#将层级聚类结果以树状图表示出来并保存为plot_dendrogram.png
P=sch.dendrogram(Z)
plt.savefig('plot_dendrogram.png')
#根据linkage matrix Z得到聚类结果:
cluster= sch.fcluster(Z, t=1,criterion='inconsistent') 

print ("Original cluster by hierarchy clustering:\n",cluster)

