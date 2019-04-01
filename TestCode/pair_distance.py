import sys

from sklearn.metrics.pairwise import pairwise_distances

A = [[1,1],[2,3],[7,9],[5,6],[6,1],[8,2]]
B = [[2,3]]
cor_distance = pairwise_distances(B,A,metric='euclidean')
print(cor_distance[0][0])
