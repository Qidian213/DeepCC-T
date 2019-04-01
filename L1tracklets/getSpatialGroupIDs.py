import sys
import numpy as np
import scipy
import scipy.cluster.hierarchy as sch 
import matplotlib.pylab as plt
from collections import Counter

### Perfroms spatial groupping of detections and returns a vector of IDs
def getSpatialGroupIDs(use_groupping,currentDetectionsIDX,pairwiseDistances,track_ops):
    spatialGroupIDs = np.ones(len(currentDetectionsIDX),int)
    if use_groupping:
        agglomeration = sch.linkage(pairwiseDistances,method='single') # Agglomerative hierarchical cluster tree
        numSpatialGroups = round(track_ops['cluster_coeff']*len(currentDetectionsIDX)/track_ops['window_width']) 
        numSpatialGroups = max(numSpatialGroups,1)

        while True:
            spatialGroupIDs = sch.fcluster(agglomeration, t=numSpatialGroups,criterion='maxclust')
            count_dict = Counter(spatialGroupIDs)
            max_key = max(count_dict,key = count_dict.get) 
            if count_dict[max_key] <=150:
                return spatialGroupIDs
            numSpatialGroups = numSpatialGroups+1

