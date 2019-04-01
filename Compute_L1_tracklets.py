import matlab.engine
import sys
import cv2
import os

import h5py
import numpy as np
from operator import itemgetter
from sklearn.metrics.pairwise import pairwise_distances

from L1tracklets import createTracklets as Tlets 

###Tracklets
track_ops = {}
track_ops['window_width'] = 50
track_ops['min_length'] = 5
track_ops['alpha']= 1
track_ops['beta'] = 0.02
track_ops['cluster_coeff'] = 0.75
track_ops['nearest_neighbors'] = 8
track_ops['speed_limit'] = 20
track_ops['threshold'] = 8

def main():
    eng = matlab.engine.start_matlab()
    eng.addpath(r'correlation_clustering/external/KL/',nargout=0)

    # h5 file to store tracklets
    tracklets_save = h5py.File('data/tracklets.h5', 'w')
    # h5 file to store final embeddings
    h5file = h5py.File('data/features.h5', 'r+')
    features_bb = list(h5file['emb'])  ## [cam,frame,x,y,w,h,features128d]
    length = len(features_bb)
    frame_num = int(features_bb[length-1][1])
    
    Tracklets = []
    for frame_start in range(0,frame_num,50):
        feature_bb = [x for x in features_bb if int(x[1]) in range(frame_start,frame_start+50)]
        if len(feature_bb) >1 :
            tracklet = Tlets.create_tracklets(track_ops,feature_bb,frame_start,frame_start+50,eng)
            Tracklets.extend(tracklet)
    Tracklets = sorted(Tracklets, key=itemgetter('startFrame','endFrame')) 
    
#    # Store the final Tracklets.
#    tracklets_s = tracklets_save.create_dataset('tracklets', data=Tracklets)
#    tracklets_save.close()
    
if __name__ == '__main__':
    main()
    


