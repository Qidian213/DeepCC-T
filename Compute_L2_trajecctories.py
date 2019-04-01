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
traje_ops = {}
traje_ops['appearance_groups'] = 0 ## determined automatically when zero
traje_ops['alpha'] = 1
traje_ops['beta']= 0.01
traje_ops['window_width'] = 300
traje_ops['overlap'] = 150
traje_ops['speed_limit'] = 30
traje_ops['indifference_time'] = 100
traje_ops['threshold'] = 8

def main():
    eng = matlab.engine.start_matlab()
    eng.addpath(r'correlation_clustering/external/KL/',nargout=0)

    #h5 file to store final embeddings
    h5file = h5py.File('data/features.h5', 'r+')
    features_bb = list(h5file['emb'])  ## [cam,frame,x,y,w,h,features128d]
    length = len(features_bb)
    frame_num = int(features_bb[length-1][1])
    
if __name__ == '__main__':
    main()
    
