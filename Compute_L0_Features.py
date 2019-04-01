import sys
import cv2
import os
from timeit import time
from argparse import ArgumentParser
from importlib import import_module
from itertools import count

import h5py
import json
import numpy as np
import tensorflow as tf
import functools

##tripletreid
from tripletreid import common
from tripletreid import generate_features 
##openpose
from external import Deepose
from external import utils
parser = ArgumentParser(description='Embed a dataset using a trained network.')

# Required
parser.add_argument(
    '--dataset_path', default='/home/zzg/Datasets/Duke/', 
    help='Dataset root.')

parser.add_argument(
    '--filename', default='data/features.h5',
    help='Name of the HDF5 file in which to store the embeddings')

# Optional

parser.add_argument(
    '--batch_size', default=256, type=common.positive_int,
    help='Batch size used during evaluation, adapt based on available memory.')

def main():
    # Verify that parameters are set correctly.
    args = parser.parse_args()

    ## openpose
    Pose = Deepose.DeepPose()
    
    # Load the args from the original experiment.
    args_file = 'tripletreid/data/args.json'
    print('Loading args from {}.'.format(args_file))
    fid = open(args_file ,'r')
    args_resumed = json.load(fid)
    for key ,value in args_resumed.items():
        args.__dict__.setdefault(key,value)
        
    # create features encoder
    model_filename = 'tripletreid/data/'
    encoder = generate_features.create_box_encoder(model_filename,batch_size=1)
    
    video_capture = cv2.VideoCapture('/home/zzg/Datasets/Duke/dukevideos/camera5/00002.MTS')
    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    camera_size = (w,h)
    
    #h5 file to store final embeddings
    f_out = h5py.File(args.filename, 'w')
    emb_storage = []
    frame_num =0
    cam = 1 
    
    fps = 0.0
    while True:
        ret, frame = video_capture.read()
        if ret != True:
            break;
        t1 = time.time()
        keypoints,output_image = Pose.getKeypoints(frame)
        bboxs = utils.bbox(keypoints,1.25)
        if len(bboxs) != 0:
            for bb in bboxs:
                cv2.rectangle(output_image,(int(bb[0]), int(bb[1])), (int(bb[0]+bb[2]), int(bb[1]+bb[3])),(255,255,255), 2)
            cv2.imshow(" ",output_image);
            features = encoder(frame,bboxs,camera_size)
            for bbox, feature in zip(bboxs, features):
                fe_bb = np.zeros(134, np.float32)  # [cam,frame,x,y,w,h,feature]
                fe_bb[0] = cam
                fe_bb[1] = frame_num
                fe_bb[2:6] = bbox
                fe_bb[6:] = feature
                emb_storage.append(fe_bb)
        frame_num = frame_num+1
        
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("frame_num= %d   fps= %f"%(frame_num,fps))
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    
    # Store the final embeddings.
    emb_dataset = f_out.create_dataset('emb', data=emb_storage)
    video_capture.release()
    cv2.destroyAllWindows()
    f_out.close()
    
if __name__ == '__main__':
    main()
    




