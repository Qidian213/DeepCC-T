import sys
import cv2
import os
import numpy as np
import tensorflow as tf
from importlib import import_module
from tripletreid.head_part import fc1024 as head
from tripletreid.body_part import resnet_v1_50 as model 
from external.utils import get_bb_image

class ImageEncoder(object):

    def __init__(self, checkpoint_filename):
        self.input_var = tf.placeholder(tf.float32, [None,256, 128, 3], name='image') ;
        endpoints, body_prefix = model.endpoints(self.input_var, is_training=False)
        with tf.name_scope('head'):
            endpoints = head.head(endpoints, 128, is_training=False)
            
        config = tf.ConfigProto()  
        config.gpu_options.allow_growth=True   ## tensorflow Adaptive GPU memory 
        self.session = tf.Session(config=config)  
        
        # Initialize the network/load the checkpoint.
        checkpoint = tf.train.latest_checkpoint(checkpoint_filename)
        tf.train.Saver().restore(self.session, checkpoint)
        
        self.output_var = endpoints['emb']
        
    def __call__(self, data_x, batch_size=32):
        features = np.zeros((len(data_x), 128), np.float32)
        features = self.session.run(self.output_var , feed_dict = {self.input_var: data_x})
        return features
        
def create_box_encoder(model_filename, batch_size=32):
    image_encoder = ImageEncoder(model_filename)

    def encoder(image, boxes,camera_size):
        image_patches = []
        for box in boxes:
            patch = get_bb_image(image,box,camera_size)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)
    return encoder
    


