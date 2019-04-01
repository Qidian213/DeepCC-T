#!/usr/bin/env python3
from argparse import ArgumentParser
from importlib import import_module
from itertools import count
import os

import h5py
import json
import numpy as np
import tensorflow as tf

from aggregators import AGGREGATORS
import common
from duke_utils import *
import scipy.io as sio
import functools

parser = ArgumentParser(description='Embed a dataset using a trained network.')

# Required

parser.add_argument(
    '--experiment_root', required=True,
    help='Location used to store checkpoints and dumped data.')

parser.add_argument(
    '--dataset_path', default='/home/zzg/Datasets/Duke/', 
    help='Dataset root.')

parser.add_argument(
    '--detections_path', default=None, required=True, 
    help='Detections .mat file')

parser.add_argument(
    '--filename', default=None, required=True,
    help='Name of the HDF5 file in which to store the embeddings')

# Optional

parser.add_argument(
    '--checkpoint', default=None,
    help='Name of checkpoint file of the trained network within the experiment '
         'root. Uses the last checkpoint if not provided.')

parser.add_argument(
    '--loading_threads', default=8, type=common.positive_int,
    help='Number of threads used for parallel data loading.')

parser.add_argument(
    '--batch_size', default=256, type=common.positive_int,
    help='Batch size used during evaluation, adapt based on available memory.')

parser.add_argument(
    '--flip_augment', action='store_true', default=False,
    help='When this flag is provided, flip augmentation is performed.')

parser.add_argument(
    '--crop_augment', choices=['center', 'avgpool', 'five'], default=None,
    help='When this flag is provided, crop augmentation is performed.'
         '`avgpool` means the full image at the precrop size is used and '
         'the augmentation is performed by the average pooling. `center` means'
         'only the center crop is used and `five` means the four corner and '
         'center crops are used. When not provided, by default the image is '
         'resized to network input size.')

parser.add_argument(
    '--aggregator', choices=AGGREGATORS.keys(), default=None,
    help='The type of aggregation used to combine the different embeddings '
         'after augmentation.')
         
parser.add_argument(
    '--quiet', action='store_true', default=False,
    help='Don\'t be so verbose.')
    
def main():
    # Verify that parameters are set correctly.
    args = parser.parse_args()
    
    # Load the args from the original experiment.
    args_file = os.path.join(args.experiment_root, 'args.json')

    if os.path.isfile(args_file):
        if not args.quiet:
            print('Loading args from {}.'.format(args_file))
        with open(args_file, 'r') as f:
            args_resumed = json.load(f)

        # Add arguments from training.
        for key, value in args_resumed.items():
            args.__dict__.setdefault(key, value)

        # A couple special-cases and sanity checks
        if (args_resumed['crop_augment']) == (args.crop_augment is None):
            print('WARNING: crop augmentation differs between training and '
                  'evaluation.')
        args.image_root = args.image_root or args_resumed['image_root']
    else:
        raise IOError('`args.json` could not be found in: {}'.format(args_file))
        
    if not args.quiet:
        print('Evaluating using the following parameters:')
        for key, value in sorted(vars(args).items()):
            print('{}: {}'.format(key, value))

    # Load the data from the CSV file.

    net_input_size = (args.net_input_height, args.net_input_width) # 256*128
    pre_crop_size = (args.pre_crop_height, args.pre_crop_width) # 288*144

    # Load detections
    matfile = sio.loadmat(args.detections_path)
    detections = matfile['detections']
    num_detections = detections.shape[0]
    
    # Setup a tf Dataset generator
    generator = functools.partial(detections_generator, args.dataset_path, detections, net_input_size[0], net_input_size[1])
    dataset = tf.data.Dataset.from_generator(generator, tf.float32, tf.TensorShape([net_input_size[0], net_input_size[1], 3]))
    
    modifiers = ['original']
    modifiers = [o + '_resize' for o in modifiers]
    
    # Group it back into PK batches.
    dataset = dataset.batch(args.batch_size)
    images = dataset.make_one_shot_iterator().get_next()
    
    # Create the model and an embedding head.
    model = import_module('nets.' + args.model_name)
    head = import_module('heads.' + args.head_name)
    
    endpoints, body_prefix = model.endpoints(images, is_training=False)
    
    with tf.name_scope('head'):
        endpoints = head.head(endpoints, args.embedding_dim, is_training=False)

    with h5py.File(args.filename, 'w') as f_out, tf.Session() as sess:
        # Initialize the network/load the checkpoint.
        checkpoint = tf.train.latest_checkpoint(args.experiment_root)
        print('Restoring from checkpoint: {}'.format(checkpoint))
        
        tf.train.Saver().restore(sess, checkpoint)
        
        # Go ahead and embed the whole dataset, with all augmented versions too.
        emb_storage = np.zeros((num_detections * len(modifiers), args.embedding_dim), np.float32)
        
        for start_idx in count(step=args.batch_size):
            try:
                print(start_idx)
                emb = sess.run(endpoints['emb'])
                print('\rEmbedded batch {}-{}/{}'.format(start_idx, start_idx + len(emb), len(emb_storage)),flush=True, end='')
                emb_storage[start_idx:start_idx + len(emb)] = emb
            except tf.errors.OutOfRangeError:
                break  # This just indicates the end of the dataset.
                
        # Store the final embeddings.
        emb_dataset = f_out.create_dataset('emb', data=emb_storage)

        # Store information about the produced augmentation and in case no crop
        # augmentation was used, if the images are resized or avg pooled.
        f_out.create_dataset('augmentation_types', data=np.asarray(modifiers, dtype='|S'))

    print(num_detections)
    print(modifiers)
    print("test---line\n")


if __name__ == '__main__':
    main()


