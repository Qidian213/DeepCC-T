import tensorflow as tf
import os
from tensorflow.python.tools import freeze_graph

import tensorflow as tf

meta_path = 'experiments/demo_hnm/checkpoint-25000.meta' # Your .meta file

with tf.Session() as sess:

    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint('experiments/demo_hnm/'))

    # Output nodes
    output_node_names =[n.name for n in tf.get_default_graph().as_graph_def().node]

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('tripletreid-25000.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())

