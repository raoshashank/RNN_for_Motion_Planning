import tensorflow as tf
import numpy as np
import os,os.path
model_ckpt_path = "/home/shashank/catkin_ws/src/rnn_ur5/modelckpt/"
files=os.listdir(model_ckpt_path)
files_meta = [i for i in files if i.endswith('.meta')]
files.sort()
meta_file = model_ckpt_path+files[-1]
print meta_file
with tf.Session() as sess:
   model=tf.train.import_meta_graph(meta_file)
   model.restore(sess,tf.train.latest_checkpoint(model_ckpt_path))
   print sess.run('prediction')

    

