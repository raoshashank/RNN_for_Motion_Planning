import tensorflow as tf
import numpy as np
import os,os.path
model_ckpt_path = "/home/shashank/catkin_ws/src/rnn_ur5/modelckpt/"
files=os.listdir(model_ckpt_path)
files_meta = [i for i in files if i.endswith('.meta')]
files.sort()
meta_file = model_ckpt_path+files[-1]
batch_size=64
num_features = 6 ##[desired_pose,current_pose]
# samples_per_path=100
# time_steps = samples_per_path
time_steps=100
num_cells = 1
num_units = 6
max_iters = 1000
ckpt_start = 0
#print meta_file
tf.reset_default_graph()
with tf.Session() as sess:
   model=tf.train.import_meta_graph(meta_file)
   model.restore(sess,tf.train.latest_checkpoint(model_ckpt_path))  
   graph=tf.get_default_graph()
   sess.run(tf.global_variables_initializer())
   inputs=graph.get_tensor_by_name('input_data:0')
   initial=graph.get_tensor_by_name('initial_state:0')
      
   predictions = graph.get_operation_by_name('MultiCell_RNN')
   dummy_input = np.random.random(size=[batch_size,time_steps,num_features])
   dummy_initial_joint_values = np.random.random(size=[num_cells, 2, batch_size,num_units])
   print sess.run(predictions,{inputs:dummy_input,initial:dummy_initial_joint_values})
