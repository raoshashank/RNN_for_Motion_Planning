'''
    This network takes as input either the joint states or the pose of the end effector as input
    It will be trained on files generated from random trajectories on the ur5 gazebo simulator
    It outputs, the joint state input required for the given end pose

    Start with giving input as initial joint states and desired output as final joint states and let model learn
    the intermediate states

    /joint state publishes data of the form:



    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~WORKPLAN:~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    0. Design RNN architecture
    1. generate training data by running robot on random trajectories.abs
    2. Preprocess the training data so that it can be fed to the rnn
    4. Test RNN architecture on dummy values
    --Start Writing Paper---
    5. Convert Network to a ROS Service
    6. Test run on a small dataset
    7. Train Network on Dataset
    8. Test Network on new path on simulation.
    9. Generate training data from real robot
    10. Train Network on Real Bot data
    11. Test Network on Real Bot
    12. Publish Result
    1e. Profit!! 
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    [(x_d,y_d,z_d),x,y,z] is given as input for the LSTM and Joint space readings are taken as feedback to the LSTM
    Linear Trajectories are considered and waypoints are generated during training which are supplied as 
    (x,y,z) in the input 
    the first input is given as the current joint values [J-1]
    ==========================================
    input = [x_d,y_d,z_d,x_c,y_c,z_c]
    J-1 = initial robot joints

    LSTM output = Estimate of next joint states till end 
    ==========================================


    PLAN : 
    1. GENERATE DATA AND PASS THROUGH DUMMY NETWORK TO FIND MATRIX SIZES
    2. PASS OUTPUT DATA TO MODEL THROUGH SERVICE
    3. REFINE MODEL AND TEST
    4. TRAIN NETWORK ON DATASET
    5. TEST NETWORK
    6. REFINE AND REPEAT
'''
#import rospy
import tensorflow as tf 
import tensorflow.contrib.rnn as rnn
import numpy as np
from keras import backend as K
import os,os.path
from data_pb2 import Sequence
from prep_data import *

prime="/home/shashank/catkin_ws/src/rnn_ur5/"
data_home=prime+"dataset"
test_results=prime+"results"
file_name = home + 'sequence_%05d'
model_ckpt_path = "/home/shashank/catkin_ws/src/rnn_ur5/modelckpt/"
model_events = "/home/shashank/catkin_ws/src/rnn_ur5/modelevents/"

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
batch_size=64
num_features = 6 ##[desired_pose,current_pose]
# samples_per_path=100
# time_steps = samples_per_path
num_cells = 1
path, dirs, files = next(os.walk(home))
num_sequences = len(files)
file_name="sequence_$05d"
num_iters=1000

max_iters = 1000
ckpt_start = 0
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#length = tf.placeholder(tf.float32,shape=(),name='sequence_length')
inputs = tf.placeholder(dtype=tf.float32,shape=[None,None,num_features],name="input_data")
joint_labels = tf.placeholder(dtype=tf.float32,shape=[None,None,num_features],name="joint_labels")
initial_joint_angles = tf.placeholder(tf.float32, [num_cells, 2, None,num_features],name="initial_state")

with tf.name_scope("MultiCell_RNN"):
    lstm_cell= rnn.BasicLSTMCell(name = "lstm_cell_1", forget_bias=1, num_units=num_units)
    multi_lstm_cells = rnn.MultiRNNCell(cells=[lstm_cell]*num_cells, state_is_tuple=True,)
    print(multi_lstm_cells.state_size)

    state_per_layer_list = tf.unstack(initial_joint_angles, axis=0)
    rnn_tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1]) for idx in range(num_cells)])
    prediction,final_state = tf.nn.dynamic_rnn(multi_lstm_cells,dtype=tf.float32,inputs=inputs,initial_state=rnn_tuple_state,sequence_length=None)
    prediction=tf.identity(prediction,'prediction')
    final_state=tf.identity(final_state,'final_state')
    loss = tf.reduce_sum(tf.losses.mean_squared_error(labels=joint_labels,predictions=prediction))
    optim = tf.train.AdamOptimizer(name="Optimizer",learning_rate=1e-3).minimize(loss)
    loss_summary = tf.summary.scalar('loss',loss)
    saver = tf.train.Saver()

if __name__ == '__main__':

    inp,label,initial=prep_data(num_cells)
    train_inputs = inp[0:500]
    test_inputs = inp[500:]

    train_labels=label[0:500]
    test_labels = label[500:]
    
    train_initial=initial[:,:,0:500,:]
    test_initial=initial[:,:,500:,:]
    with tf.Session() as session:
        
        if len(os.listdir(model_ckpt_path)) == 0:
            init = tf.global_variables_initializer()
            session.run(init)
        else:
            saver.restore(session,  os.path.join(model_ckpt_path,"weights{}.ckpt".format(ckpt_start)))
        
        merge=tf.summary.merge_all()
    
        # if min_iters != 0:
            # saver.restore(session, os.path.join(model_ckpt_path,"weights{}.ckpt".format(min_iters)))
        # dummy_input = np.random.random(size=[batch_size,time_steps,num_features])
        # dummy_labels = np.random.random(size=[batch_size,time_steps,num_features])
        # dummy_initial_joint_values = np.random.random(size=[num_cells, 2, batch_size,num_units])
        # dummy_state_c = np.random.random(size=[batch_size,num_units])
        # dummy_state_h = np.random.random(size=[batch_size,num_units])
        writer=tf.summary.FileWriter(model_events)
        writer.add_graph(session.graph)
        for i in range(ckpt_start,ckpt_start+num_iters):
            [train_loss, summary]= session.run([optim, merge], {inputs:train_inputs, joint_labels:train_labels,initial_joint_angles:train_initial})
            # [train_loss, summary] = session.run([loss, merge],
            # {inputs:train_inputs, joint_labels:train_labels,initial_joint_angles:train_initial})
            writer.add_summary(summary,i)
            if i%100 == 0:
                print(session.run(loss,{inputs:train_inputs, joint_labels:train_labels, initial_joint_angles:train_initial}))
                saved_ckpt = saver.save(session, os.path.join(model_ckpt_path,"weights{}.ckpt".format(i)))










