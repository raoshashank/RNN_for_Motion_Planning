#import rospy
import tensorflow as tf 
import tensorflow.contrib.rnn as rnn
import numpy as np
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
13. Profit!! 
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

batch_size=64
num_features = 6 ##[desired_pose,current_pose]
output_sequence_length = 6 ##6 joint angles
num_units = 6
samples_per_path=100
time_steps = samples_per_path

input = tf.placeholder(dtype=tf.float32,shape=[None,time_steps,num_features],name="input")

joint_labels = tf.placeholder(dtype=tf.float32,shape=[None,time_steps,num_features],name="joint_labels")

initial_joint_angles=tf.placeholder(dtype=tf.float32,shape=[None,time_steps],name="initial_joint_angles")


with tf.name_scope("MultiCell_RNN"):

    lstm_cell1= rnn.BasicLSTMCell(
        name = "lstm_cell_1",
        forget_bias=1,
        num_units=num_units
    )

    '''
    lstm_cell2= rnn.BasicLSTMCell(
        name = "lstm_cell_2",
        forget_bias=1,
        num_units=num_units
    )
    '''
    multi_lstm_cells = rnn.MultiRNNCell(
        cells=[lstm_cell1],
        state_is_tuple=True

    )
    print(multi_lstm_cells.state_size)
    #with tf.variable_scope("state"):
    #state_c = tf.Variable(tf.zeros([batch_size, num_units], trainable=False)
    #state_c = tf.zeros([batch_size,num_units],dtype=tf.float32)
    #state_h = tf.zeros([batch_size,num_units],dtype=tf.float32)
    #state_h = tf.Variable(tf.random_uniform([batch_size, num_units], 0, 1), trainable=False)
    #state =([tf.contrib.rnn.LSTMStateTuple(c=state_c, h=state_h)])

    #state = multi_lstm_cells.zero_state(batch_size=batch_size,dtype=tf.float32)

    #print(state[0])
    #print(state[1])
    #with tf.variable_scope("rnn"):
    prediction,final_state = tf.nn.dynamic_rnn(multi_lstm_cells,dtype=tf.float32,inputs=input,initial_state=state)


    loss = tf.reduce_sum(tf.losses.mean_squared_error(labels=joint_labels,predictions=prediction))

    optim = tf.train.RMSPropOptimizer(name="RMSPropOptimizer",learning_rate=1e-6).minimize(loss)

if __name__ == '__main__':

    with tf.Session() as session:
        writer=tf.summary.FileWriter('/tmp/logs/1')
        writer.add_graph(session.graph)
        dummy_input = np.random.random(size=[batch_size,time_steps,num_features])
        dummy_labels = np.random.random(size=[batch_size,time_steps,num_features])
        dummy_initial_joint_values = np.random.random(size=[batch_size,time_steps])
        dummy_state_c = np.random.random(size=[batch_size,num_units])
        dummy_state_h = np.random.random(size=[batch_size,num_units])
        iters=1000
        init = tf.global_variables_initializer()
        session.run(init)
        for i in range(iters):
            session.run(
                optim,{
                    input:dummy_input,
                    joint_labels:dummy_labels,
                    initial_joint_angles:dummy_initial_joint_values,
                    #state_c:dummy_state_c,
                    #state_h:dummy_state_h
                }
            )

            if i%100:
                print(session.run(
                loss,{
                    input:dummy_input,
                    joint_labels:dummy_labels,
                    initial_joint_angles:dummy_initial_joint_values
                }
            ))










