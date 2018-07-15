'''
    This network takes as input either the joint states or the pose of the end effector as input
    It will be trained on files generated from random trajectories on the ur5 gazebo simulator
    It outputs, the joint state input required for the given end pose

    Start with giving input as initial joint states and desired output as final joint states and let model learn
    the intermediate states

    /joint state publishes data of the form:

    [(x_d,y_d,z_d),x,y,z] is given as input for the LSTM and Joint space readings are taken as feedback to the LSTM
    Linear Trajectories are considered and waypoints are generated during training which are supplied as 
    (x,y,z) in the input 
    the first input is given as the current joint values [J-1]
    ==========================================
    input = [x_d,y_d,z_d,x_c,y_c,z_c]
    J-1 = initial robot joints

    LSTM output = Estimate of next joint states till end 
    ==========================================

TODO:
1. Train for more layers and check difference
2. Inference RNN --> not moving at all
3. Add error term in training for distance from end point--> input and output vectors are same length sequences.
4. Closed Loop 
5. Try different error functions
6. Check direct IK performance
7. Give t+1 xyz as labels and try for motion planning?
8. 
'''

import tensorflow as tf 
import tensorflow.contrib.rnn as rnn
import numpy as np
import os,os.path
from data_pb2 import Sequence
from prep_data import prep_data
import sys
from data_generation import call_compute_fk_service,random_valid_end_point_generator,get_poses
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
import copy
import rospy
import moveit_msgs.msg
import actionlib
import moveit_commander
import geometry_msgs.msg
from geometry_msgs.msg import Pose
import math
import random
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.srv import GetPositionFKRequest
from moveit_msgs.srv import GetPositionFKResponse
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
from std_msgs.msg import Header
from data_pb2 import Sequence
import sys,os
prime="/home/shashank/catkin_ws/src/rnn_ur5/"
data_home=prime+"dataset"
file_name = data_home + 'sequence_%05d'
model_ckpt_path = "/home/shashank/catkin_ws/src/rnn_ur5/modelckpt/"
model_events = "/home/shashank/catkin_ws/src/rnn_ur5/modelevents/"

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
batch_size=64
num_features = 12 ##[desired_pose,initial_pose,current_pose]
features_out = 3
# samples_per_path=100
# time_steps = samples_per_path
num_cells = 10
path, dirs, files = next(os.walk(data_home))
num_sequences = len(files)
file_name="sequence_$05d"
num_iters=3000
num_units = features_out
ckpt_start = 0  
train_size= 1950

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#length = tf.placeholder(tf.float32,shape=(),name='sequence_length')
inputs = tf.placeholder(dtype=tf.float32,shape=[None,None,num_features],name="input_data")
joint_labels = tf.placeholder(dtype=tf.float32,shape=[None,None,features_out],name="joint_labels")
initial_joint_angles = tf.placeholder(tf.float32, [num_cells, 2, None,num_features],name="initial_state")
prediction = tf.Variable(dtype=tf.float32,name="prediction",initial_value=tf.zeros([0,0,features_out]),trainable=False)

with tf.name_scope("MultiCell_RNN"):
    lstm_cell= rnn.BasicLSTMCell(forget_bias=1, num_units=num_units,name="BasicLSTMCell")
    multi_lstm_cells = rnn.MultiRNNCell(cells=[lstm_cell]*num_cells, state_is_tuple=True)
    #print(multi_lstm_cells.state_size)

    state_per_layer_list = tf.unstack(initial_joint_angles, axis=0,name="state_per_layer_list")
    rnn_tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1]) for idx in range(num_cells)])
    
    prediction,final_state = tf.nn.dynamic_rnn(multi_lstm_cells,dtype=tf.float32,inputs=inputs)#initial_state=rnn_tuple_state,sequence_length=None)
    final_state=tf.identity(final_state,'final_state')
    loss = tf.reduce_sum(tf.losses.mean_squared_error(labels=joint_labels,predictions=prediction))
    optim = tf.train.AdamOptimizer(name="Optimizer",learning_rate=1e-3).minimize(loss)
    loss_summary = tf.summary.scalar('loss',loss)
    saver = tf.train.Saver()

if __name__ == '__main__':
    global group_arm,current_joint_values
    current_joint_values = JointState()
    moveit_commander.roscpp_initializer.roscpp_initialize(sys.argv)
    rospy.init_node('main_node', anonymous=True)
    #sub = rospy.Subscriber("joint_states",JointState,callback=joint_subscriber)
    #rospy.wait_for_service('rnn_service')
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_arm = moveit_commander.MoveGroupCommander("manipulator")
    group_arm.set_planner_id("RRTConnectkConfig")
          
    inp,label,initial=prep_data(num_cells)
    train_inputs = inp[0:train_size]
    test_inputs = inp[train_size:]

    train_labels=label[0:train_size]   
    test_labels = label[train_size:]
    
    train_initial=initial[:,:,0:train_size,:]
    test_initial=initial[:,:,train_size:,:]
    
    with tf.Session() as session:
        print("\n\n")
        if len(os.listdir(model_ckpt_path)) == 0:
            init = tf.global_variables_initializer()
            session.run(init)
        else:
            saver.restore(session,os.path.join(model_ckpt_path,"weights{}.ckpt".format(ckpt_start)))
        
        merge=tf.summary.merge_all()
    
        writer=tf.summary.FileWriter(model_events)
        writer.add_graph(session.graph) 
        if(sys.argv[1] == 'train1'):
         for i in range(ckpt_start,ckpt_start+num_iters):
            [train_loss, summary]= session.run([optim, merge], {inputs:train_inputs, joint_labels:train_labels,initial_joint_angles:train_initial})
            # [train_loss, summary] = session.run([loss, merge],
            # {inputs:train_inputs, joint_labels:train_labels,initial_joint_angles:train_initial})
            writer.add_summary(summary,i)
            if i%100 == 0:
                print('Loss after {} iterations == '.format(i), session.run(loss,{inputs:train_inputs, joint_labels:train_labels, initial_joint_angles:train_initial}))
                saved_ckpt = saver.save(session, os.path.join(model_ckpt_path,"weights{}.ckpt".format(i)))
         print('Total loss on test set is {}'.format(session.run(loss,{inputs:test_inputs, joint_labels:test_labels, initial_joint_angles:test_initial})))
        ''' 
        elif(sys.argv[1] == 'train2'):

            for i in range(ckpt_start,ckpt_start+num_iters):
            [train_loss, summary]= session.run([optim, merge], {inputs:train_inputs, joint_labels:train_labels,initial_joint_angles:train_initial})
            # [train_loss, summary] = session.run([loss, merge],
            # {inputs:train_inputs, joint_labels:train_labels,initial_joint_angles:train_initial})
            writer.add_summary(summary,i)
            if i%100 == 0:
                print('Loss after {} iterations == '.format(i), session.run(loss,{inputs:train_inputs, joint_labels:train_labels, initial_joint_angles:train_initial}))
                saved_ckpt = saver.save(session, os.path.join(model_ckpt_path,"weights{}.ckpt".format(i)))
         print('Total loss on test set is {}'.format(session.run(loss,{inputs:test_inputs, joint_labels:test_labels, initial_joint_angles:test_initial})))

        '''

        elif(sys.argv[1]=='test1'):
            end_point1 = get_poses(False,group_arm)
            waypoints=[]
            waypoints.append(end_point1)
            plan,fraction,end_point2 = random_valid_end_point_generator(waypoints,group_arm,end_point1)
            point_i = np.array([end_point1.position.x,end_point1.position.y,end_point1.position.z])
            point_f = np.array([end_point2.position.x,end_point2.position.y,end_point2.position.z])
            
            point_f = np.array([0.205304081987, 0.525954286467,0.957953280585])
            
            # print "point 1: "+str(point_i)
            # print "point 2  "+str(point_f)
            # dir_vector = point_f-point_i
            # way=[]
            # for t in np.arange(0,1,0.001):
            #     way.append(point_i+t*dir_vector)
            distance = 0
            path_length = np.linalg.norm(point_i-point_f)
            print "path length" + str(path_length)
            # tolerance = 0.1*path_length
            #run_input_data = np.expand_dims([np.append(way[-1],i) for i in way],0)
            run_input_data = ouputs
            initial_states = np.expand_dims(get_poses(True,group_arm),0)
            t=np.zeros([num_ce  ls,2,1,num_features])
            t[0,0,:,:] = initial_states
            initial_states = t
            #while path_length>tolerance:
            pred=session.run(
                    prediction,{
                        inputs:run_input_data,
                        initial_joint_angles:initial_states
                    }
                )
            #print pred
            group_arm.clear_pose_targets()
            for i in pred[0]:
                ee = call_compute_fk_service(i)
                ee=[ee.x,ee.y,ee.z]
                print "distance"+str(np.linalg.norm(ee-point_f)) 
                group_arm.set_joint_value_target(i.tolist())
                group_arm.plan()
                group_arm.go()
                group_arm.clear_pose_targets()
            now = get_poses(False,group_arm)
            now=[now.position.x,now.position.y,now.position.z]
            print "current position:" + str(now)
            print "required target :" + str(point_f)
            print "last waypoint   :" + str(way[-1])
            print "final distance  : "+ str(np.linalg.norm(now-point_f))
            print "initial distance: "+ str(path_length)
                
            
            moveit_commander.os._exit(0)    
        
        elif sys.argv[1]=='test2':
            end_point1 = get_poses(False,group_arm)
            waypoints=[]
            waypoints.append(end_point1)
            plan,fraction,end_point2 = random_valid_end_point_generator(waypoints,group_arm,end_point1)
            point_i = np.array([end_point1.position.x,end_point1.position.y,end_point1.position.z])
            point_f = np.array([end_point2.position.x,end_point2.position.y,end_point2.position.z])
            
            f=np.array([0.57646,0.20584,0.58239])
            point_f = np.array(f)
            print "point 1: "+str(point_i)
            print "point 2  "+str(point_f)
            
            distance = 0
            path_length = np.linalg.norm(point_i-point_f)
            print "path length" + str(path_length)
            tolerance = 0.1*path_length
            
            run_input_data = np.expand_dims([np.append(point_f,point_i)],0)
            initial_states = np.expand_dims(get_poses(True,group_arm),0)
            
            t=np.zeros([num_cells,2,1,num_features])
            t[0,0,:,:] = initial_states
            initial_states = t            
            
            ee=[]
            
            group_arm.clear_pose_targets()
            while(path_length>tolerance):
                print "~~~~~~~~~~~~~~~~"
                print "run_input_data"+str(run_input_data)
                [fin,pred]=session.run(
                     [final_state,prediction],{
                        inputs:run_input_data,
                        initial_joint_angles:initial_states
                    }
                )

                #print "type" + str(type(fin))
                #print "fin shape:"+str(np.shape(fin))
                #print "fin" + str(fin)
                pred=np.squeeze(pred)
                
                print "pred" +str(pred)
                
                group_arm.set_joint_value_target(pred.tolist())
                group_arm.plan()
                group_arm.go()
                group_arm.clear_pose_targets()

                ee=call_compute_fk_servicef (pred)
                ee=np.array([ee.x,ee.y,ee.z])
                print "ee:"+str(ee)
                initial_state=fin
                run_input_data = np.expand_dims(np.expand_dims(np.append(f,ee),0),0)
                print "distance"+str(np.linalg.norm(ee-point_f)) 
                print "~~~~~~~~~~~~~~~~~"
                

            moveit_commander.os._exit(0)    
