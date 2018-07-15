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
from data_generation import random_valid_end_point_generator,get_poses,call_compute_fk_service
import matplotlib.pyplot as plt
import pickle as pkl



prime="/home/shashank/catkin_ws/src/rnn_ur5/"
data_home=prime+"dataset/"
file_name = data_home + 'sequence_%05d'
model_ckpt_path = prime+"modelckpt/"
model_events = prime+"modelevents/"
from sklearn.utils import shuffle

path, dirs, files = next(os.walk(data_home))
num_sequences = len(files)
#print num_sequences

batch_size=64
num_features = 6 ##[desired_pose,current_pose]
output_sequence_length = 6 ##6 joint angles
num_units = 6
samples_per_path=100

inputs=[]
labels=[]
initial_states=[]
temp_inputs=[]

inp=[]
out=[]

data=Sequence()

for i in range(num_sequences):
    data.Clear()
    f=open(file_name%i,"rb")
    data.ParseFromString(f.read())
    for i in range(len(data.xyz)-1):
        x=[]
        y=[]
        [x.append(j) for j in data.xyz[i].poses]
        [x.append(j) for j in data.xyz[i+1].poses]
        [x.append(j/np.pi) for j in data.thetas[i].theta_values]
        [y.append(j) for j in data.thetas[i+1].theta_values]
        inp.append(x)
        out.append(y)

    f.close()

inp=np.array(inp)
out=np.array(out)
out=out/np.pi

ckpt_start=29  

# for i in np.arange(0,len(out),1):
# 	print '~~~~~~'
# 	print out[i]
# 	print '\n\n'
# 	print inp[i]
# 	print '~~~~~~'
# # In[89]:


import tensorflow as tf


# In[90]:


input_dimension=12
output_dimension=6

inputs = tf.placeholder(shape=[None,input_dimension],dtype=tf.float32,name='INPUT')
labels = tf.placeholder(shape=[None,output_dimension],dtype=tf.float32,name='OUTPUT')

print "\n\n\n\n"
print np.shape(inp)
print np.shape(out)

units = [[30,50,50,32,12,6]]#,[50,64,12,6],[100,50,12,6]]
# In[92]:
arch=0

with tf.name_scope('Dense_1'):
    fc1 = tf.layers.dense(inputs=inputs,units=units[arch][0],use_bias=False,activation=tf.nn.relu)

with tf.name_scope('Dense_2'):
    fc2=tf.layers.dense(inputs=fc1,units=units[arch][1],use_bias=False,activation=tf.nn.relu)

with tf.name_scope('Dense_3'):  
    fc3=tf.layers.dense(inputs=fc2,units=units[arch][2],use_bias=False,activation=tf.nn.relu)    

with tf.name_scope('Dense_4'):  
    fc4=tf.layers.dense(inputs=fc3,units=units[arch][3],use_bias=False,activation=tf.nn.relu)    

with tf.name_scope('Dense_5'):  
    fc5=tf.layers.dense(inputs=fc4,units=units[arch][4],use_bias=False,activation=tf.nn.relu)    

with tf.name_scope('Output'):
    output=tf.layers.dense(inputs=fc5,units=units[arch][5],use_bias=False,activation=tf.nn.tanh)

with tf.name_scope('LOSS'):
    loss=tf.reduce_sum(tf.losses.mean_squared_error(labels=labels,predictions=output))
    
    
    
optim = tf.train.AdamOptimizer(name="Optimizer",learning_rate=1e-5).minimize(loss)

loss_summary = tf.summary.scalar('loss',loss)
saver = tf.train.Saver()

# In[96]:
write = 0
leng=len(inp)
train_size = int(0.8*leng)
train_x = inp[0:train_size,:]
train_y = out[0:train_size,:]
test_x  = inp[train_size:,:]
test_y  = out[train_size:,:]

if __name__ == '__main__':
    mode=sys.argv[1]
    global group_arm,current_joint_values
    current_joint_values = JointState()
    moveit_commander.roscpp_initializer.roscpp_initialize(sys.argv)
    rospy.init_node('main_node', anonymous=True)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_arm = moveit_commander.MoveGroupCommander("manipulator")
    group_arm.set_planner_id("RRTConnectkConfig")
    
    with tf.Session() as sess:
    
        batch_size=200
        epochs = 10
        iterations = len(inp)/batch_size

        print("\n\n")
        if len(os.listdir(model_ckpt_path)) == 0:
            init = tf.global_variables_initializer()
            sess.run(init)
            print "new"
        else:
            saver.restore(sess, os.path.join(model_ckpt_path, "weights{}.ckpt".format(ckpt_start)))
            print "restored"
       
        if mode=='train':
            merge=tf.summary.merge_all()
            writer = tf.summary.FileWriter(model_events)
            writer.add_graph(sess.graph)

            for e in range(epochs):
                
                x,y = shuffle(train_x,train_y,random_state=0)

                input_=x
                output_=y

                input_ = np.array_split(input_,iterations)
                output_ = np.array_split(output_,iterations)
                num_batches = len(input_)

                for j in range(num_batches):
                    write+=1
                    [_,train_loss,summary]=sess.run([optim,loss,merge],{inputs:input_[j],labels:output_[j]})
                    if j%100 == 0:
                        [train_loss]=sess.run([loss],{
            	        inputs:input_[j],
            	        labels:output_[j]})
            	        print "epoch :"+str(e)+"   iteration:"+str(j)+"       loss: "+str(train_loss)
                        writer.add_summary(summary,write)
                saved_ckpt = saver.save(sess, os.path.join(model_ckpt_path, "weights{}.ckpt".format(e)))
            
            print "test_loss: " + str(sess.run([loss],{
                inputs:test_x,
                labels:test_y}))
        
        if mode=='test':
            waypoints=[]
            initial_ee_pose = get_poses(False,group_arm)
            initial_joint_angles = get_poses(True,group_arm)
            
            waypoints.append(initial_ee_pose)

            final_ee_pose = geometry_msgs.msg.Pose()
            final_ee_pose.position.x =-0.79074
            final_ee_pose.position.y =  0.29145
            final_ee_pose.position.z =  0.40617
            

            final_ee_pose.orientation.x = group_arm.get_current_pose().pose.orientation.x
            final_ee_pose.orientation.y = group_arm.get_current_pose().pose.orientation.y
            final_ee_pose.orientation.z = group_arm.get_current_pose().pose.orientation.z
            final_ee_pose.orientation.w = group_arm.get_current_pose().pose.orientation.w
            waypoints.append(copy.deepcopy(final_ee_pose))
            
            (plan, fraction) = group_arm.compute_cartesian_path(waypoints, 0.01, 0.0)
            group_arm.plan()
            intermediate_ee_poses = []

            print "end point: " + str(final_ee_pose.position)

            path_length = len(plan.joint_trajectory.points)
            intermediate_joint_values = [plan.joint_trajectory.points[j].positions for j in range(path_length)]
            [intermediate_ee_poses.append(call_compute_fk_service(joints)) for joints in intermediate_joint_values]
            print np.shape(intermediate_ee_poses)
            intermediate_ee_poses=np.array(intermediate_ee_poses)

            intermediate_ee_poses_=[]
            for i in intermediate_ee_poses:
                intermediate_ee_poses_.append([i.x,i.y,i.z])
            intermediate_ee_poses=np.array(intermediate_ee_poses_)

            print np.shape(intermediate_ee_poses)

            intermediate_ee_poses_1  = np.roll(intermediate_ee_poses,-1,0)
            print np.shape(intermediate_ee_poses_1)

            intermediate_joint_values = np.array(intermediate_joint_values)/np.pi
            intermediate_joint_values_1 = np.roll(intermediate_joint_values,-1,0)

            intermediate_ee_poses=intermediate_ee_poses[:-1,:]
            intermediate_ee_poses_1=intermediate_ee_poses_1[:-1,:]
            intermediate_joint_values=intermediate_joint_values[:-1,:]
            intermediate_joint_values_1=intermediate_joint_values_1[:-1,:]

            print np.shape(intermediate_ee_poses)
            print np.shape(intermediate_ee_poses_1)
            print np.shape(intermediate_joint_values)
            print np.shape(intermediate_joint_values_1)


            inputs_ = np.hstack([intermediate_ee_poses,intermediate_ee_poses_1,intermediate_joint_values])
            outputs_ = intermediate_joint_values_1

            # vidia



            print np.shape(inputs_)
            print np.shape(outputs_)

            [theta_hat,loss_] = sess.run([output,loss],feed_dict={inputs:inputs_,labels:outputs_})
            # print "network-output:" +str(theta_hat)
            print "loss : " +str(loss_)
            print "output shape:"+str(np.shape(theta_hat))
            print type(theta_hat)
            for i in theta_hat:
                group_arm.clear_pose_targets()
                i=i*np.pi
                #i[0]=0
                group_arm.set_joint_value_target(i.tolist())
                group_arm.plan()
                group_arm.go()
                rospy.sleep(0.1)
                #print "moved"
            
            print group_arm.get_current_pose().pose.position
            file_ = open('theta_hat.pkl','wb')
            pkl.dump(theta_hat,file_)

            file_.close()            
            moveit_commander.os._exit(0)    
            