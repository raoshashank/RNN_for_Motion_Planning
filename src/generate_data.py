`=============-import rospy
from sensor_msgs.msg import JointState
import rosbag
import sys
from sensor_msgs.msg import JointState
sys.path.append("~/catkin_ws/src/rnn_ur5/src")
'''
In order to generate robot data, we have to record joint values and corresponding end effector positions
This can be done through ros bag files. 
Ros bag files will be generated when random trajectories are executed on the moveit plenner and the joint states and 
given end effector pose is recorded.

Topic to record joint angle values : /joint_states
To find end effector pose : 
'''
import csv
js = []
bag = rosbag.Bag("2018-06-13-21-24-54.bag")
for (topic, msg, t) in bag.read_messages():
    js.append(msg.position)

print js[0]



