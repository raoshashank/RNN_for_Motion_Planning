import tensorflow as tf 
import tensorflow.contrib.rnn as rnn
import numpy as np
import os,os.path
from data_pb2 import Sequence
from prep_data import prep_data2
import sys
from data_generation import call_compute_fk_service,random_valid_end_point_generator,get_poses
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
import sys
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




