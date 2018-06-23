#!/usr/bin/env python

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
import math
import random

def deg_to_rad(angle):
    a = (math.pi * angle)/180
    return a


if __name__ == "__main__":
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('planning_and_execution', anonymous=True)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_arm = moveit_commander.MoveGroupCommander("manipulator")
    while not rospy.is_shutdown():
        
        #print "Robot Pose == ", group_arm.get_current_pose()
        #print "Joint Angles: ",group_arm.get_current_joint_values()
        print group_arm.get_current_pose().pose.position.x
        print group_arm.get_current_joint_values()
        print "~~~~~~~~~"

        
moveit_commander.roscpp_shutdown()
moveit_commander.os._exit(0)
