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
    rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory)

    while not rospy.is_shutdown():
        print " ********* Generating Pick Plan ********* \n"
        joint_variables = [random.uniform(deg_to_rad(-25), deg_to_rad(8)), random.uniform(deg_to_rad(-45), deg_to_rad(-12)),
                           random.uniform(deg_to_rad(68), deg_to_rad(96)), random.uniform(deg_to_rad(15.72), deg_to_rad(20)),
                           deg_to_rad(90), random.uniform(0, deg_to_rad(180))]
        group_arm.set_joint_value_target(joint_variables)
        print joint_variables
        group_arm.plan()
        group_arm.go()
        rospy.sleep(0.1)
        print "\n ********* Gripper Closed ********* \n"
        rospy.sleep(0.1)
        print "Robot Pose == ", group_arm.get_current_pose()
        print "Robot RPY == ", group_arm.get_current_rpy()
        print "Robot Current values == ", group_arm.get_joint_value_target()
        print "** Generating Place Plan **"
        joint_variables = [random.uniform(deg_to_rad(-25), deg_to_rad(8)), random.uniform(deg_to_rad(-45), deg_to_rad(-12)),
                           random.uniform(deg_to_rad(68), deg_to_rad(96)), random.uniform(deg_to_rad(15.72), deg_to_rad(20)),
                           deg_to_rad(90), random.uniform(0, deg_to_rad(180))]
        group_arm.set_joint_value_target(joint_variables)
        group_arm_variable_values = group_arm.set_joint_value_target(joint_variables)
        group_arm.plan()
        group_arm.go()
        print " \n ** Planning Done ** "
        rospy.sleep(5)

moveit_commander.roscpp_shutdown()
moveit_commander.os._exit(0)
