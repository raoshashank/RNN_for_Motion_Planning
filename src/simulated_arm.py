#! /usr/bin/env python

import rospy
import actionlib
import random
import math
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint


def deg_to_rad(angle):
    a = (math.pi * angle)/180
    return a


def joint_commands():
    while not rospy.is_shutdown():
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.header.stamp = rospy.Time.now()
        goal.trajectory.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint" , "wrist_1_joint", "wrist_2_joint",
                                       "wrist_3_joint"]
        points = JointTrajectoryPoint()
        points.positions = [random.uniform(deg_to_rad(0), deg_to_rad(180)), random.uniform(deg_to_rad(5), deg_to_rad(42)),
                            random.uniform(deg_to_rad(0), deg_to_rad(96)), random.uniform(deg_to_rad(15.72), deg_to_rad(20)),
                            deg_to_rad(90), random.uniform(0, deg_to_rad(180))]
        points.velocities = [2, 2, 2, 2, 2, 2]

        points.time_from_start = rospy.Duration(2)

        goal.trajectory.points.append(points)
        client.send_goal_and_wait(goal, execute_timeout=rospy.Duration(5))
        rospy.loginfo("Commands are being Sent to Server")
        rospy.sleep(1)


if __name__ == "__main__":
    rospy.init_node('arm_controller_client_node')
    client = actionlib.SimpleActionClient("/arm_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
    client.wait_for_server()
    joint_commands() 

