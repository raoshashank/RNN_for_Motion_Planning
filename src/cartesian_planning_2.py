#! /usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import geometry_msgs.msg
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.srv import GetPositionFKRequest
from moveit_msgs.srv import GetPositionFKResponse
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
from std_msgs.msg import Header
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

#%matplotlib inline   

def call_compute_fk_service(joints):
    rospy.wait_for_service('compute_fk')
    try:
        moveit_fk = rospy.ServiceProxy('compute_fk',GetPositionFK)
        fk_link=['ee_link']
        jointNames = ['elbow_joint', 'shoulder_lift_joint', 'shoulder_pan_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        jointPositions = joints
        header=Header(0,rospy.Time.now(),"/world")
        rs = RobotState()
        rs.joint_state.name = jointNames
        rs.joint_state.position = jointPositions
        ee = moveit_fk(header,fk_link,rs)
        return  ee.pose_stamped[0].pose.position 
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
        


moveit_commander.roscpp_initializer.roscpp_initialize(sys.argv)
rospy.init_node('cartesian_planning_node', anonymous=True)
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
group_arm = moveit_commander.MoveGroupCommander("manipulator")
group_arm.set_planner_id("RRTConnectkConfig")
group_arm.clear_pose_targets()
waypoints=[]
#call_compute_fk_service()
#print group_arm.get_current_pose()
# print group_arm.get_current_joint_values()
intial_pose = group_arm.get_current_pose().pose
waypoints.append(intial_pose)
wpose = geometry_msgs.msg.Pose()
##0.673728
##0.186072
##0.876376

wpose.position.x = 0.60331
wpose.position.y = 0.42961
wpose.position.z = 0.70924

wpose.orientation.x = group_arm.get_current_pose().pose.orientation.x
wpose.orientation.y = group_arm.get_current_pose().pose.orientation.y
wpose.orientation.z = group_arm.get_current_pose().pose.orientation.z
wpose.orientation.w = group_arm.get_current_pose().pose.orientation.w
waypoints.append(copy.deepcopy(wpose))
# (plan, fraction) = group_arm.compute_cartesian_path(waypoints, 0.01, 0.0)
group_arm.clear_pose_targets()
group_arm.set_pose_target(wpose)

group_arm.set_goal_orientation_tolerance(0.01)
group_arm.set_goal_position_tolerance(0.01)

plan1=group_arm.plan()
print "GOING"
rospy.sleep(5)
group_arm.go(wait=True)
#group_arm.execute(plan,wait=True)
print "GONE"

# ee_way=[]
# for i in plan.joint_trajectory.points:
#     j=call_compute_fk_service(i.positions)
#     ee_way.append([j.x,j.y,j.z]) 



# fig = plt.figure()
# ax = plt.axes(projection='3d')
# zdata = [i[2] for i in ee_way]
# xdata = [i[0] for i in ee_way]
# ydata = [i[1] for i in ee_way]

# print "plotting"
# ax.scatter3D(xdata, ydata, zdata, cmap='Greens')
# plt.show()

#for i in plan.joint_trajectory.points:
#    print str(i.time_from_start.secs)+"."+str(i.time_from_start.nsecs)
# joint_state = GetPositionFK()
# joint_state.robot = robot
# fk_link_names = group_arm.get_joints()
#wpose2=group_arm.get_random_pose()

# fraction=0
# while(fraction!=1):
#     try:
#         del waypoints[1]
#     except IndexError:
#         print "not yet"
#     random_pose = group_arm.get_random_pose().pose
#     random_pose.orientation = intial_pose.orientation
#     waypoints.append(random_pose)
#     (plan, fraction) = group_arm.compute_cartesian_path(waypoints, 0.01, 0.0)

# print random_pose
# print "~~"
# print intial_pose
# print fraction
#joint_state.fk_link_names = 


#print plan
# print "Execution"
'''
group_arm.plan()
print "GOING BRO"
rospy.sleep(5)
group_arm.go(wait=True)
#rospy.sleep(5)
group_arm.execute(plan,wait=True)
print "GONE BRO"
'''
#rospy.sleep(5)
#group_arm.plan()
# # print waypoints

moveit_commander.roscpp_initializer.roscpp_shutdown()
moveit_commander.os._exit(0)

