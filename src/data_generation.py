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
'''
 This node:
 a. training:
     1. acquires initial,final,intermediate poses from moveit
     2. acquires Joint information from moveit
     3. queries rnn service with above values in training mode
     4. Displays losses after training
 
 b. testing:
     1. queries rnn service with initial and final poses
     2. acquires commands from service 
     3. executes acquired waypoint commands
 
 
 1. acquiring poses from moveit server:
     a. joint angles: 
         group_arm.get_current_joint_values()
     b. ee pose :
         group_arm.get_current_pose().pose...
 
 2. format expected for rnn input:
     [batch_size,sequence_length,num_features]
     [   1 batch looks like this 
         [
          [X_d,Y_d,Z_d,X_c,Y_c,Z_c]
          [X_d,Y_d,Z_d,X_c,Y_c,Z_c]
          [X_d,Y_d,Z_d,X_c,Y_c,Z_c]
          [X_d,Y_d,Z_d,X_c,Y_c,Z_c]
          [X_d,Y_d,Z_d,X_c,Y_c,Z_c]
         ]
 
         [
             [X_d,Y_d,Z_d,X_c,Y_c,Z_c]
             [X_d,Y_d,Z_d,X_c,Y_c,Z_c]
             [X_d,Y_d,Z_d,X_c,Y_c,Z_c]
             [X_d,Y_d,Z_d,X_c,Y_c,Z_c]
         ]
     ]
 3. format for rnn labels: 
     [batch_size,sequence_length,num_features]
     [   1 batch looks like this 
         [
          [J_0,J_1,J_2,J_3,J_4,J_5]
          [J_0,J_1,J_2,J_3,J_4,J_5]
          [J_0,J_1,J_2,J_3,J_4,J_5]
          [J_0,J_1,J_2,J_3,J_4,J_5]
          [J_0,J_1,J_2,J_3,J_4,J_5]
         ]
 
         [
          [J_0,J_1,J_2,J_3,J_4,J_5]
          [J_0,J_1,J_2,J_3,J_4,J_5]
          [J_0,J_1,J_2,J_3,J_4,J_5]
          [J_0,J_1,J_2,J_3,J_4,J_5]
          [J_0,J_1,J_2,J_3,J_4,J_5]
         ]
     ]
'''
home="/home/shashank/catkin_ws/src/rnn_ur5/dataset/"
file_name = home + 'sequence_%05d'
 
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





def get_poses(joint = False):
    global group_arm  
    if joint:
        value =group_arm.get_current_joint_values() 
    else:
        value = group_arm.get_current_pose().pose
    return  value

def random_valid_end_point_generator():
    global waypoints,group_arm
    fraction = 0 
    final_ee_pose=geometry_msgs.msg.Pose()
    while(fraction<0.8):
        try:
            del waypoints[1]
        except IndexError:
            print "not yet"
        final_ee_pose=group_arm.get_random_pose().pose
        waypoints.append(final_ee_pose)
        (plan, fraction) = group_arm.compute_cartesian_path(waypoints, 0.01, 0.0)
    return plan,fraction,final_ee_pose

def main():
    global group_arm,waypoints,plan,fraction,end_point
    num_examples = 100
    data = Sequence()
    valid_count =0
    for i in range(num_examples):
            data.Clear()
            waypoints=[]
            initial_ee_pose = get_poses()
            initial_joint_angles = get_poses(True)
            
            waypoints.append(initial_ee_pose)

            final_ee_pose = geometry_msgs.msg.Pose()
            random_end_point = random_valid_end_point_generator()

            (plan,fraction,final_ee_pose)=random_valid_end_point_generator()
            print "fraction: "+str(fraction)
            #get joint values:
            path_length = len(plan.joint_trajectory.points)
            intermediate_joint_values = [plan.joint_trajectory.points[j].positions for j in range(path_length)]
            print "~~~~~~~~~~~~~joint plan~~~~~~~~~~~~~~~~~~~"
            print len(intermediate_joint_values)
            intermediate_ee_poses = []
            print "~~~~~~~~~~~~~ee plan~~~~~~~~~~~~~~~~~~~"
            [intermediate_ee_poses.append(call_compute_fk_service(joints)) for joints in intermediate_joint_values]
            print len(intermediate_ee_poses)
            ##now i have all path-joint and ee values. Time to format data and store it.
            data.ends.initial_pose.extend([initial_ee_pose.position.x,initial_ee_pose.position.y,initial_ee_pose.position.z])
            data.ends.final_pose.extend([final_ee_pose.position.x,final_ee_pose.position.y,final_ee_pose.position.z])
            [data.thetas.add().theta_values.extend(joints) for joints in intermediate_joint_values] 
            [data.xyz.add().poses.extend([pos.x,pos.y,pos.z]) for pos in intermediate_ee_poses]
            print data.ends.initial_pose
            print data.ends.final_pose
            f = open(str(file_name%valid_count), "wb")
            f.write(data.SerializeToString(data))
            f.close()
            valid_count+=1
                    
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
    main()
    moveit_commander.os._exit(0)
    

