
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
    return [ 0.673728,0.186072,0.876376]        

def main():
    global group_arm
    num_examples = 1
    for i in range(num_examples):
            waypoints=[]
            initial_ee_pose = get_poses()
            initial_joint_angles = get_poses(True)
            
            waypoints.append(initial_ee_pose)

            final_ee_pose = geometry_msgs.msg.Pose()
            random_end_point = random_valid_end_point_generator()
            final_ee_pose.position.x = random_end_point[0]
            final_ee_pose.position.y = random_end_point[1]
            final_ee_pose.position.z = random_end_point[2]
            ##Apparently orientation remains same idk why
            final_ee_pose.orientation.x = group_arm.get_current_pose().pose.orientation.x
            final_ee_pose.orientation.y = group_arm.get_current_pose().pose.orientation.y
            final_ee_pose.orientation.z = group_arm.get_current_pose().pose.orientation.z
            final_ee_pose.orientation.w = group_arm.get_current_pose().pose.orientation.w
            
            waypoints.append(copy.deepcopy(final_ee_pose))

            (plan, fraction) = group_arm.compute_cartesian_path(waypoints, 0.01, 0.0)
            print "fraction: "+str(fraction)
            ##Check for successful plan
            if(fraction>0.8):
                #get joint values:
                path_length = len(plan.joint_trajectory.points)
                intermediate_joint_values = [plan.joint_trajectory.points[i].positions for i in range(path_length)]
                print "~~~~~~~~~~~~~joint plan~~~~~~~~~~~~~~~~~~~"
                print len(intermediate_joint_values)
                intermediate_ee_poses = []
                print "~~~~~~~~~~~~~ee plan~~~~~~~~~~~~~~~~~~~"
                [intermediate_ee_poses.append(call_compute_fk_service(joints)) for joints in intermediate_joint_values]
                print len(intermediate_ee_poses)
                ##now i have all path-joint and ee values. Time to format data and store it.
                


                    
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
    

