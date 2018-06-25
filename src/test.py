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

moveit_commander.roscpp_initializer.roscpp_initialize(sys.argv)
rospy.init_node('cartesian_planning_node', anonymous=True)
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
group_arm = moveit_commander.MoveGroupCommander("manipulator")
group_arm.set_planner_id("RRTConnectkConfig")
group_arm.clear_pose_targets()
waypoints=[]

intial_pose = group_arm.get_current_pose().pose
waypoints.append(intial_pose)


wpose.position.x =  -0.516393634674
wpose.position.y =  0.355826552764
wpose.position.z = 0.176156007518
wpose.orientation.x = 0.0401904438092
wpose.orientation.y = 0.983963015316
wpose.orientation.z = -0.172781621015
wpose.orientation.w = 0.0186554055213
waypoints.append(copy.deepcopy(wpose))
(plan, fraction) = group_arm.compute_cartesian_path(waypoints, 0.01, 0.0)

