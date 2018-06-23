from data_pb2 import Sequence
# test = Sequence()
# ##writing to intial and final pose
# test.ends.initial_pose.extend([0,0,0])
# test.ends.final_pose.extend([1,1,1])
# ##writing to thetas
# test.thetas.add().theta_values.extend([2,2,2,2,2,2])
# test.thetas.add().theta_values.extend([2.1,2.1,2.1,2.1,2.1,2.1])
# test.thetas.add().theta_values.extend([2.2,2.2,2.2,2.2,2.2,2.2])
# ##writing to xyz
# test.xyz.add().poses.extend([3,3,3,3,3,3])
# test.xyz.add().poses.extend([3.1,3.1,3.1,3.1,3.1,3.1])
# test.xyz.add().poses.extend([3.2,3.2,3.2,3.2,3.2,3.2])
# print test
# ##Write to file
# f=open("pb2_tutorial","wb")
# f.write(test.SerializeToString(test))
# f.close()
# ##Reading from file
# test.Clear()
home="/home/shashank/catkin_ws/src/rnn_ur5/dataset/"
file_name = home + 'sequence_%05d'
test = Sequence()
home="/home/shashank/catkin_ws/src/rnn_ur5/dataset/"
file_name = home + 'sequence_%05d'

path, dirs, files = next(os.walk(home))
num_sequences = len(files
for f in files:
    test.Clear()
    try:
        f = open(home+f, "rb")
        test.ParseFromString(f.read())
        print test.ends.initial_pose
        print test.ends.final_pose
        
        f.close()
    except IOError:
        print "Couldnt open file"

##Accessing elements :

# print "~~~~~~~~~~~~~"
# print "test.xyz[0] = "+str(test.xyz[0])
# print "test.ends.initial_pose = " + str(test.ends.initial_pose)
# print "test.thetas[0]" + str(test.thetas[0])
# print "test.thetas[0].theta_values[0]" + str(test.thetas[0].theta_values[0])
# print "~~~~~~~~~~~~~"

