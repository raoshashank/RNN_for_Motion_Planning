import tensorflow as tf 
import tensorflow.contrib.rnn as rnn
import numpy as np
import os,os.path
from data_pb2 import Sequence
home="/home/shashank/catkin_ws/src/rnn_ur5/dataset/"
file_name = home + 'sequence_%05d'

path, dirs, files = next(os.walk(home))
num_sequences = len(files)
#print num_sequences

batch_size=64
num_features = 6 ##[desired_pose,current_pose]
output_sequence_length = 6 ##6 joint angles
num_units = 6
samples_per_path=100

def prep_data():
    data=Sequence()
    for i in range(num_sequences):
        data.Clear()    
        thetas=[]
        xyz=[] #shape : sequence_length *   
        xyz_f=[]
        data.Clear()
        f=open(str(file_name%i),"rb")
        data.ParseFromString(f.read())
        
        #Prep input values
        xyz_f = data.ends.final_pose
        for i in data.xyz:
            j=[]
            j.append([r for r in i.poses])
            xyz.append(j)
        xyz=np.array(xyz)
        xyz=np.squeeze(xyz)
        print np.shape(xyz) 
        print xyz  
         

prep_data()
        
