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

inputs=[]
labels=[]
intial_states=[]
temp_inputs=[]

def prep_data():
    global inputs,labels,intial_states,temp_inputs
    max_len = 0
    data=Sequence()
    for i in range(num_sequences):
        data.Clear()
        thetas=[]
        xyz=[] #shape : sequence_length *   
        xyz_f=[]
        data.Clear()
        f=open(str(file_name%i),"rb")
        data.ParseFromString(f.read())
        #if len(data.thetas)
        #Prep input values
        xyz_f = data.ends.final_pose
        for t in data.xyz:
            j=[]
            j.append([r for r in t.poses])
            xyz.append(j)
        
        j=0;t=0
        xyz=np.array(xyz)
        #xyz=np.squeeze(xyz) 
        th=[]
        for t in data.thetas:
            th.append([j for j in t.theta_values])
        ##for now lets not use the ends attribute and treat last and first elements of thetas and xyz as ends
        [intial_states.append([x for x in data.thetas[-1].theta_values])]
        #[labels.append([[j for j in t.theta_values] for t in data.thetas])]
        
        j=0;t=0;temp_inputs=[]
        [temp_inputs.append(np.array([j for j in t.poses])) for t in data.xyz]                                                          
        t=[]
        [t.append(np.append(r,temp_inputs[-1])) for r in temp_inputs]
        t=np.expand_dims(t,0).tolist()
        inputs = t+inputs
        l=[]
        temp_labels=[]
        [temp_labels.append(np.array([j for j in t.theta_values])) for t in data.thetas]                                                         
        temp_labels=np.expand_dims(temp_labels,0).tolist()
        labels = temp_labels+labels
    max_len=0
    
    ##padding
    for i in inputs:
        if(max_len<np.shape(i)[0]):
            max_len = np.shape(i)[0]
    inputs=[np.pad(i,[(0,max_len-np.shape(i)[0]),(0,0)],mode='constant') for i in inputs]
    labels=[np.pad(i,[(0,max_len-np.shape(i)[0]),(0,0)],mode='constant') for i in labels]
                
# prep_data()
# # t=[np.shape(i) for i in inputs]        
# # print t
# # t=[np.shape(i) for i in labels]        
# # print t
# print np.shape(initial_states)
# print
