import os

dir_path = "/home/shashank/Desktop/bag_dirs"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

idx = 1
while idx < 10:
file_path = os.path.join(dir_path, "file_name{}".format(idx))