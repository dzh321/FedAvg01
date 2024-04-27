import pickle

import numpy as np


x=[i for i in range(5)]
y=[2,3,4,5,7]
z=np.vstack((x,y))
print(z)
print(z.shape) # 2*5

zh=np.hstack((x,y))
print(zh)
print(zh.shape)
print("================================================================")
idxs=[1,3,6,8,3,5,9,2,1,0,4,6,8,2,6,9,42,56,13,647,2676,2356,2346,456]
num_imgs=2
dict_users = {i: np.array([]) for i in range(5)}
for i in range(5):
    for rand in range(2):
        dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
print(dict_users)

print("============================")
x=np.array([1,2,3,4,5])
print(x-1)


setX={1:"123",2:"3.2",3:"desf",4:"dfg",5:"fesrdtf"}
print(setX.get(3))

print("====================================================")
import copy
import torch

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

print(")))))))))))))))))))))))))))))))")

import torch

# 输出 CUDA 设备数量
num_cuda_devices = torch.cuda.device_count()
print(f"系统中有 {num_cuda_devices} 个 CUDA 设备可用。")

# 输出每个 CUDA 设备的索引和名称
for i in range(num_cuda_devices):
    print(f"GPU 索引 {i}: {torch.cuda.get_device_name(i)}")
print("0000000000000000000000000000000000000000000000")

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import os
# epoch_loss=[1,2,3,4,5]
# plt.figure()
# plt.plot(range(len(epoch_loss)), epoch_loss)
# plt.xlabel('epochs')
# plt.ylabel('Train loss')
# plt.savefig('../save/nn_{}_{}_{}.png'.format("ceshi", "ceshi", "1"))

file_name = '../save/objects/myceshi123321.pkl'
os.makedirs(os.path.dirname(file_name), exist_ok=True)
with open(file_name, 'wb') as f:
    pickle.dump([1, 2], f)  #







































