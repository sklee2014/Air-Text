import matplotlib.pyplot as plt
import numpy as np
import glob
import os


save_dir = './images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
file_list = glob.glob('./data/*/*.txt')

length = len(file_list)
ct = 1

for filename in file_list:
    temp = filename.split('/')
    word_num = temp[2]
    word_name = temp[3].split('_')[0]
    f = open(filename, 'r')
    print(f.readline())
    trajs = np.loadtxt(f, dtype = 'float32')
    f.close()
    plt.style.use('dark_background')
    fig = plt.figure()
    plt.axis('off')
    plt.plot(trajs[:,0], -trajs[:, 1], 'w', linewidth=10)
    plt.savefig(save_dir + '/' + word_name + '_' + word_num + '.png')
    print(str(ct) + '/' + str(length) + 'done')
    ct += 1
    plt.close(fig)