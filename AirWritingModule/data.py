import os
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import random
import cv2


label2map_dict = {'SingleOne' : [1], 'SingleTwo' : [1, 2], 'SingleThree' : [1, 2, 3],
                  'SingleFour' : [1, 2, 3, 4], 'SingleFive' : [0, 1, 2, 3, 4],
                  'SingleSix' : [0, 4], 'SingleSeven' : [0, 1, 4], 'SingleEight' : [0, 1],
                  'SingleNine' : [1], 'SingleGood' : [0], 'SingleBad' : [0]}
label_order = ['SingleOne', 'SingleTwo', 'SingleThree', 'SingleFour', 'SingleFive',
               'SingleSix', 'SingleSeven', 'SingleEight', 'SingleNine', 'SingleGood', 'SingleBad']


# Image normalization
def normalize_img(img):
    img = img.astype(np.float32) / 255.0
    img -= img.mean()
    return img


# Gaussian2d calculation
def gaussian2d(sigma=1.5, coordinate=None):
    x, y = coordinate
    j, i = np.ogrid[0:60, 0:80]
    a = 2 * np.square(sigma)
    b = 1.0 / (a * np.pi)
    gaussian = np.exp(- (np.square(i - x) + np.square(j - y)) / a)
    return gaussian


# Fingertip heatmap generation with Gaussian2d
def encoder(gt, label_key):
    fingermap_idx = label2map_dict[label_key]
    fingermap_num = len(fingermap_idx)
    fingermap = np.zeros(shape=(5, 60, 80), dtype=np.float32)
    h, w = fingermap.shape[1:]
    for i in range(fingermap_num):
        x = w * float(gt[4 * i])
        y = h * float(gt[4 * i + 1])
        fingermap[fingermap_idx[i], :, :] = gaussian2d(coordinate=(x, y))
    return fingermap


def txt_to_list(txt_list):
    frame_list = []
    for txt in txt_list:
        f = open(txt, 'r')
        frame_list += f.readlines()

    for i, frame in enumerate(frame_list):
        frame_list[i] = frame.rstrip().split('    ')
    return frame_list


# Load image and normalize
def load_image(img_path, target_size=(160, 120)):
    image = cv2.imread(img_path)
    resize_image = cv2.resize(image, target_size)
    final_image = normalize_img(resize_image)
    return final_image


class FingerTipGestureDataset(Dataset):
    def __init__(self, data_path, start=0.0, end=1.0, n_class=8):
        self.data_path = data_path
        txt_list = glob.glob(os.path.join(self.data_path, 'label/*'))
        frame_list = txt_to_list(txt_list)
        random.Random(0).shuffle(frame_list)
        frame_len = len(frame_list)
        self.frames = frame_list[int(frame_len*start):int(frame_len*end)]
        self.n_class = n_class

    def __getitem__(self, index):
        frame = self.frames[index]
        img_temp = frame[0].split('/')
        label_key = img_temp[2]
        img_path = os.path.join(self.data_path, img_temp[2], img_temp[3])
        ftip = frame[5:]
        img = load_image(img_path, (160, 120))
        heatmap_label = encoder(ftip, label_key)
        class_label = label_order.index(label_key)

        return torch.from_numpy(img.transpose(2, 0, 1)).float(), heatmap_label, torch.tensor(class_label).long()

    def __len__(self):
        return len(self.frames)