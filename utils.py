import os
import numpy as np

import torchvision.transforms as transforms
from torch.autograd import Variable

import torch as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import glob
import random
from torch.utils.data import Dataset
from PIL import Image

###############################################
# Methods for Image DataLoader
###############################################

def convert_to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%sA" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))
        # print("self.files_B", self.files_B)
        """
        Will print below array with all file names
        ['/content/drive/MyDrive/All_Datasets/summer2winter_yosemite/trainB/2005-06-26 14:04:52.hpg',
        ]
        """

        def __getitem__(self,index):
