import numpy as np
import itertools
import time
import datetime

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import make_grid
import torch.nn.functional as F
import torch
from matplotlib.pyplot import figure
from IPython.display import clear_output

from PIL import Image

from utils import *
from cyclegan import *

cuda = True if torch.cuda.is_available() else False
print("Using CUDA" if cuda else "Not using CUDA")

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


###############################################
# Defining all hyperparameters
###############################################
class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

hp = Hyperparameters(
    epoch = 0,
    n_epochs = 200,
    dataset_train_mode = "train",
    dataset_test_mode = "test",
    batch_size = 4,
    lr = 0.0002,
    decay_start_epoch = 100,
    b1 = 0.5,
    b2 = 0.999,
    n_cpu = 8,
    img_size = 128,
    channels =3,
    n_critic = 5,
    sample_interval = 100,
    num_residual_blocks = 19,
    lambda_cyc = 10.0,
    lambda_id = 5.0
)

###############################################
# Setting Root Path for Google Drive or Kaggle
###############################################

# Root Path for Google Drive

root_path = "/content/drive/MyDrive/All_Datasets/summer2winter_yosemite"
