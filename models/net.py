from collections import OrderedDict
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import model_zoo
import copy
import numpy as np
from torchvision import utils
from .modules import *

class backbone(nn.Module):
    def __init__(self, Encoder, num_features, block_channel):
        super(backbone, self).__init__()
        self.E = Encoder
        self.MFF = MFF(block_channel)

class model_ll(nn.Module):
    def __init__(self, backbone, num_tasks, block_channel, is_training=False):
        super(model_ll, self).__init__()
        self.E = backbone.E
        self.MFF = backbone.MFF
                
        self.tasks = {}
        self.num_tasks = num_tasks

        for i in range(self.num_tasks):            
            if is_training:
                if((self.num_tasks > 1) and (i != (self.num_tasks-1))):
                    self.tasks[i] = backbone.tasks[i].cuda()
                else:
                    self.tasks[i] = Uncertainty_depth(block_channel).cuda()   
                self.add_module('task'+ str(i),self.tasks[i])
            else:
                self.tasks[i] = Uncertainty_depth(block_channel).cuda()   
                self.add_module('task'+ str(i),self.tasks[i])

        
    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)       
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4, [x_block1.size(2) * 2, x_block1.size(3) * 2])

        out = {}
        for i in range(self.num_tasks):      
            out[i] = self.tasks[i](x_mff)
         
        return out, x_mff
