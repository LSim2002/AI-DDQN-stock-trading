# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 13:53:08 2024

@author: loulo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import numpy as np
import gym
import random
from collections import deque

from environmentLouis2 import env

import random


class ConvDQN(nn.Module):   ##creation DU MODELE!   heritage de nn.Module
    
    def __init__(self):  #constructeur du modele
        super(ConvDQN, self).__init__() ## calls the constructor of the parent class nn.Module. It's necessary to initialize the ConvDQN object properly, as it inherits functionality from nn.Module.
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate the input size for the fully connected layer
        self.fc_input_size = self._get_fc_input_size()        
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def _get_fc_input_size(self):
        # Dummy input to calculate the output size of the convolutional layers
        x = torch.randn(1, 1, 84, 84)
        x = self.conv_layers(x)
        return x.view(-1).size(0)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.fc_input_size)
        x = self.fc_layers(x)
        return x
    
    
    
    
    
    
env = env('ADBE','2013-01-01','2020-01-04')
state = env.getCurrState() 
state = torch.FloatTensor(state)
#print(state.shape) #returns (1, 84, 84)
test=ConvDQN()    
qvals=test.forward(state)
print(qvals)    
    
    