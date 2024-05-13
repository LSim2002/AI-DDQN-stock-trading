# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 23:11:03 2024

@author: loulo
"""
import os
import cv2
from data_preparation import generate_candlestick_image

    
class env:
    def __init__(self, ticker):
        
        ##ticker is given as a string input 

        self.dir = f'./candlestick_images/{ticker}/'  #string
        
        self.curr_state_id = 0 ##initialized to zero for initial state
        self.curr_state_dir = self.dir + f'./{self.curr_state_id}.png'
        self.curr_state_image = cv2.imread(self.curr_state_dir)  ##curr state as a python image object? taken directly from curr_state_dir

#        self.curr_state_matrix = None ##curr state as a numpy matrix object? used directly by the model?

        self.data_size = len(os.listdir(self.dir))  ##number of training steps
        self.actionspace = [1, -1, 0]  # Long, Short, No position


    def getState(self):
        
        return self.curr_state_matrix

    def reset(self):
        self.curr_state_id = 0

    def step(self):##returns reward and sets state_id to state_id+1

        #action does not influence next step, the next state is simply the next image in the directory.
        self.curr_state_id += 1

        # Define reward (not specified in the outline)
        reward = 0  # Placeholder reward, needs to be defined based on your trading strategy

        return reward
    
    