# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 12:30:28 2024

@author: loulo
"""
import matplotlib.pyplot as plt
import yfinance as yf
import mplfinance as mpf
import numpy as np
import cv2
from data_preparation import generate_candlestick_image
import torch 

class env:
    def __init__(self, ticker, start_date, end_date):

        self.ticker = ticker #string 
        self.start_date = start_date #string
        self.end_date = end_date #string 
        
        # Fetch financial data
        self.data = yf.download(ticker, start=start_date, end=end_date) 
        self.dir = f'./candlestick_images/{ticker}/'  #string
        
        self.curr_step_id = 0 #id of first day of current state 

        
        self.data_size = len(self.data)-28 ##number of steps ie nb of images
        self.actionspace = [1, -1, 0]  # Long, Short, No position
        
        
        

    #####
    def getActionIDs(self,actions): #expects a 1d tensor 
        return torch.tensor([self.actionspace.index(action) for action in actions])
    
    def getMatrix(self,startId):
        im = generate_candlestick_image(self.data.iloc[startId:startId+28])        
        return im
    
    def SaveImage(self,im): #actually returns a matrix! 
        # Save the image
        plt.imsave(self.dir+f'./{self.curr_state_id}.png', im, cmap='gray', format='png')
        return im
    
    #####
    
    
    def getCurrState(self): #returns the matrix for that step id
        twoDmatrix = self.getMatrix(self.curr_step_id)
        # Reshape the current state matrix to add the channel dimension
        ## print ( twoDmatrix[np.newaxis, :, :] .shape )
        res= twoDmatrix[np.newaxis, :, :] ##returns an array of dimension [C,W,H] channel 
        return torch.FloatTensor(res) ##A OPTIMISER?????

    def reset(self):
        # Reset environment to initial state
        self.curr_step_id = 0

    
    def stepReward(self, action): ##returns reward and sets state_id to state_id+1
    ##action is action not actionID
        
        # Calculate reward based on daily returns
        if self.curr_step_id < self.data_size: ##ie si on n'est pas a la derniere image
            # Calculate reward based on daily return between current and previous time steps
            curr_close = self.data['Close'][self.curr_step_id+27]
            next_close = self.data['Close'][self.curr_step_id+28]
            daily_return = (next_close - curr_close) / curr_close
            reward = np.sign(action) * daily_return  # Need to implement NRM
        else:
            reward = 0  # End of data, no further reward

        # Transition to the next time step based on action
        self.curr_step_id += 1
        
        return reward
    
    
    
    
##################################################################################
##################################################################################
##################################################################################


    
    def stepRewardNRM(self, action): ##returns reward and sets state_id to state_id+1
    ##action is action not actionID
        
        # Calculate reward based on daily returns
        if self.curr_step_id < self.data_size: ##ie si on n'est pas a la derniere image
            # Calculate reward based on daily return between current and previous time steps
            curr_close = self.data['Close'][self.curr_step_id+27]
            next_close = self.data['Close'][self.curr_step_id+28]
            daily_return = (next_close - curr_close) / curr_close
            initial_reward = np.sign(action) * daily_return  
            
            if initial_reward >= 0:
                reward = initial_reward 
            else:
                if action == 1:  # Buy action
                    reward = initial_reward * 2 #really bad 
                elif action == -1:  # Sell action
                    reward = initial_reward * 1.5 #bad
            
            
            
            
            
        else:
            reward = 0  # End of data, no further reward

        # Transition to the next time step based on action
        self.curr_step_id += 1
        
        return reward
