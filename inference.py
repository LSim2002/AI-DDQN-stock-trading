# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 20:28:24 2024

@author: loulo
"""

import torch
import yfinance as yf
from data_preparation import generate_candlestick_image
import numpy as np

def run_inferenceNRM(argmodel, ticker, start_date, end_date):
    
    # Load the trained model
    model = argmodel.eval()  # Set model to evaluation mode
    ##do I need to import the param dict or do they come with the model? 
    
    # load the data
    data = yf.download(ticker, start=start_date, end=end_date) 
    data_size = len(data)-28 ##number of steps ie nb of images
    
    test_reward = 0
    
    #investment = 100    
    #invested = False
    
    for left_day_number in range(data_size): #jour n 0
        
        curr_data = data.iloc[left_day_number:left_day_number+28] #jours n 0 a 27
        curr_state = generate_candlestick_image(curr_data) #jours n 0 a 27
        curr_state = torch.FloatTensor(curr_state[np.newaxis, :, :])
        
        qvals_main = model.forward(curr_state)  ##calcul des qvals pour chaque action par le main network
        actionid = np.argmax(qvals_main.cpu().detach().numpy())   ##on produit laction
        action= [1, -1, 0][actionid]

        curr_close = data['Close'][left_day_number+27]  #jour n 27
        next_close = data['Close'][left_day_number+28] #jour n 28 (pas encore vu)
        daily_return = (next_close - curr_close) / curr_close  #retour geometrique journalier
        initial_reward = np.sign(action) * daily_return  
            
        if initial_reward >= 0:
            reward = initial_reward 
        else:
            if action == 1:  # Buy action
                reward = initial_reward * 2 #really bad 
            elif action == -1:  # Sell action
                reward = initial_reward * 1.5 #bad
                
        test_reward+=reward
        

    return test_reward

def run_inference(argmodel, ticker, start_date, end_date):
    
    # Load the trained model
    model = argmodel.eval()  # Set model to evaluation mode
    ##do I need to import the param dict or do they come with the model? 
    
    # load the data
    data = yf.download(ticker, start=start_date, end=end_date) 
    data_size = len(data)-28 ##number of steps ie nb of images
    
    test_reward = 0
    
    #investment = 100    
    #invested = False
    
    for left_day_number in range(data_size): #jour n 0
        
        curr_data = data.iloc[left_day_number:left_day_number+28] #jours n 0 a 27
        curr_state = generate_candlestick_image(curr_data) #jours n 0 a 27
        
        qvals_main = model.forward(curr_state)  ##calcul des qvals pour chaque action par le main network
        actionid = np.argmax(qvals_main.cpu().detach().numpy())   ##on produit laction
        action= [1, -1, 0][actionid]

        curr_close = data['Close'][left_day_number+27]  #jour n 27
        next_close = data['Close'][left_day_number+28] #jour n 28 (pas encore vu)
        daily_return = (next_close - curr_close) / curr_close  #retour geometrique journalier
        initial_reward = np.sign(action) * daily_return  
            

                
        test_reward+=initial_reward
        

    return test_reward
