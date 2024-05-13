# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:30:21 2024

@author: loulo
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_training_rewards(array, save_dir,title):
    plt.plot(array)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Ep. Training Return')
    plt.savefig(os.path.join(save_dir, 'training_rewards.png'))
    plt.close()
    
    
def plot_training_and_testing_rewards(testarray, trainarray, save_dir,title):
    plt.plot(trainarray, color='blue', label='training returns')
    plt.plot(testarray, color='red', label='testing returns')
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Ep. Training Return')
    plt.savefig(os.path.join(save_dir, 'training_rewards.png'))
    plt.close()
    
    
'''  
rewards = np.array([])
rewards = np.append(rewards,1)
rewards = np.append(rewards,2)
rewards = np.append(rewards,3)
print(rewards)

rewards = np.load(os.path.join("checkpoints", "rewards.npy"))
print(rewards)

plot_training_rewards(rewards,"checkpoints","ADBE")
'''