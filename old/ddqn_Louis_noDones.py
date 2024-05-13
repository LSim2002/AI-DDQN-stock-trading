# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:57:03 2024

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

###########################################################################################
###########################################################################################
###########################################################################################




class BasicBuffer:  ##buffer class

  def __init__(self, max_size): ##constructeur
      self.max_size = max_size
      self.buffer = deque(maxlen=max_size)

  def push(self, state, action, reward, next_state): ##add an experience tuple to the buffer
      experience = [state, action, reward, next_state]
      self.buffer.append(experience)

  def sample(self, batch_size):  #pick a random batch of size batch_size from the buffer
        # Sample a random batch of experience tuples from the buffer
        batch = random.sample(self.buffer, batch_size)
        #print(batch)
        return batch

  def __len__(self):   ##get the current number of experience tuples in the buffer
      return len(self.buffer)






###########################################################################################
###########################################################################################
###########################################################################################



#make direct state access within step ite
def mini_batch_train(env, agent, max_episodes, batch_size, target_update_freq): ##TRAINER func with MINIBATCHES
    episode_rewards = []

    for episode in range(max_episodes):
        env.reset()
        episode_reward = 0  ##on initialise le episode reward
        step_count=0 ##initialisation du step count pour le hard update
        
        for step in range(env.data_size):
           # print(step_count)
            state=env.getCurrState()
            action = agent.get_action(state) #l'agent donne une action
            print('step: '+ str(step_count) + '/'+str(env.data_size))

            reward = env.stepReward(action) #get reward etactualiser l'env 
            print('daily reward:' + str(reward))
            episode_reward += reward   ##
            next_state=env.getCurrState()
            agent.replay_buffer.push(state, action, reward, next_state) #on push le tuple au buffer

            #check si le buffer contient au moins un batch pour s'updater (vrai quasi tout le temps sauf lors des quelques premieres steps)
            if len(agent.replay_buffer) > batch_size: 
                agent.update_main(batch_size)   
                
                if step_count % target_update_freq == 0:
                    print('TARGET NETWORK UPDATED')
                    agent.update_target()
                

            if step == env.data_size-1:       ##cas ou episode d'entrainement fini
                episode_rewards.append(episode_reward)
                print("Reward for Episode " + str(episode) + ": " + str(episode_reward))
                break
            
            step_count+=1

    return episode_rewards


###########################################################################################
###########################################################################################
###########################################################################################

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


###########################################################################################
###########################################################################################
###########################################################################################



class DDQNAgent:

    def __init__(self, env, learning_rate=3e-4, gamma=0.99, tau=0.01, buffer_size=10000):
        ##init attributes
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
	
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ##declaration du main model et target model de l'agent 
        self.model =        ConvDQN().to(self.device)
        self.target_model = ConvDQN().to(self.device)
        
        # hard copy model parameters to target model parameters for initialization 
        for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)

        ##alternatively we could initialize the two models on the same line to avoid copying the weights over as they would be initialized with the same weights


        self.optimizer = torch.optim.Adam(self.model.parameters())
    ##END INIT##    
      
    ##ici cest le main model qui donne le nxt state en fonction des qvals quIL aura calculé lui meme
    def get_action(self, state, eps=0.20):  
        ##if we have an exploration (given by the exploration rate eps) the agent takes a random action instead of exploiting its learned policy
        #This encourages the agent to explore the environment and discover new actions that it might not have considered otherwise.
        if(np.random.randn() < eps):
            action = random.choice(self.env.actionspace)
        else:
            state = state.to(self.device)
            qvals_main = self.model.forward(state)  ##calcul des qvals pour chaque action par le main network
            #print(qvals_main)
            actionid = np.argmax(qvals_main.cpu().detach().numpy())   ##on produit laction
            action=env.actionspace[actionid]

        #print('action:'+str(action))
        return action
    
    ####################

    def compute_loss(self, batch):     ##1 loss value per tuple, so this returns an array of loss values (batch)
        
        # Unpack the batch        
        states, actions, rewards, next_states = zip(*batch)
        #print(states)
        
        ##RESIZE
        states = torch.stack(states).to(self.device)
        #print(states)        

        actions = torch.LongTensor(actions).to(self.device)
        #print('actions for curr batch : ')
        #print(actions)
        
        
        rewards = torch.FloatTensor(rewards).to(self.device)
        #print(rewards)
        
        next_states = torch.stack(next_states).to(self.device)
        #print(next_states)


        actionIDs = self.env.getActionIDs(actions)
        #print('actionIDs for curr batch : ')
        #print(actionIDs)
        
        
        ########## compute loss   ###DDQN  ###########
        
        ##############
        
        #get main model's estimation for q values for every tuples in the batch
        #ie Q(s,a,theta) for every tuple in batch
        Q_values_temp = self.model.forward(states)#.gather(1, actions) 
        #print('Q(s,a,theta) for every action for every state sample in batch : ')
        #print(Q_values_temp)
        curr_Q_main = Q_values_temp[torch.arange(Q_values_temp.size(0)), actionIDs]
        #print('Q(s,a,theta) for every state/action tuple in batch : ')
        #print(curr_Q_main)
        
        ##############
        
        ##now to find Qtarget for every tuple in the batch

        # Get the best actions for the next states according to the main network
        # ie argmax Q(s',a',theta) for a' for every tuple in batch
        next_Q_main = self.model.forward(next_states)    
        #print('next_Q_main :')
        #print(next_Q_main)

        best_next_actionIDs_main = torch.argmax(next_Q_main, dim=1, keepdim=False)
        #print('best_next_actionIDs_main :')
        #print(best_next_actionIDs_main)


        # Get the Q-value for the next state using the target network and best actions according to main network for each tuple in batch
        #ie Q(s',a',theta-) where a' = best action for next state as seen by main network
        next_Q_target_temp = self.target_model.forward(next_states)
        next_Q_target_best_actions = next_Q_target_temp[torch.arange(next_Q_target_temp.size(0)), best_next_actionIDs_main]

        # Compute the target Q-values using Double DQN formula for every tuple in batch
        Q_target = rewards + self.gamma * next_Q_target_best_actions

        ##############
        
        #print('Q(s,a,theta) for every state/action tuple in batch : ')
        #print(curr_Q_main)
        #print('Qtarget for every tuple in batch : ')
        #print(Q_target)
        
        #finally, now that we have Qtarget AND Q_main we can calculate the loss:
        loss = F.mse_loss(curr_Q_main, Q_target.detach())    
        
        
        return loss

    def update_main(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        ##update main network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(param)

MAX_EPISODES = 1000
BATCH_SIZE = 8
target_update_freq = 100 ##steps

env = env('ADBE','2013-01-01','2020-01-04')
agent = DDQNAgent(env)
episode_rewards = mini_batch_train(env, agent, MAX_EPISODES, BATCH_SIZE,target_update_freq)