# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:57:03 2024

@author: loulo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import os
import numpy as np
import gym
import random
from collections import deque
from inference import run_inferenceNRM
from environmentLouis2 import env
from utilities import plot_training_rewards, plot_training_and_testing_rewards

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
def mini_batch_train(env, agent, max_episodes, batch_size, target_update_freq,checkpoint_dir,use_NRM,test_start_date, test_end_date,testing): ##TRAINER func with MINIBATCHES

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    episode_rewards = np.array([])
    test_rewards = np.array([])
    
    start_episode = 0

    # Check for existing checkpoints and load 
    if any(filename.startswith('episode') for filename in os.listdir(checkpoint_dir)):
        file_map = [(f, int(f.split('_')[-1])) for f in os.listdir(checkpoint_dir) if f.startswith("episode")]
        last_checkpoint_file, start_episode = max(file_map, key=lambda x: x[1])
        agent.load_checkpoint(os.path.join(checkpoint_dir, last_checkpoint_file))
        print(f"\n Resuming training from episode {start_episode + 1}")
      
    # Check for existing training rewards array and load (only one file)
    if sum(1 for file in os.listdir(checkpoint_dir) if file == "rewards.npy") == 1: 
        episode_rewards = np.load(os.path.join(checkpoint_dir, "rewards.npy"))
        
    # Check for existing testing rewards array and load (only one file)
    if sum(1 for file in os.listdir(checkpoint_dir) if file == "test_rewards.npy") == 1: 
        episode_rewards = np.load(os.path.join(checkpoint_dir, "test_rewards.npy"))


    for episode in range(start_episode+1,max_episodes):
        env.reset()
        episode_reward = 0  ##on initialise le episode reward
        step_count=0 ##initialisation du step count pour le hard update
        
        for step in range(env.data_size):
           # print(step_count)
            state=env.getCurrState()
            action = agent.get_action(state) #l'agent donne une action
            #print('step: '+ str(step_count) + '/'+str(env.data_size))
            
            if use_NRM == True:
                reward = env.stepRewardNRM(action) #get reward etactualiser l'env 
            else:
                reward = env.stepReward(action) #get reward etactualiser l'env 

            #print('daily reward:' + str(reward))
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
                episode_rewards = np.append(episode_rewards,episode_reward)
                ##run inference and plot test_rewards if live testing is enabled
                if testing == True:
                    test_reward = run_inferenceNRM(agent.model, env.ticker, test_start_date, test_end_date)
                    test_rewards = np.append(test_rewards, test_reward)
                    plot_training_and_testing_rewards(test_rewards, episode_rewards, checkpoint_dir,env.ticker)
                else:
                    test_reward = 0  #zero if testin was disabled
                    test_rewards = np.append(test_rewards, test_reward)
                    plot_training_rewards(episode_rewards,checkpoint_dir,env.ticker)

                    
                
                print("Reward for Episode " + str(episode) + ": " + str(episode_reward))
                agent.save_checkpoint(os.path.join(checkpoint_dir, f"episode_{episode}"))
                np.save(os.path.join(checkpoint_dir,"test_rewards.npy"), test_rewards)
                np.save(os.path.join(checkpoint_dir,"rewards.npy"), episode_rewards)
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

    def __init__(self, env, gamma=0.99, buffer_size=1000):
        ##init attributes
        self.env = env
        self.gamma = gamma
        
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
            
            
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_checkpoint(self, checkpoint_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)            
            
            
            
            
#MODEL PARAMETERS

experiment = 'checkpoints4'

buffer_size=1000
BATCH_SIZE = 16
gamma=0.99 ##discount rate
target_update_freq = 100 ##steps
MAX_EPISODES = 1000
use_NRM = True



env = env('ADBE','2013-01-01','2020-01-04')
agent = DDQNAgent(env, gamma, buffer_size)

testing=True
test_start_date = '2020-01-02'
test_end_date = '2020-06-30'


##START TRAINING
mini_batch_train(env, agent, MAX_EPISODES, BATCH_SIZE,target_update_freq, experiment ,use_NRM,test_start_date,test_end_date,testing)




















