import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import numpy as np
import gym
import random
from collections import deque





###########################################################################################
###########################################################################################
###########################################################################################




class BasicBuffer:  ##buffer class

  def __init__(self, max_size): ##constructeur
      self.max_size = max_size
      self.buffer = deque(maxlen=max_size)

  def push(self, state, action, reward, next_state, done): ##add an experience tuple to the buffer
      experience = (state, action, np.array([reward]), next_state, done)
      self.buffer.append(experience)

  def sample(self, batch_size):  #pick a random batch of size batch_size from the buffer
      state_batch = []
      action_batch = []
      reward_batch = []
      next_state_batch = []
      done_batch = []

      batch = random.sample(self.buffer, batch_size)


      for experience in batch:
          state, action, reward, next_state, done = experience
          state_batch.append(state)
          action_batch.append(action)
          reward_batch.append(reward)
          next_state_batch.append(next_state)
          done_batch.append(done)

      return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)
      ##un batch contient differents experience tuples, pris randomly depuis le buffer !



  def __len__(self):   ##get the current number of experience tuples in the buffer
      return len(self.buffer)






###########################################################################################
###########################################################################################
###########################################################################################



##agent refers to the DQNAgent model, which contains both models ! 
def mini_batch_train(env, agent, max_episodes, max_steps, batch_size): ##TRAINER func with MINIBATCHES
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0  ##on initialise le episode reward

        for step in range(max_steps):
            action = agent.get_action(state) #l'agent donne une action
            next_state, reward, done, _ = env.step(action) ##on recupere le next state et le reward et on voit si cest la fin 
            agent.replay_buffer.push(state, action, reward, next_state, done) #on push le tuple au buffer
            episode_reward += reward   ##

            if len(agent.replay_buffer) > batch_size: #check si le buffer contient au moins un batch pour s'updater (vrai quasi tout le temps sauf lors des quelques premieres steps)
                agent.update(batch_size)   

            if done or step == max_steps-1:       ##cas ou episode fini
                episode_rewards.append(episode_reward)
                print("Reward for Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

    return episode_rewards


###########################################################################################
###########################################################################################
###########################################################################################

class ConvDQN(nn.Module):   ##creation DU MODELE!   heritage de nn.Module
    
    def __init__(self, input_dim, output_dim):  #constructeur du modele
        super(ConvDQN, self).__init__() ## calls the constructor of the parent class nn.Module. It's necessary to initialize the ConvDQN object properly, as it inherits functionality from nn.Module.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_input_dim = self.feature_size()
        
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )
        
        
    def forward(self, state):
        features = self.conv_net(state)
        features = features.view(features.size(0), -1)
        qvals = self.fc(features)
        return qvals

    def feature_size(self):
        return self.conv_net(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)

###########################################################################################
###########################################################################################
###########################################################################################

##SAME as ConvDQN except this one does no use a set of convolutional layers prior to the fc layers
class DQN(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, state):
        qvals = self.fc(state)
        return qvals

###########################################################################################
###########################################################################################
###########################################################################################


class DQNAgent:

    def __init__(self, env, use_conv=True, learning_rate=3e-4, gamma=0.99, tau=0.01, buffer_size=10000):
        ##init attributes
        self.env = env
        self.use_conv = use_conv
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
	
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ##declaration du main model et target model de l'agent en fonction de si on utilise la conv ou pas..
        if self.use_conv:
            self.model =        ConvDQN(env.observation_space.shape, env.action_space.n).to(self.device)
            self.target_model = ConvDQN(env.observation_space.shape, env.action_space.n).to(self.device)
        else:
            self.model =        DQN(env.observation_space.shape, env.action_space.n).to(self.device)
            self.target_model = DQN(env.observation_space.shape, env.action_space.n).to(self.device)
        ##evidemment ce sont deux modeles à la meme structure, initialisés pareillement au debut de lentrainement
        
        # hard copy model parameters to target model parameters for initialization 
        for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)

        ##alternatively we could initialize the two models on the same line to avoid copying the weights over as they would be initialized with the same weights


        self.optimizer = torch.optim.Adam(self.model.parameters())
    ##END INIT##    
      
    ##ici cest le main model qui donne le nxt state en fonction des qvals quIL aura calculé lui meme
    def get_action(self, state, eps=0.20):  
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)  ##calcul des qvals pour chaque action
        action = np.argmax(qvals.cpu().detach().numpy())   ##on produit laction
        
        ##if we have an exploration (given by the exploration rate eps) the agent takes a random action instead of exploiting its learned policy
        #This encourages the agent to explore the environment and discover new actions that it might not have considered otherwise.
        if(np.random.randn() < eps):
            return self.env.action_space.sample() ##random action

        return action
    
    ####################

    def compute_loss(self, batch):     
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)

        # resize tensors
        actions = actions.view(actions.size(0), 1)
        dones = dones.view(dones.size(0), 1)

        # compute loss
        curr_Q = self.model.forward(states).gather(1, actions)
        
        next_Q = self.target_model.forward(next_states) #target 
        max_next_Q = torch.max(next_Q, 1)[0]
        max_next_Q = max_next_Q.view(max_next_Q.size(0), 1)
        expected_Q = rewards + (1 - dones) * self.gamma * max_next_Q
        
        loss = F.mse_loss(curr_Q, expected_Q.detach())
        
        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        ##update main network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # target network update (should be every 100 steps?)
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


env_id = "CartPole-v0"
MAX_EPISODES = 1000
MAX_STEPS = 500
BATCH_SIZE = 32

env = gym.make(env_id)
agent = DQNAgent(env, use_conv=False)
episode_rewards = mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)