# added entropy to the loss function

import torch
import torch.nn as nn
import numpy as np
import wandb
import argparse
from pdb import set_trace
import cv2

from model.actor_critic import Actor, Critic
from sprites_env.envs.sprites import SpritesStateEnv
from general_utils import AttrDict

class PPO_images3(nn.Module):

    def __init__(self, envs, encoder, args, decoder, state_dim = 64):

        super().__init__()

        self.envs = envs
        self.state_dim = state_dim
        self.action_dim = self.envs.action_space.shape[1]

        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.mini_epochs = args.mini_epochs
        self.clip = args.clip
        self.distractors = args.distractions
        self.rollout_mem = args.rollout_mem
        self.device = "cuda:" + args.device   # which gpu number to use
        self.log = args.log
        self.lr = args.lr
        self.num_envs = args.num_envs
        self.fine_tune = args.fine_tune
        self.run = args.run
        self.schedrate = args.schedrate
        self.entropy_coeff = args.entropy_coeff
        self.step = 0

        self.gamma = 0.99
        self.gae = 0.95

        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)
        
        self.encoder = encoder 
        self.encoder.to(self.device)

        if decoder is not None:
            self.decoder = decoder
            self.decoder.to(self.device)

        ac_hidden = 64 # hidden layer size for actor and critic
        self.actor = Actor(self.state_dim, self.action_dim, ac_hidden, self.cov_mat).to(self.device)
        self.critic = Critic(self.state_dim, ac_hidden).to(self.device)

        param = list(self.actor.parameters()) + list(self.critic.parameters()) 
        if self.fine_tune:
            self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=2.5e-4)
            self.encoder_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=self.schedrate, gamma=0.95)

        self.optimizer = torch.optim.Adam((param), lr=self.lr)
              
    def rollout_data(self):

        rewards = []; actions = []; probs = []; values = []; states = []; discounted_rewards = []; encoded_states = []; dones = []; 

        obs, _ = self.envs.reset()
        done = False 

        while not done:
            #set_trace()
            obs = np.array(obs) * 2 - 1  # inconsistency in the encoder training and environment
            obs = torch.tensor(obs, dtype=torch.float).detach().to(self.device)

            obs_encoded = self.encoder(obs.unsqueeze(1)).detach()

            # decoding = self.decoder(obs_encoded)
            # img_gt = (obs.unsqueeze(1).cpu().numpy() + 1.0)* 255./ 2
            # img_pd = (decoding.cpu().detach().numpy() + 1.0)* 255./ 2

            # gt_seq = cv2.hconcat(img_gt.transpose(0, 2, 3, 1))
            # pd_seq = cv2.hconcat(img_pd.transpose(0, 2, 3, 1))

            # # vertically stack the images
            # check = cv2.vconcat([gt_seq, pd_seq])
            # cv2.imwrite(f"images/test_reward_parallel_rollout.jpg", check)
            
            dist = self.actor(obs_encoded)
            action = dist.sample()
            prob = dist.log_prob(action).detach()

            state, reward, done, _, _ = self.envs.step(action.cpu().detach().numpy())
            #print(reward)
            value = self.critic(obs_encoded).detach().squeeze()

            encoded_states.append(obs_encoded)
            actions.append(action)
            probs.append(prob)
            rewards.append(torch.tensor(reward, dtype=torch.float))
            states.append(obs)
            dones.append(done)
            values.append(value)

            done = done[0]
            obs = state

        if self.log:
            #self.step += 40*self.num_envs
            wandb.log({"mean rewards": float(torch.mean(sum(rewards)))})

        #set_trace()

        discounted_rewards = self.discounted_rewards_gae(rewards, dones, values)
        discounted_rewards = torch.cat(discounted_rewards)

        states = torch.cat(states)
        encoded_states = torch.cat(encoded_states)
        actions = torch.cat(actions)
        probs = torch.cat(probs)
        rewards = torch.cat(rewards)
        values = torch.cat(values)

        return states.to(self.device), encoded_states.to(self.device), values.to(self.device), actions.to(self.device), probs.to(self.device), discounted_rewards.to(self.device)
     
    def discounted_rewards(self, rewards):

        discounted_rewards = []
        G = np.zeros(self.num_envs, dtype=np.float32)
        rewards.reverse()

        for i in range(len(rewards)):
            G = rewards[i] + self.gamma * G
            discounted_rewards.insert(0, G)
    
        return discounted_rewards
    
    def discounted_rewards_gae(self, rewards, dones, values):
            
        discounted_rewards = []

        last_advantage = 0
        last_value = values[-1]

        for i in reversed(range(len(rewards))):
            mask = 1 - dones[i]
            last_value = last_value.cpu() * mask
            last_advantage = last_advantage * mask

            delta = rewards[i].cpu() + self.gamma * last_value - values[i].cpu()
            last_advantage = delta + self.gamma * self.gae * last_advantage

            discounted_rewards.insert(0, last_advantage.type(torch.float).to(self.device) + values[i])
            last_value = values[i]
    
        return discounted_rewards

    def update_policy(self):

        states, encoded_states, values, actions, probs, discounted_rewards = self.rollout_data() # collect data
        #print("hur")

        #set_trace()
        for data in range(self.mini_epochs):

            indices = np.random.choice(range(states.shape[0]), self.batch_size)
            indices = torch.tensor(indices, dtype=torch.long).to(self.device)

            states_x = torch.index_select(states, 0, indices)
            encoded_states_x = torch.index_select(encoded_states, 0, indices)

            if self.fine_tune:
                encoded_states_x = self.encoder(states_x.unsqueeze(1))
            # decoding = self.decoder(encoded_states_x)

            # img_gt = (states_x.unsqueeze(1).cpu().numpy() + 1.0)* 255./ 2
            # img_pd = (decoding.cpu().detach().numpy() + 1.0)* 255./ 2
            
            # gt_seq = cv2.hconcat(img_gt.transpose(0, 2, 3, 1))
            # pd_seq = cv2.hconcat(img_pd.transpose(0, 2, 3, 1))

            # vertically stack the images
            # check = cv2.vconcat([gt_seq, pd_seq])
            # cv2.imwrite(f"images/test_reward_parallel.jpg", check)
            
            values_x = torch.index_select(values, 0, indices)
            actions_x = torch.index_select(actions, 0, indices)
            old_probs_x = torch.index_select(probs, 0, indices)
            discounted_rewards_x = torch.index_select(discounted_rewards, 0, indices)
            

            dist = self.actor(encoded_states_x)
            new_probs = dist.log_prob(actions_x)
            entropy = dist.entropy().mean() ###############
            
            prob_ratio = new_probs.exp() / old_probs_x.exp()
            
            with torch.no_grad():
                advantage = discounted_rewards_x - values_x

            surr1 = prob_ratio * advantage
            surr2 = torch.clamp(prob_ratio, 1 - self.clip, 1 + self.clip) * advantage

            actor_loss = -(torch.min(surr1, surr2)).mean()
            
            x = self.critic(encoded_states_x)
            x = x.squeeze()

            discounted_rewards_x = discounted_rewards_x.squeeze()

            critic_loss = nn.MSELoss()(x, discounted_rewards_x)

            if self.log:
                wandb.log({"actor loss": actor_loss.item()})
                wandb.log({"critic loss": critic_loss.item()})
            
            total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coeff * entropy ###############

            self.optimizer.zero_grad()
            if self.fine_tune:
                self.encoder_optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if self.fine_tune:
                self.encoder_optimizer.step()
                self.encoder_scheduler.step()

    def train(self):
            for k in range(self.epochs):
                self.update_policy()
                if k % 1000 == 0:
                    self.save_model()

    def save_model(self):
        torch.save(self.encoder.state_dict(), f"ppo_save/dist{self.distractors}/{self.run}_encoder.pth")
        torch.save(self.actor.state_dict(), f"ppo_save/dist{self.distractors}/{self.run}_actor.pth")
        torch.save(self.critic.state_dict(), f"ppo_save/dist{self.distractors}/{self.run}_critic.pth")

class PPO_images2(nn.Module):

    def __init__(self, envs, encoder, args, decoder, state_dim = 64):

        super().__init__()

        self.envs = envs
        self.state_dim = state_dim
        self.action_dim = self.envs.action_space.shape[1]

        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.mini_epochs = args.mini_epochs
        self.clip = args.clip
        self.distractors = args.distractions
        self.rollout_mem = args.rollout_mem
        self.device = "cuda:" + args.device   # which gpu number to use
        self.log = args.log
        self.lr = args.lr
        self.num_envs = args.num_envs
        self.fine_tune = args.fine_tune

        self.gamma = 0.99
        self.gae = 0.95

        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)
        
        self.encoder = encoder 
        self.encoder.to(self.device)

        if decoder is not None:
            self.decoder = decoder
            self.decoder.to(self.device)

        ac_hidden = 64 # hidden layer size for actor and critic
        self.actor = Actor(self.state_dim, self.action_dim, ac_hidden, self.cov_mat).to(self.device)
        self.critic = Critic(self.state_dim, ac_hidden).to(self.device)

        param = list(self.actor.parameters()) + list(self.critic.parameters()) 
        if self.fine_tune:
            param = param + list(self.encoder.parameters())
        self.optimizer = torch.optim.Adam((param), lr=self.lr)
              
    def rollout_data(self):

        rewards = []; actions = []; probs = []; values = []; states = []; discounted_rewards = []; encoded_states = []; dones = []

        obs, _ = self.envs.reset()
        done = False 

        while not done:
            #set_trace()
            obs = np.array(obs) * 2 - 1  # inconsistency in the encoder training and environment
            obs = torch.tensor(obs, dtype=torch.float).detach().to(self.device)

            obs_encoded = self.encoder(obs.unsqueeze(1)).detach()

            # decoding = self.decoder(obs_encoded)
            # img_gt = (obs.unsqueeze(1).cpu().numpy() + 1.0)* 255./ 2
            # img_pd = (decoding.cpu().detach().numpy() + 1.0)* 255./ 2

            # gt_seq = cv2.hconcat(img_gt.transpose(0, 2, 3, 1))
            # pd_seq = cv2.hconcat(img_pd.transpose(0, 2, 3, 1))

            # # vertically stack the images
            # check = cv2.vconcat([gt_seq, pd_seq])
            # cv2.imwrite(f"images/test_reward_parallel_rollout.jpg", check)
            
            dist = self.actor(obs_encoded)
            action = dist.sample()
            prob = dist.log_prob(action).detach()

            state, reward, done, _, _ = self.envs.step(action.cpu().detach().numpy())
            #print(reward)
            value = self.critic(obs_encoded).detach().squeeze()

            encoded_states.append(obs_encoded)
            actions.append(action)
            probs.append(prob)
            rewards.append(torch.tensor(reward, dtype=torch.float))
            states.append(obs)
            dones.append(done)
            values.append(value)

            done = done[0]
            obs = state

        if self.log:
            wandb.log({"mean rewards": float(torch.mean(sum(rewards)))})

        #set_trace()

        discounted_rewards = self.discounted_rewards_gae(rewards, dones, values)
        discounted_rewards = torch.cat(discounted_rewards)

        states = torch.cat(states)
        encoded_states = torch.cat(encoded_states)
        actions = torch.cat(actions)
        probs = torch.cat(probs)
        rewards = torch.cat(rewards)
        values = torch.cat(values)

        return states.to(self.device), encoded_states.to(self.device), values.to(self.device), actions.to(self.device), probs.to(self.device), discounted_rewards.to(self.device)
     
    def discounted_rewards(self, rewards):

        discounted_rewards = []
        G = np.zeros(self.num_envs, dtype=np.float32)
        rewards.reverse()

        for i in range(len(rewards)):
            G = rewards[i] + self.gamma * G
            discounted_rewards.insert(0, G)
    
        return discounted_rewards
    
    def discounted_rewards_gae(self, rewards, dones, values):
            
        discounted_rewards = []

        last_advantage = 0
        last_value = values[-1]

        for i in reversed(range(len(rewards))):
            mask = 1 - dones[i]
            last_value = last_value.cpu() * mask
            last_advantage = last_advantage * mask

            delta = rewards[i].cpu() + self.gamma * last_value - values[i].cpu()
            last_advantage = delta + self.gamma * self.gae * last_advantage

            discounted_rewards.insert(0, last_advantage.type(torch.float).to(self.device) + values[i])
            last_value = values[i]
    
        return discounted_rewards

    def update_policy(self):


        states, encoded_states, values, actions, probs, discounted_rewards = self.rollout_data() # collect data
        #print("hur")

        #set_trace()
        for data in range(self.mini_epochs):

            indices = np.random.choice(range(states.shape[0]), self.batch_size)
            indices = torch.tensor(indices, dtype=torch.long).to(self.device)

            states_x = torch.index_select(states, 0, indices)
            encoded_states_x = torch.index_select(encoded_states, 0, indices)

            if self.fine_tune:
                encoded_states_x = self.encoder(states_x.unsqueeze(1))
            # decoding = self.decoder(encoded_states_x)

            # img_gt = (states_x.unsqueeze(1).cpu().numpy() + 1.0)* 255./ 2
            # img_pd = (decoding.cpu().detach().numpy() + 1.0)* 255./ 2
            
            # gt_seq = cv2.hconcat(img_gt.transpose(0, 2, 3, 1))
            # pd_seq = cv2.hconcat(img_pd.transpose(0, 2, 3, 1))

            # vertically stack the images
            # check = cv2.vconcat([gt_seq, pd_seq])
            # cv2.imwrite(f"images/test_reward_parallel.jpg", check)
            
            values_x = torch.index_select(values, 0, indices)
            actions_x = torch.index_select(actions, 0, indices)
            old_probs_x = torch.index_select(probs, 0, indices)
            discounted_rewards_x = torch.index_select(discounted_rewards, 0, indices)

            dist = self.actor(encoded_states_x)
            new_probs = dist.log_prob(actions_x)
            entropy = dist.entropy().mean() ###############
            
            prob_ratio = new_probs.exp() / old_probs_x.exp()
            
            with torch.no_grad():
                advantage = discounted_rewards_x - values_x

            surr1 = prob_ratio * advantage
            surr2 = torch.clamp(prob_ratio, 1 - self.clip, 1 + self.clip) * advantage

            actor_loss = -(torch.min(surr1, surr2)).mean()
            
            x = self.critic(encoded_states_x)
            x = x.squeeze()

            discounted_rewards_x = discounted_rewards_x.squeeze()

            critic_loss = nn.MSELoss()(x, discounted_rewards_x)

            if self.log:
                wandb.log({"actor loss": actor_loss.item()})
                wandb.log({"critic loss": critic_loss.item()})
            
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy ###############

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

    def train(self):
            for k in range(self.epochs):
                #print(k)
                self.update_policy()
                if k % 1000 == 0:
                    self.save_model()

    def save_model(self):
        torch.save(self.encoder.state_dict(), f"ppo_save/dist{self.distractors}/encoder.pth")
        torch.save(self.actor.state_dict(), f"ppo_save/dist{self.distractors}/actor.pth")
        torch.save(self.critic.state_dict(), f"ppo_save/dist{self.distractors}/critic.pth")

class PPO_states2(nn.Module):

    def __init__(self, envs, args):

        super().__init__()

        self.envs = envs
        self.state_dim = self.envs.observation_space.shape[1]
        self.action_dim = self.envs.action_space.shape[1]

        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.mini_epochs = args.mini_epochs
        self.clip = args.clip
        self.rollout_mem = args.rollout_mem
        self.device = "cuda:" + args.device   # which gpu number to use
        self.log = args.log
        self.lr = args.lr
        self.num_envs = args.num_envs
        self.distractors = args.distractions
        self.run = args.run

        self.gamma = 0.99
        self.gae = 0.95

        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

        ac_hidden = 64 # hidden layer size for actor and critic
        self.actor = Actor(self.state_dim, self.action_dim, ac_hidden, self.cov_mat).to(self.device)
        self.critic = Critic(self.state_dim, ac_hidden).to(self.device)

        param = list(self.actor.parameters()) + list(self.critic.parameters()) 
        self.optimizer = torch.optim.Adam((param), lr=self.lr)
              
    def rollout_data(self):

        rewards = []; actions = []; probs = []; values = []; states = []; discounted_rewards = []; dones = []
        ep_rewards = []

        obs, _ = self.envs.reset()
        done = False 

        #print("hello")
        #et_trace()

        while not done:
            obs = torch.tensor(np.array(obs), dtype=torch.float).detach().to(self.device)
            dist = self.actor(obs)
            action = dist.sample()
            prob = dist.log_prob(action).detach()

            state, reward, done, _, _ = self.envs.step(action.cpu().detach().numpy())
            value = self.critic(obs).detach().squeeze()

            actions.append(action)
            probs.append(prob)
            rewards.append(torch.tensor(reward, dtype=torch.float))
            states.append(obs)
            dones.append(done)
            values.append(value)

            done = done[0]
            obs = state

        if self.log:
            wandb.log({"mean rewards": float(torch.mean(sum(rewards)))})

        #set_trace()
        discounted_rewards = self.discounted_rewards_gae(rewards, dones, values)
        discounted_rewards = torch.cat(discounted_rewards)

        states = torch.cat(states)
        actions = torch.cat(actions)
        probs = torch.cat(probs)
        rewards = torch.cat(rewards)
        values = torch.cat(values)

        return states.to(self.device), values.to(self.device), actions.to(self.device), probs.to(self.device), discounted_rewards.to(self.device)
     
    def discounted_rewards(self, rewards):

        discounted_rewards = []
        G = np.zeros(self.num_envs, dtype=np.float32)
        rewards.reverse()

        for i in range(len(rewards)):
            G = rewards[i] + self.gamma * G
            discounted_rewards.insert(0, G)
    
        return discounted_rewards

    def discounted_rewards_gae(self, rewards, dones, values):
            
        discounted_rewards = []

        last_advantage = 0
        last_value = values[-1]

        for i in reversed(range(len(rewards))):
            mask = 1 - dones[i]
            last_value = last_value.cpu() * mask
            last_advantage = last_advantage * mask

            delta = rewards[i].cpu() + self.gamma * last_value - values[i].cpu()
            last_advantage = delta + self.gamma * self.gae * last_advantage

            discounted_rewards.insert(0, last_advantage.type(torch.float).to(self.device) + values[i])
            last_value = values[i]
    
        return discounted_rewards
    
    def update_policy(self):

        states, values, actions, probs, discounted_rewards = self.rollout_data() # collect data
        #print("hur")

        #set_trace()
        for data in range(self.mini_epochs):

            indices = np.random.choice(range(states.shape[0]), self.batch_size)
            indices = torch.tensor(indices, dtype=torch.long).to(self.device)

            states_x = torch.index_select(states, 0, indices)
            values_x = torch.index_select(values, 0, indices)
            actions_x = torch.index_select(actions, 0, indices)
            old_probs_x = torch.index_select(probs, 0, indices)
            discounted_rewards_x = torch.index_select(discounted_rewards, 0, indices)

            dist = self.actor(states_x)
            new_probs = dist.log_prob(actions_x)
            entropy = dist.entropy().mean() ###############
            
            prob_ratio = new_probs.exp() / old_probs_x.exp()
            
            with torch.no_grad():
                advantage = discounted_rewards_x - values_x

            surr1 = prob_ratio * advantage
            surr2 = torch.clamp(prob_ratio, 1 - self.clip, 1 + self.clip) * advantage

            actor_loss = -(torch.min(surr1, surr2)).mean()
            
            x = self.critic(states_x)
            x = x.squeeze()

            discounted_rewards_x = discounted_rewards_x.squeeze()

            critic_loss = nn.MSELoss()(x, discounted_rewards_x)

            if self.log:
                wandb.log({"actor loss": actor_loss.item()})
                wandb.log({"critic loss": critic_loss.item()})
            
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy ###############

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

    def train(self):
            for k in range(self.epochs):
                #print(k)
                self.update_policy()
                if k % 1000 == 0:
                    self.save_model()

    def save_model(self):
        torch.save(self.actor.state_dict(), f"ppo_save/dist{self.distractors}/{self.run}_actor.pth")
        torch.save(self.critic.state_dict(), f"ppo_save/dist{self.distractors}/{self.run}_critic.pth")

class PPO_states(nn.Module):

    def __init__(self, env, args):

        super().__init__()

        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.mini_epochs = args.mini_epochs
        self.clip = args.clip
        self.rollout_mem = args.rollout_mem
        self.device = "cuda:" + args.device   # which gpu number to use
        self.log = args.log
        self.lr = args.lr

        self.gamma = 0.99

        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

        ac_hidden = 64 # hidden layer size for actor and critic
        self.actor = Actor(self.state_dim, self.action_dim, ac_hidden, self.cov_mat).to(self.device)
        self.critic = Critic(self.state_dim, ac_hidden).to(self.device)

        param = list(self.actor.parameters()) + list(self.critic.parameters()) 
        self.optimizer = torch.optim.Adam((param), lr=self.lr)
              
    def rollout_data(self):

        rewards = []; actions = []; probs = []; values = []; states = []; discounted_rewards = []
        ep_rewards = []

        while iters < self.rollout_mem:

            obs = self.env.reset()
            done = False
            rewards = []
            
            while not done:

                obs = np.array(obs)
                
                dist = self.actor(torch.tensor(obs, dtype=torch.float).to(self.device))
                action = dist.sample()
                prob = dist.log_prob(action).detach()
                
                state, reward, done, _ = self.env.step(action.cpu().detach().numpy())

                if done: 
                    break
                
                iters +=1 

                states.append(obs)
                rewards.append(reward)
                actions.append(action)
                probs.append(prob)
                
                obs = state
                
            discounted_rewards = discounted_rewards + self.discounted_rewards(rewards)
            ep_rewards.append(sum(rewards))

            if iters % 5 == 0: 

                if self.log:
                    wandb.log({"mean rewards": np.mean(ep_rewards)})
                ep_rewards = []
        
        values = self.critic(torch.tensor(np.array(states), dtype=torch.float).to(self.device)).detach()

        return torch.tensor(np.array(states), dtype=torch.float).to(self.device), values, torch.stack(actions), torch.stack(probs), torch.tensor(discounted_rewards, dtype=torch.float).to(self.device)
     
    def discounted_rewards(self, rewards):

        discounted_rewards = []
        G = 0
        rewards.reverse()

        for i in range(len(rewards)):
            G = rewards[i] + self.gamma * G
            discounted_rewards.insert(0, G)
    
        return discounted_rewards

    def update_policy(self):

        states, values, actions, probs, discounted_rewards = self.rollout_data() # collect data

        for data in range(self.mini_epochs):

            indices = np.random.choice(range(self.rollout_mem), self.batch_size)
            indices = torch.tensor(indices, dtype=torch.long).to(self.device)

            states_x = torch.index_select(states, 0, indices)
            values_x = torch.index_select(values, 0, indices)
            actions_x = torch.index_select(actions, 0, indices)
            old_probs_x = torch.index_select(probs, 0, indices)
            discounted_rewards_x = torch.index_select(discounted_rewards, 0, indices)

            dist = self.actor(states_x)
            new_probs = dist.log_prob(actions_x)
            
            prob_ratio = new_probs.exp() / old_probs_x.exp()
            
            with torch.no_grad():
                advantage = discounted_rewards_x - values_x

            surr1 = prob_ratio * advantage
            surr2 = torch.clamp(prob_ratio, 1 - self.clip, 1 + self.clip) * advantage

            actor_loss = -(torch.min(surr1, surr2)).mean()
            
            x = self.critic(states_x)
            x = x.squeeze()

            discounted_rewards_x = discounted_rewards_x.squeeze()

            critic_loss = nn.MSELoss()(x, discounted_rewards_x)

            if self.log:
                wandb.log({"actor loss": actor_loss.item()})
                wandb.log({"critic loss": critic_loss.item()})
            
            total_loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

    def train(self):

        for k in range(self.epochs):
            self.update_policy()

class PPO_images(nn.Module):

    def __init__(self, env, encoder, args, state_dim = 64):

        super().__init__()

        self.env = env
        self.state_dim = state_dim
        self.action_dim = self.env.action_space.shape[0]

        self.mini_epochs = args.mini_epochs
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.epochs = args.epochs
        self.clip = args.clip
        self.rollout_mem = args.rollout_mem
        self.log = args.log
        self.lr = args.lr
        self.freeze = args.freeze
        self.encoder_lr = args.encoder_lr
        self.batch_size = args.batch_size

        self.gamma = 0.99

        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)            

        self.encoder = encoder
        self.encoder.to(self.device)

        ac_hidden = 64 # hidden layer size for actor and critic
        self.actor = Actor(self.state_dim, self.action_dim, ac_hidden, self.cov_mat).to(self.device)
        self.critic = Critic(self.state_dim, ac_hidden).to(self.device)

        param = list(self.actor.parameters()) + list(self.critic.parameters()) 
        if not self.freeze:
            param = param + list(self.encoder.parameters()) # add encoder parameters to optimizer

        self.combined_optimizer = torch.optim.Adam((param), lr=self.lr)

    def rollout_data(self):

        rewards = []; actions = []; probs = []; values = []; states = []; encoded_states = []; discounted_rewards = []
        ep_rewards = []

        iters = 0

        while iters < self.rollout_mem:

            obs = self.env.reset()
            done = False
            rewards = []

            while not done:
                
                #set_trace()
                obs = np.array(obs)
                obs_encoded = self.encoder(torch.tensor(obs, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(self.device)).detach()
                obs_encoded = obs_encoded.squeeze()
            
                dist = self.actor(obs_encoded)
                action = dist.sample()
                prob = dist.log_prob(action).detach()
                
                state, reward, done, _ = self.env.step(action.cpu().detach())

                if done: 
                    break
                
                iters +=1 

                encoded_states.append(obs_encoded)
                states.append(obs)
                rewards.append(reward)
                actions.append(action)
                probs.append(prob)

                obs = state
            
            discounted_rewards = discounted_rewards + self.discounted_rewards(rewards)
            ep_rewards.append(sum(rewards))

            if iters % 5 == 0: 
                if self.log:
                    wandb.log({"mean rewards": np.mean(ep_rewards)})
                ep_rewards = []

        encoded_states = torch.stack(encoded_states)
        values = self.critic(encoded_states).detach()
        
        return torch.tensor(np.array(states), dtype=torch.float).unsqueeze(1).to(self.device), encoded_states, values, torch.stack(actions), torch.stack(probs), torch.tensor(discounted_rewards, dtype=torch.float).to(self.device)
    
    def discounted_rewards(self, rewards):

        discounted_rewards = []
        G = 0
        rewards.reverse()

        for i in range(len(rewards)):
            G = rewards[i] + self.gamma * G
            discounted_rewards.insert(0, G)
    
        return discounted_rewards
    
    def update_policy(self):
        
        #set_trace()
        states, enc_states, values, actions, probs, discounted_rewards = self.rollout_data()
    
        for data in range(self.mini_epochs):

            indices = np.random.choice(range(self.rollout_mem), self.batch_size)
            indices = torch.tensor(indices, dtype=torch.long).to(self.device)

            states_x = torch.index_select(states, 0, indices)
            enc_states_x = torch.index_select(enc_states, 0, indices)
            values_x = torch.index_select(values, 0, indices)
            actions_x = torch.index_select(actions, 0, indices)
            old_probs_x = torch.index_select(probs, 0, indices)
            discounted_rewards_x = torch.index_select(discounted_rewards, 0, indices)
            
            #if not self.freeze:
                #enc_states_x = self.encoder(states_x)
                #self.encoder_optimizer.zero_grad()

            dist = self.actor(enc_states_x)
            new_probs = dist.log_prob(actions_x)
            
            prob_ratio = new_probs.exp() / old_probs_x.exp()
        
            with torch.no_grad():
                advantage = discounted_rewards_x - values_x

            surr1 = prob_ratio * advantage
            surr2 = torch.clamp(prob_ratio, 1 - self.clip, 1 + self.clip) * advantage
            actor_loss = -(torch.min(surr1, surr2)).mean()

            x = self.critic(enc_states_x)
            x = x.squeeze()
            discounted_rewards_x = discounted_rewards_x.squeeze()
            critic_loss = nn.MSELoss()(x, discounted_rewards_x)

            if self.log:
                wandb.log({"actor loss": actor_loss.item()})
                wandb.log({"critic loss": critic_loss.item()})
            
            total_loss = actor_loss + 0.5 * critic_loss


            self.combined_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()) , 1)
            self.combined_optimizer.step()

            #if not self.freeze:
              #  self.encoder_optimizer.step()

    def train(self):

        for k in range(self.epochs):
            self.update_policy()
    