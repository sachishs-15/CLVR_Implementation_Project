import gym
from gym.envs.registration import register
from general_utils import AttrDict
from sprites_env.envs.sprites import SpritesEnv
import numpy as np
import torch
from model.actor_critic import Actor, Critic
from model.encoder import Encoder
from pdb import set_trace

register(
    id='Sprites-v0',
    entry_point='sprites_env.envs.sprites:SpritesEnv',
    kwargs={'n_distractors': 0}
)

def make_env(gym_id, data_spec, seed):
    def thunk():
        env = gym.make(gym_id)
        env.set_config(data_spec)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        return env
    return thunk

data_spec = AttrDict(
            resolution=64,
            max_ep_len=40,
            max_speed=0.05,      
            obj_size=0.2,       
            follow=True,
        )

envs = gym.vector.SyncVectorEnv([make_env('Sprites-v0', data_spec, seed) for seed in range(16)])
print(envs.observation_space)
observations = envs.reset()
print(observations)

states = []
rewards = []
actions = []
probs = []

actor = Actor(64, 2, 64, torch.diag(torch.full(size=(2,), fill_value=0.5)))
encoder = Encoder(64, 1)

obs, _ = envs.reset()

for i in range(40):

    obs = np.array(obs)
    
    enc_obs = encoder(torch.tensor(obs, dtype=torch.float).unsqueeze(1)).detach()
    enc_obs = enc_obs.squeeze()
    dist = actor(enc_obs)
    action = dist.sample()
    prob = dist.log_prob(action).detach()
    
    state, reward, done, _, _ = envs.step(action.cpu().detach().numpy())
    
    states.append(obs)
    rewards.append(reward)
    actions.append(action)
    probs.append(prob)
    
    obs = state

set_trace()
