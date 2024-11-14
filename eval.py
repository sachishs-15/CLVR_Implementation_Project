import torch
import wandb
from pdb import set_trace
import argparse
import numpy as np
from PIL import Image
import imageio

from sprites_env.envs.sprites import SpritesStateEnv, SpritesEnv
from general_utils import AttrDict
from model.actor_critic import Actor, Critic
from model.encoder import Encoder, ImageRecEncoder

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some numbers')

    parser.add_argument('--run', type=str, default="PPO_States_Env")  
    parser.add_argument('--distractions', type=int, default=0)
    parser.add_argument('--exp', type=str, default="oracle")

    args = parser.parse_args()
    
    data_spec = AttrDict(
            resolution=64,
            max_ep_len=40,
            max_speed=0.05,      
            obj_size=0.2,      
            follow=True,
        )
    
    env = SpritesEnv(n_distractors = args.distractions)
    env.set_config(data_spec)
    obs, _ = env.reset()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    cov_var = torch.full(size=(action_dim,), fill_value=0.5)
    cov_mat = torch.diag(cov_var)

    ac_hidden = 64 # hidden layer size for actor and critic
    actor = Actor(64, action_dim, ac_hidden, cov_mat)

    if args.exp == 'oracle':
        encoder = Encoder(state_dim, 1, 64)
    elif 'image_rec' in args.exp:
        encoder = ImageRecEncoder()
    elif 'reward_prediction' in args.exp:
        encoder = Encoder(64, 1, 64)
    elif 'cnn' in args.exp:
        encoder = Encoder(1024, 1, 64)
    

    actor.load_state_dict(torch.load(f"ppo_save/dist{args.distractions}/{args.exp}_actor.pth"))
    encoder.load_state_dict(torch.load(f"ppo_save/dist{args.distractions}/{args.exp}_encoder.pth"))

total_test_episodes = 5
max_ep_len = 40
gif_images_dir = 'images/gif'

for ep in range(1, total_test_episodes+1):
    
    ep_reward = 0
    state = env.reset()
    images = []

    for t in range(1, max_ep_len+1):
        obs = np.array(obs)

        obs_encoded = encoder(torch.tensor(obs, dtype=torch.float).unsqueeze(0).unsqueeze(0)).detach()
        obs_encoded = obs_encoded.squeeze()
        dist = actor(torch.tensor(obs_encoded, dtype=torch.float))
        action = dist.sample()
        state, reward, done, _, _ = env.step(action.detach().numpy())
        ep_reward += reward

        obs = state
        img = env.render(mode = 'rgb_array')

        img = Image.fromarray(img)
        images.append(img)
        #img.save(gif_images_dir + '/' + str(t).zfill(6) + '.jpg')
        
        if done:
            break
    imageio.mimsave(f'images/gif/{args.exp}_{ep}.gif', images)

    print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))


env.close()