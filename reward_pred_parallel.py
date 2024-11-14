# added random seeds

import torch
import torch.nn as nn
import argparse
import wandb
from gym.envs.registration import register
import gym
import numpy as np

from sprites_env.envs.sprites import SpritesEnv
from general_utils import AttrDict
from model.ppo import PPO_images2, PPO_images3
from model.encoder import Encoder
from model.decoder import Decoder
from sprites_env.envs.make_env import make_env
from numpy import random

register(
    id='Sprites-v0',
    entry_point='sprites_env.envs.sprites:SpritesEnv',
    kwargs={'n_distractors': 0}
)

register(
    id='Sprites-v1',
    entry_point='sprites_env.envs.sprites:SpritesEnv',
    kwargs={'n_distractors': 1}
)

register(
    id='Sprites-v2',
    entry_point='sprites_env.envs.sprites:SpritesEnv',
    kwargs={'n_distractors': 2}
)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some numbers')

    parser.add_argument('--run', type=str, default="reward_prediction")
    parser.add_argument('--rollout_mem',  type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--mini_epochs', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--clip', type=int, default=0.2)
    parser.add_argument('--distractions', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--log', type=int, default=1)
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--num_envs', type=int, default=16)
    parser.add_argument('--max_ep_len', type=int, default=40)  # extra argument for this script
    parser.add_argument('--fine_tune', type=int, default=0)
    parser.add_argument('--schedrate', type=int, default=1000)
    parser.add_argument('--entropy_coeff', type=float, default=0)
    
    args = parser.parse_args()

    args.run = args.run + ("ft" if args.fine_tune else "")
    
    data_spec = AttrDict(
            resolution=64,
            max_ep_len=40,
            max_speed=0.05,      
            obj_size=0.2,       
            follow=True,
        )
    
    args.max_ep_len = data_spec.max_ep_len
    
    random_numbers = random.randint(100, size=(args.num_envs))
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    envs = gym.vector.SyncVectorEnv([make_env(f'Sprites-v{args.distractions}', data_spec, seed) for seed in random_numbers])
    obs, _ = envs.reset()

    encoder = Encoder(data_spec.resolution, 1)
    path = f"pretrained_models/encoder_cond_{args.distractions}.pth"
    encoder_dict = torch.load(path)
    encoder.load_state_dict(encoder_dict)

    if args.log:
        wandb.init(project=f"Sprites-v{args.distractions}", name=args.run, config=args)
    
    ppo = PPO_images3(envs, encoder, args, decoder=None)
    ppo.train()
    
    # if args.save:
    #         ft = "frozen" if args.freeze else "ft"
    #         torch.save(args.encoder.state_dict(), f"pretrained_models/ppo/reward_pred_{ft}_{args.env.n_distractors}/encooder.pth")
    #         torch.save(args.actor.state_dict(), f"pretrained_models/ppo/reward_pred_{ft}_{args.env.n_distractors}/actor.pth")
    #         torch.save(args.critic.state_dict(), f"pretrained_models/ppo/reward_pred_{ft}_{args.env.n_distractors}/critc.pth")