import torch
import wandb
from pdb import set_trace
import argparse
import numpy as np
import gym
from gym.envs.registration import register

from sprites_env.envs.sprites import SpritesStateEnv
from general_utils import AttrDict
from model.ppo import PPO_states2
from sprites_env.envs.make_env import make_env

register(
    id='SpritesState-v0',
    entry_point='sprites_env.envs.sprites:SpritesStateEnv',
    kwargs={'n_distractors': 0}
)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some numbers')

    parser.add_argument('--run', type=str, default="oracle")  
    parser.add_argument('--rollout_mem',  type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--mini_epochs', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=30000)
    parser.add_argument('--clip', type=int, default=0.2)
    parser.add_argument('--distractions', type=int, default=0)
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--log', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--num_envs', type=int, default=16)

    args = parser.parse_args()

    print(args)
    
    data_spec = AttrDict(
            resolution=64,
            max_ep_len=40,
            max_speed=0.05,      
            obj_size=0.2,      
            follow=True,
        )

    envs = gym.vector.SyncVectorEnv([make_env(f'SpritesState-v{args.distractions}', data_spec, seed) for seed in range(args.num_envs)])
    obs, _ = envs.reset()

    if args.log:
        wandb.init(project=f"Sprites-v{args.distractions}", name=args.run, config=args)

    ppo = PPO_states2(envs, args)
    ppo.train()

    if args.save:
        torch.save(ppo.actor.state_dict(), f"pretrained_models/ppo/oracle_{args.distractions}/actor.pth")
        torch.save(ppo.critic.state_dict(), f"pretrained_models/ppo/oracle_{args.distractions}/critic.pth")