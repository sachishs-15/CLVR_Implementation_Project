import torch
import wandb
from pdb import set_trace
import argparse

from sprites_env.envs.sprites import SpritesStateEnv
from general_utils import AttrDict
from model.ppo import PPO_states

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some numbers')

    parser.add_argument('--run', type=str, default="PPO_States_Env")  
    parser.add_argument('--rollout_mem',  type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--mini_epochs', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--clip', type=int, default=0.2)
    parser.add_argument('--distractions', type=int, default=0)
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--log', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--save', type=int, default=0)

    args = parser.parse_args()

    print(args)
    
    data_spec = AttrDict(
            resolution=64,
            max_ep_len=40,
            max_speed=0.05,      
            obj_size=0.2,      
            follow=True,
        )
    
    env = SpritesStateEnv(n_distractors = args.distractions)
    env.set_config(data_spec)
    obs = env.reset()

    if args.log:
        wandb.init(project=f"Sprites-v{args.distractions}", name=args.run, config=args, save_code=True)

    ppo = PPO_states(env, args)
    ppo.train()

    if args.save:
        torch.save(ppo.actor.state_dict(), f"pretrained_models/ppo/oracle_{args.distractions}/actor.pth")
        torch.save(ppo.critic.state_dict(), f"pretrained_models/ppo/oracle_{args.distractions}/critic.pth")