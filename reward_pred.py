import torch
import torch.nn as nn
import argparse
import wandb

from sprites_env.envs.sprites import SpritesEnv
from general_utils import AttrDict
from model.ppo import PPO_images
from model.encoder import Encoder

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some numbers')

    parser.add_argument('--run', type=str, default="reward_prediction")
    parser.add_argument('--rollout_mem',  type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--mini_epochs', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--clip', type=int, default=0.2)
    parser.add_argument('--distractions', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--log', type=int, default=1)
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--save', type=int, default=0)

    args = parser.parse_args()
    
    data_spec = AttrDict(
            resolution=64,
            max_ep_len=40,
            max_speed=0.05,      
            obj_size=0.2,       
            follow=True,
        )
    
    encoder = Encoder(data_spec.resolution, 1)
    path = f"pretrained_models/encoder_cond_{args.distractions}.pth"
    encoder_dict = torch.load(path)

    env = SpritesEnv(n_distractors = args.distractions)
    env.set_config(data_spec)
    obs = env.reset()

    if args.log:
        wandb.init(project=f"Sprites-v{args.distractions}", name=args.run, config=args)

    ppo = PPO_images(env, encoder, args)
    ppo.train()
    
    if args.save:
            ft = "frozen" if args.freeze else "ft"
            torch.save(args.encoder.state_dict(), f"pretrained_models/ppo/reward_pred_{ft}_{args.env.n_distractors}/encooder.pth")
            torch.save(args.actor.state_dict(), f"pretrained_models/ppo/reward_pred_{ft}_{args.env.n_distractors}/actor.pth")
            torch.save(args.critic.state_dict(), f"pretrained_models/ppo/reward_pred_{ft}_{args.env.n_distractors}/critc.pth")