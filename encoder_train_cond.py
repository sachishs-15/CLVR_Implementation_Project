
# store the encoder_cond_{distractions}.pth file in the pretrained_models folder

import torch
import cv2
import numpy as np
import torch.nn as nn
import wandb
from time import sleep 
from pdb import set_trace
import argparse

from sprites_datagen.rewards import VertPosReward, HorPosReward, AgentXReward, AgentYReward, TargetXReward, TargetYReward
from general_utils import AttrDict

from data.get_data import SpritesDataset
from model.encoder import Encoder, LSTM, RewardMLP, MLP
from model.decoder import Decoder


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def eval(encoder, decoder, Loader, reward_fn):

    for i, t in enumerate(Loader):

        encoding = encoder(t['images'][0])
        prediction = decoder(encoding)

        img_gt = (t['images'][0].cpu().numpy() + 1.0)* 255./ 2
        img_pd = (prediction.cpu().detach().numpy() + 1.0)* 255./ 2
        
        gt_seq = cv2.hconcat(img_gt.transpose(0, 2, 3, 1))
        pd_seq = cv2.hconcat(img_pd.transpose(0, 2, 3, 1))

        # vertically stack the images
        check = cv2.vconcat([gt_seq, pd_seq])
        cv2.imwrite(f"images/{reward_fn}.jpg", check)

        sleep(5)

def train():
    pass

def preprocess_data(data):
    for _, t in enumerate(data):  # loading to GPU
        t['images'] = torch.tensor(t['images']).to(device)

        for k, v in t['rewards'].items():
            t['rewards'][k] = torch.tensor(v).to(device)

def process_rewards(input):
    
    if input == 'VertPosReward':
        return [VertPosReward]
    elif input == 'HorPosReward':
        return [HorPosReward]
    elif input == 'all':
        return [AgentXReward, AgentYReward, TargetXReward, TargetYReward]
    else:
        return [VertPosReward]
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Rewards')
    parser.add_argument('--rewards', type=str, default='VertPosReward', help='Reward function to use. Default: VertPosReward')
    parser.add_argument('--save', type=bool, default=False, help='Save the model. Default: False')
    parser.add_argument('--shapes', type=int, default=1, help='Number of shapes per trajectory. Default: 1')
    args = parser.parse_args()

    rewards_spec = process_rewards(args.rewards)
    sprites_no = args.shapes
    total_frames = 40
    conditioning_frames = 10
    dataset_size = 5000 


    spec = AttrDict(   # defining the specification for the dataset

            resolution=64,
            max_seq_len=total_frames,
            max_speed=0.05,      # total image range [0, 1]
            obj_size=0.2,       # size of objects, full images is 1.0
            shapes_per_traj=sprites_no,      # number of shapes per trajectory
            rewards=rewards_spec,
        )

    #wandb.init(project="Enc-Dec", name = f"Encoder-Decoder-Cond_{args.rewards}_{args.shapes-2}")

    print("Reading data...")
    data = SpritesDataset(spec, dataset_size) # creating the dataset
    print("Data read.")

    preprocess_data(data) # preprocess the data to be loaded to the GPU
    print("Data transferred to GPU")

    Loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)  

    heads = len(rewards_spec)

    encoder = Encoder(spec.resolution, 1, 64).to(device)
    mlp = MLP(64, 32, 64).to(device)
    lstm = LSTM(64, 256, 64, conditioning_frames).to(device)
    rewardmlp = RewardMLP(64, 32, heads).to(device)
    decoder = Decoder(64, 1, 64).to(device)

    optimizer = torch.optim.RAdam(list(encoder.parameters()) + list(mlp.parameters()) + list(lstm.parameters()) + list(rewardmlp.parameters()), lr=0.001, betas=(0.9, 0.999))    
    d_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)

    epoch = 20
    loss_val = 0
    d_loss_val = 0

    for i in range(epoch):            
        print("Epoch: ", i)
            
        for x, t in enumerate(Loader):

            optimizer.zero_grad()
            d_optimizer.zero_grad()
            
            #set_trace()
            encodings = encoder(t['images'][0])
            lstm_input = mlp(encodings)
            lstm_out = lstm(lstm_input.unsqueeze(0))

            predictions = rewardmlp(lstm_out)

            rew_gt = torch.cat([t['rewards'][k].transpose(0,1)[conditioning_frames:] for k in t['rewards'].keys()], dim=1) # ground truth
            
            #print("Predictions: ", predictions.transpose(0,1))
            #print("Ground Truth: ", rew_gt.transpose(0,1))
            
            decodings = decoder(encodings.detach())

            d_lossfn = nn.MSELoss()
            lossfn = nn.MSELoss()

            loss = lossfn(predictions, rew_gt)
            d_loss = d_lossfn(decodings, t['images'][0])

            loss_val += loss.item()
            d_loss_val += d_loss.item()

            if((x+1)%10 == 0):
                
                print("Enc Loss: ", loss_val/10)
                #wandb.log({"Encoder Loss": loss_val/10})
                #print("Dec Loss: ", d_loss_val/10)
                #wandb.log({"Decoder Loss": d_loss_val/10})

                loss_val = 0
                d_loss_val = 0

            loss.backward()
            d_loss.backward()

            optimizer.step()
            d_optimizer.step()

    if args.save:
        torch.save(encoder.state_dict(), f"pretrained_models/encoder_{args.shapes-2}.pth") # save the model
        torch.save(decoder.state_dict(), f"pretrained_models/decoder_{args.shapes-2}.pth")

    eval(encoder, decoder, Loader, args.rewards)