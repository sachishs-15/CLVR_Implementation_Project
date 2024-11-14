from pdb import set_trace
import torch
import torch.nn as nn
import argparse
from time import sleep
import cv2

from sprites_datagen.rewards import HorPosReward
from general_utils import AttrDict
from data.get_data import SpritesDataset
from model.encoder import ImageRecEncoder
from model.decoder import ImageRecDecoder

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def preprocess_data(data):
    for _, t in enumerate(data):  # loading to GPU
        t['images'] = torch.tensor(t['images']).to(device)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some numbers')
    parser.add_argument('--distractions', type=int, default=0)
    parser.add_argument('--dataset_size', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=20)

    args = parser.parse_args()

    spec = AttrDict(   # defining the specification for the dataset

            resolution=64,
            max_seq_len=40,
            max_speed=0.05,      # total image range [0, 1]
            obj_size=0.2,       # size of objects, full images is 1.0
            shapes_per_traj=2 + args.distractions,      # number of shapes per trajectory
            rewards=[HorPosReward],
        )
    

    dataset_size = args.dataset_size
    print("Reading data...")
    data = SpritesDataset(spec, dataset_size) # creating the dataset
    print("Data read.")

    preprocess_data(data) # preprocess the data to be loaded to the GPU
    print("Data transferred to GPU")

    Loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

    encoder = ImageRecEncoder().to(device)
    decoder = ImageRecDecoder().to(device)


    optimizer = torch.optim.RAdam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001, weight_decay=1e-5)

    for epoch in range(args.epochs):
        for i, t in enumerate(Loader):

            optimizer.zero_grad()

            #set_trace()
            inter = encoder(t['images'][0])
            prediction = decoder(inter)

            loss = nn.MSELoss()(prediction, t['images'][0])

            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item()}")
        
    print("Done")

    torch.save(encoder.state_dict(), f"pretrained_models/image_rec/encoder_{args.distractions}.pth")
    torch.save(decoder.state_dict(), f"pretrained_models/image_rec/decoder_{args.distractions}.pth")

    dataset_size = 5
    print("Reading data for testing...")
    data = SpritesDataset(spec, dataset_size) # creating the dataset
    print("Test data read!")

    preprocess_data(data)

    for i, t in enumerate(Loader):

        inter = encoder(t['images'][0])
        prediction = decoder(inter)

        img_gt = (t['images'][0].cpu().detach().numpy() + 1.0)* 255./ 2
        img_pd = (prediction.cpu().detach().numpy() + 1.0)* 255./ 2

        gt_seq = cv2.hconcat(img_gt.transpose(0, 2, 3, 1))
        pd_seq = cv2.hconcat(img_pd.transpose(0, 2, 3, 1))

        # vertically stack the images
        check = cv2.vconcat([gt_seq, pd_seq])
        cv2.imwrite(f"images/image_rec_{args.distractions}.jpg", check)

        sleep(5)

        

    