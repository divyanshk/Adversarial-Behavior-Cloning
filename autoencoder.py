'''

LSTM based encoder-decoder model

Author: Divyansh Khanna

'''

import os
import torch
import pickle
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.autograd import Variable

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--epoch_step', type=int, default=10)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--rollout_size', type=int, default=200)
parser.add_argument('--obs_shape', type=int, required=True)
parser.add_argument('--actions_shape', type=int, required=True)
parser.add_argument('--save_model', type=bool, required=True, default=True)
args = parser.parse_args()

with open('rollouts/'+args.env+'.pkl', 'rb') as f:
    rollouts = pickle.load(f)

NUM_OF_INPUTS = len(rollouts)
ROLLOUT_SIZE = min(args.rollout_size, len(rollouts[0]['observations']))
INPUT_SPACE_DIM = args.obs_shape #env.observation_space.shape[0]
ACTION_SPACE_DIM = args.actions_shape#env.action_space.shape[0]
HIDDEN_DIM = args.hidden_dim

print("Number of rollouts are {}".format(NUM_OF_INPUTS))
print("Size of rollouts are {}".format(ROLLOUT_SIZE))
print("Input space dim is {}".format(INPUT_SPACE_DIM))
print("Action space dim is {}".format(ACTION_SPACE_DIM))
print("Hidden dim is {}".format(HIDDEN_DIM))

lr = 1e-4
batch_size = 1
latent_space_dim = HIDDEN_DIM
#torch.backends.cudnn.enabled=False
isCuda = True if torch.cuda.is_available() else False
#isCuda = False
print(isCuda)

class Disc(nn.Module):

    def __init__(self):
        super(Disc, self).__init__()
        self.fc1 = nn.Linear(latent_space_dim, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.sigmoid(self.fc3(x))

class LSTMEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, numOfActions, rollout_dim):
        super(LSTMEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rollout_dim = rollout_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.init_hidden()

    def init_hidden(self,x=None):
        if isCuda:
            if x==None:
                self.hidden = (Variable(torch.zeros(1, 1, self.hidden_dim).cuda()),
                    Variable(torch.zeros(1, 1, self.hidden_dim).cuda()))
            else:
                self.hidden = (Variable(x[0].data.cuda()), Variable(x[1].data.cuda()))
        else:
            if x==None:
                self.hidden = (Variable(torch.zeros(1, 1, self.hidden_dim)),
                    Variable(torch.zeros(1, 1, self.hidden_dim)))
            else:
                self.hidden = (Variable(x[0].data), Variable(x[1].data))

    def forward(self, sequence):
        lstm_out, self.hidden = self.lstm(sequence.view(self.rollout_dim, 1, -1), self.hidden)
        self.init_hidden(self.hidden)
        return lstm_out, self.hidden

class LSTMDecoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, numOfActions, rollout_dim):
        super(LSTMDecoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rollout_dim = rollout_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim)

        self.hiddenToAction = nn.Linear(hidden_dim, numOfActions)
        self.init_hidden()

    def init_hidden(self,x=None):
        if isCuda:
            if x==None:
                self.hidden = (Variable(torch.zeros(1, 1, self.hidden_dim).cuda()),
                    Variable(torch.zeros(1, 1, self.hidden_dim).cuda()))
            else:
                self.hidden = (Variable(x[0].data.cuda()), Variable(x[1].data.cuda()))
        else:
            if x==None:
                self.hidden = (Variable(torch.zeros(1, 1, self.hidden_dim)),
                    Variable(torch.zeros(1, 1, self.hidden_dim)))
            else:
                self.hidden = (Variable(x[0].data), Variable(x[1].data))

    def forward(self, sequence):
        lstm_out, self.hidden = self.lstm(sequence.view(self.rollout_dim, 1, -1), self.hidden)
        action_scores = self.hiddenToAction(lstm_out.view(self.rollout_dim, -1))
        self.init_hidden(self.hidden)
        return action_scores, self.hidden

enc = LSTMEncoder(INPUT_SPACE_DIM, HIDDEN_DIM, ACTION_SPACE_DIM, ROLLOUT_SIZE)
dec = LSTMDecoder(INPUT_SPACE_DIM, HIDDEN_DIM, ACTION_SPACE_DIM, ROLLOUT_SIZE)
disc = Disc()
if isCuda:
    enc.cuda()
    dec.cuda()
    disc.cuda()

optimizerE = optim.Adam(enc.parameters(), lr=lr) # enocder trying to learn from discriminator
optimizerD = optim.Adam(dec.parameters(), lr=lr) # autoencoder loss
optimizerDisc = optim.Adam(disc.parameters(), lr=lr) # discriminator loss

lossPerEpoch = []
for e in range(0, args.epochs+1):

    for rollout in rollouts:

        if len(rollout['observations']) < ROLLOUT_SIZE:
            continue

        enc.train()
        dec.train()
        disc.train()

        enc.zero_grad()
        dec.zero_grad()
        disc.zero_grad()

        data = torch.from_numpy(rollout['observations'][:ROLLOUT_SIZE]).float().view(ROLLOUT_SIZE, 1, -1)
        target = torch.from_numpy(rollout['actions'][:ROLLOUT_SIZE]).float().view(ROLLOUT_SIZE, -1)

        if isCuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        _, latent = enc(data)
        
        dec.init_hidden(latent) # init the decoder with the hidden layer of encoder (latent tensor)
        scores, _ = dec(data)
        
        recon_loss = F.mse_loss(scores, target)
        recon_loss.backward(retain_graph=False)
        optimizerE.step()
        optimizerD.step()

        trueLabel = torch.ones(batch_size, 1)
        falseLabel = torch.zeros(batch_size, 1)
        trueSample = torch.randn(batch_size, latent_space_dim)
        if isCuda:
            trueLabel, falseLabel, trueSample = trueLabel.cuda(), falseLabel.cuda(), trueSample.cuda()
        trueLabel, falseLabel, trueSample = Variable(trueLabel), Variable(falseLabel), Variable(trueSample)

        Disc_real_loss = F.binary_cross_entropy(disc(trueSample), trueLabel)
        _, (latent, _) = enc(data)
        Disc_fake_loss = F.binary_cross_entropy(disc(latent.detach()), falseLabel)
        Disc_loss = Disc_real_loss + Disc_fake_loss
        Disc_loss.backward()
        optimizerDisc.step()

        _, (latent, _) = enc(data)
        enc_loss = F.binary_cross_entropy(disc(latent), trueLabel)
        enc_loss.backward()
        optimizerE.step()

    lossPerEpoch.append(recon_loss.data.cpu().numpy())

    if e%args.epoch_step == 0:
        print('Train Epoch: {}/{}\t Reconstruction Loss: {:.5f}, Discriminator Loss: {:.5f}, Encoder Loss: {:.5f}'.format(e, args.epochs, recon_loss, Disc_loss, enc_loss))

if args.save_model:
    filename = 'AAEmodels/{}/inputs{}rolloutsize{}epochs{}.pkl'.format(args.env, NUM_OF_INPUTS, ROLLOUT_SIZE, args.epochs)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wb') as f:
        torch.save(dec.state_dict(), f) # save the decoder for generating rollouts
    print('Model written to file '+filename)

    filename = 'AAEmodels/Loss/{}/inputs{}rolloutsize{}epochs{}.pkl'.format(args.env, NUM_OF_INPUTS, ROLLOUT_SIZE, args.epochs)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wb') as f:
        pickle.dump(lossPerEpoch, f) # save the losses
    print('Loss written to file '+filename)
