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
isCuda = True if torch.cuda.is_available() else False

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
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sequence):
        lstm_out, self.hidden = self.lstm(sequence.view(self.rollout_dim, 1, -1), self.hidden)
        return lstm_out, self.hidden

class LSTMDecoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, numOfActions, rollout_dim):
        super(LSTMDecoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rollout_dim = rollout_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim)

        self.hiddenToAction = nn.Linear(hidden_dim, numOfActions)
        self.hidden = self.init_hidden()

    def init_hidden(self, hidden=None):
        # The axes semantics are (num_layers, minibatch_# gesize, hidden_dim)
        if not hidden:
            return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))
        else:
            return (hidden, torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sequence):
        lstm_out, self.hidden = self.lstm(sequence.view(self.rollout_dim, 1, -1), self.hidden)
        action_scores = self.hiddenToAction(lstm_out.view(self.rollout_dim, -1))
        return action_scores, self.hidden

enc = LSTMEncoder(INPUT_SPACE_DIM, HIDDEN_DIM, ACTION_SPACE_DIM, ROLLOUT_SIZE)
dec = LSTMDecoder(INPUT_SPACE_DIM, HIDDEN_DIM, ACTION_SPACE_DIM, ROLLOUT_SIZE)
disc = Disc()
if isCuda:
    enc.cuda()
    dec.cuda()
    disc.cuda()

optimizerE = optim.Adam(enc.parameters(), lr=lr) # enocder trying to learn from discriminator
optimizerAE = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr) # autoencoder loss
optimizerD = optim.Adam(disc.parameters(), lr=lr) # discriminator loss

for e in range(0, args.epochs+1):

    for rollout in rollouts:

        # TODO: variable rollout size
        if len(rollout['observations']) < ROLLOUT_SIZE:
            continue

        enc.zero_grad()
        dec.zero_grad()
        disc.zero_grad()

        data = Variable(torch.from_numpy(rollout['observations'][:ROLLOUT_SIZE]).float().view(ROLLOUT_SIZE, 1, -1))
        target = Variable(torch.from_numpy(rollout['actions'][:ROLLOUT_SIZE]).float().view(ROLLOUT_SIZE, -1))

        if isCuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data.view(batch_size, -1)), Variable(target)

        optimizerAE.zero_grad()
        _, (latent, _) = enc(data)
        dec.hidden = dec.init_hidden(latent) # init the decoder with the hidden layer of encoder (latent tensor)
        scores, _ = dec(data)
        loss = F.mse_loss(scores, target)
        loss.backward(retain_graph=True)
        optimizerAE.step()

        trueLabel = torch.ones(batch_size, 1)
        falseLabel = torch.zeros(batch_size, 1)
        trueSample = torch.randn(batch_size, latent_space_dim)
        if isCuda:
            trueLabel, falseLabel, trueSample = trueLabel.cuda(), falseLabel.cuda(), trueSample.cuda()
        trueLabel, falseLabel, trueSample = Variable(trueLabel), Variable(falseLabel), Variable(trueSample)

        D_real_loss = F.binary_cross_entropy(disc(trueSample), trueLabel)
        D_real_loss.backward()
        D_fake_loss = F.binary_cross_entropy(disc(latent.detach()), falseLabel)
        D_fake_loss.backward()
        optimizerD.step()

        enc_loss = F.binary_cross_entropy(disc(latent.detach()), trueLabel)
        enc_loss.backward()
        optimizerE.step()

    if e%args.epoch_step == 0:
        print('Train Epoch: {}/{}\t AutoEncoder Loss: {:.3f}, AE Loss: {:.3f}, Encoder Loss: {:.3f}'.format(e, args.epochs, loss, (D_fake_loss+D_real_loss), enc_loss))

if args.save_model:
    filename = 'AAEmodels/{}/inputs{}rolloutsize{}epochs{}.pkl'.format(args.env, NUM_OF_INPUTS, ROLLOUT_SIZE, args.epochs)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wb') as f:
        torch.save(dec.state_dict(), f) # save the decoder for generating rollouts
    print('Model written to file '+filename)

# TODO: is the AAE correct ?
# TODO: generate rollouts
# TODO: pretraining using supervised learning on the encoder
