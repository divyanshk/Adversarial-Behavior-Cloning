'''

LSTM based encoder-decoder model

Author: Divyansh Khanna

'''

import os
import gym
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
parser.add_argument('--epochStep', type=int, default=10)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--save_model', action='store_true')
args = parser.parse_args()

env = gym.make(args.env)

with open('rollouts/'+args.env+'.pkl', 'rb') as f:
    rollouts = pickle.load(f)

NUM_OF_INPUTS = len(rollouts)
ROLLOUT_SIZE = len(rollouts[0]['observations'])
INPUT_SPACE_DIM = env.observation_space.shape[0]
ACTION_SPACE_DIM = env.action_space.shape[0]
HIDDEN_DIM = args.hidden_dim

print("Number of rollouts are {}".format(NUM_OF_INPUTS))
print("Size of rollouts are {}".format(ROLLOUT_SIZE))
print("Input space dim is {}".format(INPUT_SPACE_DIM))
print("Action space dim is {}".format(ACTION_SPACE_DIM))
print("Hidden dim is {}".format(HIDDEN_DIM))

class LSTMEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, numOfActions, rollout_dim):
        super(LSTMEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rollout_dim = rollout_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim)

        self.hiddenToAction = nn.Linear(hidden_dim, numOfActions)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sequence):
        lstm_out, self.hidden = self.lstm(sequence.view(self.rollout_dim, 1, -1), self.hidden)
        action_scores = self.hiddenToAction(lstm_out.view(self.rollout_dim, -1))
        return action_scores, self.hidden

model = LSTMEncoder(INPUT_SPACE_DIM, HIDDEN_DIM, ACTION_SPACE_DIM, ROLLOUT_SIZE)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# set up the training
for e in range(1, args.epochs+1):
    for rollout in rollouts:

        model.zero_grad()

        data = Variable(torch.from_numpy(rollout['observations']).float().view(ROLLOUT_SIZE, 1, -1))
        target = Variable(torch.from_numpy(rollout['actions']).float().view(ROLLOUT_SIZE, -1))

        model.hidden = model.init_hidden()

        scores, _ = model(data)

        loss = loss_fn(scores, target)
        loss.backward()
        optimizer.step()

    if e%args.epochStep == 0:
        print('Train Epoch: {}/{}\t Loss: {:.3f}'.format(e, args.epochs, loss))

if args.save_model:
    filename = 'models/{}/inputs{}rolloutsize{}epochs{}.pkl'.format(args.env, NUM_OF_INPUTS, ROLLOUT_SIZE, args.epochs)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wb') as f:
        torch.save(model, f)
