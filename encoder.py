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

isCuda = True if torch.cuda.is_available() else False
#isCuda = False
print(isCuda)

class LSTMEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, numOfActions, rollout_dim):
        super(LSTMEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rollout_dim = rollout_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim)

        self.hiddenToAction = nn.Linear(hidden_dim, numOfActions)
        self.init_hidden()

    def init_hidden(self,x=None):
        if x==None:
            self.hidden = (Variable(torch.zeros(1, 1, self.hidden_dim).cuda()),
                Variable(torch.zeros(1, 1, self.hidden_dim).cuda()))
        else:
            self.hidden = (Variable(x[0].data.cuda()), Variable(x[1].data.cuda()))

    def forward(self, sequence):
        lstm_out, self.hidden = self.lstm(sequence.view(self.rollout_dim, 1, -1), self.hidden)
        action_scores = self.hiddenToAction(lstm_out.view(self.rollout_dim, -1))
        self.init_hidden(self.hidden)
        return action_scores, self.hidden

model = LSTMEncoder(INPUT_SPACE_DIM, HIDDEN_DIM, ACTION_SPACE_DIM, ROLLOUT_SIZE)
if isCuda:
    model.cuda()

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# set up the training 
lossPerEpoch = []
for e in range(1, args.epochs+1):

    losses = []
    for rollout in rollouts:

        model.zero_grad()

        if len(rollout['observations']) < ROLLOUT_SIZE:
            continue

        data = torch.from_numpy(rollout['observations']).float().view(ROLLOUT_SIZE, 1, -1)
        target = torch.from_numpy(rollout['actions']).float().view(ROLLOUT_SIZE, -1)

        if isCuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
       
        # model.hidden = model.init_hidden()

        scores, _ = model(data)

        loss = loss_fn(scores, target)
        loss.backward()
        optimizer.step()
        
        losses.append(loss)

    lossPerEpoch.append(sum(losses)/len(losses))

    if e%args.epoch_step == 0:
        print('Train Epoch: {}/{}\t Loss: {:.5f}'.format(e, args.epochs, loss))

if args.save_model:
    filename = 'LSTMmodels/{}/inputs{}rolloutsize{}epochs{}.pkl'.format(args.env, NUM_OF_INPUTS, ROLLOUT_SIZE, args.epochs)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wb') as f:
        torch.save(model.state_dict(), f)
    print('Model written to file '+filename)

    filename = 'LSTMmodels/Loss/{}/inputs{}rolloutsize{}epochs{}.pkl'.format(args.env, NUM_OF_INPUTS, ROLLOUT_SIZE, args.epochs)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wb') as f:
        torch.save(lossPerEpoch, f) # save the decoder for generating rollouts
    print('Loss written to file '+filename)
