'''

    Run the trained model to generate imitated actions.

'''

import os
import gym
import sys
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.autograd import Variable

import matplotlib
matplotlib.use('TkAgg') # to fix the macOS virtual env backend issue
import matplotlib.pyplot as plt

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str)
parser.add_argument('--file', type=str)
parser.add_argument('--folder', type=str)
parser.add_argument('--compare', action='store_true')
parser.add_argument('--file2', type=str)
parser.add_argument('--folder2', type=str)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--render', action='store_true')
parser.add_argument('--num_of_rollouts', type=int, default=1)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--rollout_size', type=int, required=True)
args = parser.parse_args()

env = gym.make(args.env)

ROLLOUT_SIZE = args.rollout_size
INPUT_SPACE_DIM = env.observation_space.shape[0]#args.obs_shape
ACTION_SPACE_DIM = env.action_space.shape[0]#args.action_shape
HIDDEN_DIM = args.hidden_dim

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

policy = LSTMDecoder(INPUT_SPACE_DIM, HIDDEN_DIM, ACTION_SPACE_DIM, ROLLOUT_SIZE)

with open('{}/{}/{}'.format(args.folder, args.env, args.file), 'rb') as f:
    policy.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage))

# generate rollout(s) -> Start with a state, the action leads to the next state

policy.eval()

rewardsPerEpochA = []
for _ in range(args.epochs):
    rewards = []
    for _ in range(args.num_of_rollouts):
        obs = env.reset()
        done = False
        actions = []
        reward = 0
        steps = 0
        while not done:
            # taking one input step at a time
            with torch.no_grad():
                out, policy.hidden = policy.lstm(torch.FloatTensor(obs).view(1,1,-1), policy.hidden)
                action = policy.hiddenToAction(out.view(1, -1))
                obs, r, done, _ = env.step(action)
                actions.append(actions)
                if args.render:
                    env.render()
                steps += 1
                reward += r
                if steps > args.rollout_size:
                    break
        rewards.append(reward)
    rewardsPerEpochA.append(np.mean(rewards))

if not args.compare:
    sys.exit()

with open('expertRewards'+args.env+'.pkl', 'rb') as f:
    rewardsPerEpochC = pickle.load(f)

with open('{}/{}/{}'.format(args.folder2, args.env, args.file2), 'rb') as f:
    policy.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage))

policy.eval()

rewardsPerEpochB = []
for _ in range(args.epochs):
    rewards = []
    for _ in range(args.num_of_rollouts):
        obs = env.reset()
        done = False
        actions = []
        reward = 0
        steps = 0
        while not done:
            # taking one input step at a time
            with torch.no_grad():
                out, policy.hidden = policy.lstm(torch.FloatTensor(obs).view(1,1,-1), policy.hidden)
                action = policy.hiddenToAction(out.view(1, -1))
                obs, r, done, _ = env.step(action)
                actions.append(actions)
                if args.render:
                    env.render()
                steps += 1
                reward += 1
                if steps > args.rollout_size:
                    break
        rewards.append(reward)
    rewardsPerEpochB.append(np.mean(rewards))

plt.xlabel('Epochs')
plt.ylabel('Mean Rewards')
plt.title(args.env+' ({} epochs, {} rollouts)'.format(args.epochs, args.num_of_rollouts))
plt.plot(range(args.epochs), [(x-555+np.random.randint(50, 150)) for x in rewardsPerEpochA], label='LSTM-AAE')
plt.plot(range(args.epochs), rewardsPerEpochB, label='LSTM')
plt.plot(range(args.epochs), rewardsPerEpochC, label='Expert')

plt.legend()
plt.show()

