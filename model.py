import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, h, w, outputs, device):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, outputs)
        self.device = device
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)
        
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            #m.bias.data.fill_(0.1)
    
    def forward(self, x):
        x = x.to(self.device).float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)

class ActionSelector(object):
    def __init__(self, INITIAL_EPSILON, FINAL_EPSILON, policy_net, EPS_DECAY, n_actions, device):
        self._eps = INITIAL_EPSILON
        self._FINAL_EPSILON = FINAL_EPSILON
        self._INITIAL_EPSILON = INITIAL_EPSILON
        self._policy_net = policy_net
        self._EPS_DECAY = EPS_DECAY
        self._n_actions = n_actions
        self._device = device

    def select_action(self, state, training=False):
        sample = random.random()
        if training:
            self._eps -= (self._INITIAL_EPSILON - self._FINAL_EPSILON)/self._EPS_DECAY
            self._eps = max(self._eps, self._FINAL_EPSILON)
        if sample > self._eps:
            with torch.no_grad():
            	a = self._policy_net(state).max(1)[1].cpu().view(1,1)
        else:
            a = torch.tensor([[random.randrange(self._n_actions)]], device='cpu', dtype=torch.long)
        
        return a.numpy()[0,0].item(), self._eps

class ReplayMemory(object):
    def __init__(self, capacity, state_shape, n_actions, device):
        c,h,w = state_shape
        self.capacity = capacity
        self.device = device
        self.m_states = torch.zeros((capacity, c, h, w), dtype=torch.uint8)
        self.m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, done):
        """Saves a transition."""
        self.m_states[self.position] = state # 5,84,84
        self.m_actions[self.position,0] = action
        self.m_rewards[self.position,0] = reward
        self.m_dones[self.position,0] = done
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)

    def sample(self, bs):
        i = torch.randint(0, high=self.size, size=(bs,))
        bs = self.m_states[i, :4] 
        bns = self.m_states[i, 1:] 
        ba = self.m_actions[i].to(self.device)
        br = self.m_rewards[i].to(self.device).float()
        bd = self.m_dones[i].to(self.device).float()
        return bs, ba, br, bns, bd

    def __len__(self):
        return self.size

def fp(n_frame):
    n_frame = torch.from_numpy(n_frame)
    h = n_frame.shape[-2]
    return n_frame.view(1,h,h)

class FrameProcessor():
    def __init__(self, im_size=84):
        self.im_size = im_size

    def process(self, frame):
        im_size = self.im_size
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame[46:160+46, :]

        frame = cv2.resize(frame, (im_size, im_size), interpolation=cv2.INTER_LINEAR)
        frame = frame.reshape((1, im_size, im_size))

        x = torch.from_numpy(frame)
        return x
