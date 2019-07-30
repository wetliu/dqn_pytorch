import gym
import math
import random
import numpy as np
import os
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import model as m

# 1. GPU settings
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if gpu is to be used

# 2. FrameProcessor
fp = m.FrameProcessor()

# 3. environment reset
env = gym.envs.make('BreakoutDeterministic-v4')
#env = gym.envs.make('MontezumaRevengeDeterministic-v4')
lives = env.unwrapped.ale.lives() + 1
c,h,w = fp.process(env.reset()).shape
n_actions = 4

# 4. Network reset
policy_net = m.DQN(h, w, n_actions, device).to(device)
target_net = m.DQN(h, w, n_actions, device).to(device)
policy_net.apply(policy_net.init_weights)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# 5. DQN hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 1000000
TARGET_UPDATE = 10000
NUM_STEPS = 50000000
M_SIZE = 1000000
POLICY_UPDATE = 4
optimizer = optim.Adam(policy_net.parameters(), lr=1e-5)

# replay memory and action selector
memory = m.ReplayMemory(M_SIZE, [5,h,w], n_actions, device)
sa = m.ActionSelector(EPS_START, EPS_END, policy_net, EPS_DECAY, n_actions, device)

steps_done = 0

def optimize_model(train):
    if not train:
        return
    state_batch, action_batch, reward_batch, n_state_batch, done_batch = memory.sample(BATCH_SIZE)

    q = policy_net(state_batch).gather(1, action_batch)
    nq = target_net(n_state_batch).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (nq * GAMMA)*(1.-done_batch[:,0]) + reward_batch[:,0]

    # Compute Huber loss
    loss = F.smooth_l1_loss(q, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

q = deque(maxlen=5)
done = True
eps = 0
sum_reward = 0
episode_len = 0
for step in range(NUM_STEPS):
    if done: # life reset !!!
        env.reset()
        lives = env.unwrapped.ale.lives()
        print('%d (%d), reward: %d, eps: %.3f, episode len: %d' % (step, len(memory), sum_reward, eps, episode_len))
        sum_reward = 0
        episode_len = 0
        img, _, _, _ = env.step(1) # BREAKOUT specific !!!
        for i in range(10): # no-op
            img, _, _, _ = env.step(0)
            n_frame = fp.process(img)
            q.append(n_frame)
        
    train = len(memory) > 50000
    # Select and perform an action
    state = torch.cat(list(q))[1:].unsqueeze(0)
    action, eps = sa.select_action(state, train)
    img, reward, done, info = env.step(action)
    # reward
    sum_reward += reward
        
    reward = min(max(-1,reward), 1) # reward clipping !!!

    # 5 frame as memory
    n_frame = fp.process(img)
    q.append(n_frame)
    memory.push(torch.cat(list(q)).unsqueeze(0), action, reward, done or lives > env.unwrapped.ale.lives()) # here the n_frame means next frame from the previous time step
    if lives > env.unwrapped.ale.lives(): # env changed
        lives = env.unwrapped.ale.lives()
    episode_len += 1

    # Perform one step of the optimization (on the target network)
    if step % POLICY_UPDATE == 0:
        optimize_model(train)
    
    # Update the target network, copying all weights and biases in DQN
    if step % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
