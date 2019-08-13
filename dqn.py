import math
import random
import numpy as np
import os
from collections import deque
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import model as m
from atari_wrappers import wrap_deepmind, make_atari

# 1. GPU settings
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if gpu is to be used

# 3. environment reset
#env_name = 'Breakout'
#env_name = 'SpaceInvaders'
env_name = 'Riverraid'
#env_name = 'Seaquest'
#env_name = 'MontezumaRevenge'
env_raw = make_atari('{}NoFrameskip-v4'.format(env_name))
env = wrap_deepmind(env_raw, frame_stack=False, episode_life=True, clip_rewards=True)

c,h,w = m.fp(env.reset()).shape
n_actions = env.action_space.n
print(n_actions)

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
EVALUATE_FREQ = 200000
optimizer = optim.Adam(policy_net.parameters(), lr=0.0000625, eps=1.5e-4)

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

def evaluate(step, policy_net, device, env, n_actions, eps=0.05, num_episode=5):
    env = wrap_deepmind(env)
    sa = m.ActionSelector(eps, eps, policy_net, EPS_DECAY, n_actions, device)
    e_rewards = []
    q = deque(maxlen=5)
    for i in range(num_episode):
        env.reset()
        e_reward = 0
        for _ in range(10): # no-op
            n_frame, _, done, _ = env.step(0)
            n_frame = m.fp(n_frame)
            q.append(n_frame)

        while not done:
            state = torch.cat(list(q))[1:].unsqueeze(0)
            action, eps = sa.select_action(state, train)
            n_frame, reward, done, info = env.step(action)
            n_frame = m.fp(n_frame)
            q.append(n_frame)
            
            e_reward += reward
        e_rewards.append(e_reward)

    f = open("file.txt",'a') 
    f.write("%f, %d, %d\n" % (float(sum(e_rewards))/float(num_episode), step, num_episode))
    f.close()

q = deque(maxlen=5)
done = True
eps = 0
episode_len = 0

progressive = tqdm(range(NUM_STEPS), total=NUM_STEPS, ncols=50, leave=False, unit='b')
for step in progressive:
    if done: # life reset !!!
        env.reset()
        sum_reward = 0
        episode_len = 0
        img, _, _, _ = env.step(1) # BREAKOUT specific !!!
        for i in range(10): # no-op
            n_frame, _, _, _ = env.step(0)
            n_frame = m.fp(n_frame)
            q.append(n_frame)
        
    train = len(memory) > 50000
    # Select and perform an action
    state = torch.cat(list(q))[1:].unsqueeze(0)
    action, eps = sa.select_action(state, train)
    n_frame, reward, done, info = env.step(action)
    n_frame = m.fp(n_frame)

    # 5 frame as memory
    q.append(n_frame)
    memory.push(torch.cat(list(q)).unsqueeze(0), action, reward, done) # here the n_frame means next frame from the previous time step
    episode_len += 1

    # Perform one step of the optimization (on the target network)
    if step % POLICY_UPDATE == 0:
        optimize_model(train)
    
    # Update the target network, copying all weights and biases in DQN
    if step % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    if step % EVALUATE_FREQ == 0:
        evaluate(step, policy_net, device, env_raw, n_actions, eps=0.05, num_episode=15)


