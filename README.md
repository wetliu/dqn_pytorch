# dqn_pytorch
## Introduction
This implementation follows the benchmark and implementations of DQN. The training steps are bounded by 50M with epsilon is 0.1. Testing epsilon is 0.05.

## Usage
```
python dqn.py
```
You are able to change the env in dqn.py

## Benchmark Results
Testing score is calculated by averaging 15 testing episodes every 200,000 training steps, and is compared with the best scores reported in the paper - [DQN](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf). After the paper score is reached, the game is terminated. 

 Env | Best testing score | # of training steps when best score reached
:--:|:--:|:--:
 Breakout | 403.666667 | 19000000 (19M)
 SpaceInvaders | 1996.333333 | 11600000 (~12M)
 Seaquest | 7274.666667 | 8600000 (8.6M)
 RiverRaid | 8382.000000 | 6400000 (6.4M)

## Requirements

* **python 3.6**
* **gym**
* **cv2**
* **pytorch-1.1.0** 
* **numpy**
* **35+GB memory**
