# DRL_001_Navigation

## Learning Algorithm

The agent used in this project is based upon a [Deep Q-Network](https://en.wikipedia.org/wiki/Q-learning#Variants) with random replay memory and fixed q-targets using an [Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) Learning Rate Optimization. 
(See [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980.pdf) for reference.)

  1.  Replay Memory/Buffer used to record agent experience so as to be able to randomly sample for additional training, which tends to be especially useful in rarely occuring states.

  2.  Fixed Q-Targerts are used since updating based only on a guess can lead to harmful correlations.

***   ***   ***   ***   ***   ***   ***   ***   ***   

## Hyperparameters
```python
BUFFER_SIZE = int(1e5)  # Replay Buffer Size
BATCH_SIZE = 64         # Minibatch Size
GAMMA = 0.99            # Discount Factor
TAU = 1e-3              # Soft Update Of Target Parameters
LR = 5e-4               # Learning Rate
UPDATE_EVERY = 4        # How Oft' To Update Network

eps_start = 1.0         # Epsilon Start Value
eps_end = 0.1           # Epsilon End Value
eps_decay = 0.995       # Rate At Which To Decay Epsilon

n_episodes = 2000       # Max Number Of Episodes
max_t = 1000            # Max Time Step
```

## Model Architecture

Input => FC => FC => FC => Ouput

The Deep Q-Network is rather straight-foward, containing only two hidden layers and [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) (Rectified Linear Unit) activation function.

***   ***   ***   ***   ***   ***   ***   ***   ***


## Plot Of Rewards

![Rewards](/images/scores.png)

***   ***   ***   ***   ***   ***   ***   ***   ***


## Ideas for Future Work

Implement Prioritized Experience Replay so as to see if the agent can/will learn more effectively from the uniform sampling of the replay buffer in attempts to, possibly, highlight important state transitions.

Implement Double DQN network architecture to lessen the tendency of overestimating action values.
