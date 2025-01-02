import copy
import math
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class OUNoise:
    """Ornstein-Uhlenbeck process.
    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py
    """

    def __init__(
            self,
            size: int,
            mu: float = 0.0,
            theta: float = 0.15,
            sigma: float = 0.2,
            scale: float = 1.
    ):
        """Initialize parameters and noise process."""
        self.state = np.float64(0.0)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.scale = scale
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for _ in range(len(x))]
        )
        self.state = x + dx
        return self.scale * self.state


class GaussianNoise:
    def __init__(self, size, mu=0, sigma=0.1, scale=1.):
        self.size = size
        self.mu = mu
        self.std = sigma
        self.scale = scale

    def sample(self):
        return self.scale * np.random.normal(self.mu, self.std, self.size)


def initialize_uniformly(layer, init_w=3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

        initialize_uniformly(self.fc3)
        initialize_uniformly(self.fc1, 1 / math.sqrt(state_dim))
        initialize_uniformly(self.fc2, 1 / math.sqrt(400))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x)) * self.action_bound


class ValueNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400 + action_dim, 300)
        self.fc3 = nn.Linear(300, 1)

        initialize_uniformly(self.fc3)
        initialize_uniformly(self.fc1, 1 / math.sqrt(state_dim))
        initialize_uniformly(self.fc2, 1 / math.sqrt(400 + action_dim))

    def forward(self, x, a):
        x_s = F.relu(self.fc1(x))
        x = F.relu(self.fc2(torch.cat([x_s, a], dim=1)))
        return self.fc3(x)


class DDPG:
    """DDPGAgent interacting with environment.

        Attribute:
            actor (nn.Module): target actor model to select actions
            targe_actor (nn.Module): actor model to predict next actions
            actor_optimizer (Optimizer): optimizer for training actor
            critic (nn.Module): critic model to predict state values
            target_critic (nn.Module): target critic model to predict state values
            critic_optimizer (Optimizer): optimizer for training critic
            gamma (float): discount factor
            tau (float): parameter for soft target update
            noise (OUNoise): noise generator for exploration
    """

    def __init__(self, state_dim, action_dim, action_bound,
                 actor_lr, critic_lr, weight_decay, noise, tau, gamma, device, initial_random_steps):
        self.actor = PolicyNet(state_dim, action_dim, action_bound).to(device)
        self.critic = ValueNet(state_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, action_dim, action_bound).to(device)
        self.target_critic = ValueNet(state_dim, action_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=weight_decay)
        self.gamma = gamma
        self.noise = noise
        self.action_bound = action_bound
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device
        self.initial_random_steps = initial_random_steps
        self.total_step = 0

    def take_action(self, state, eval=False):
        if self.total_step < self.initial_random_steps and not eval:
            action = env.action_space.sample()
        else:
            state = torch.FloatTensor([state]).to(self.device)
            action = self.actor(state).squeeze().detach().cpu().numpy()
            if not eval:
                action = np.clip(action + self.noise.sample(),
                                 -self.action_bound, self.action_bound)
        self.total_step += 1
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = F.mse_loss(self.critic(states, actions), q_targets.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self, folder='models'):
        torch.save(self.actor.state_dict(), folder + "/ddpg_actor")
        torch.save(self.critic.state_dict(), folder + "/ddpg_critic")

    def load_actor(self, folder='models'):
        self.actor.load_state_dict(torch.load(folder + "/ddpg_actor"))

    def visual(self):
        self.load_actor()
        utils.visualization(self, env_name)


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alg_name = 'DDPG'
actor_lr = 1e-4
critic_lr = 1e-3
weight_decay = 1e-2
num_episodes = 5000
hidden_dim = 256
gamma = 0.99
tau = 0.001  # 软更新参数
buffer_size = 1e6
initial_random_steps = 10000  # Time steps initial random policy is used
minimal_size = 10000  # Update begin step
batch_size = 64
update_interval = 1

env_name = 'Reacher-v5'
env_name = 'Hopper-v5'
env_name = 'Humanoid-v5'
env_name = 'Walker2d-v5'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

# noise = OUNoise(action_dim, theta=0.15, sigma=0.2, scale=action_bound)
noise = GaussianNoise(action_dim, mu=0, sigma=0.1, scale=action_bound)

replay_buffer = utils.ReplayBuffer(buffer_size)
agent = DDPG(state_dim, action_dim, action_bound,
             actor_lr, critic_lr, weight_decay,
             noise, tau, gamma, device, initial_random_steps)

if __name__ == '__main__':
    print(env_name)
    return_list = utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer,
                                               minimal_size, batch_size, update_interval,
                                               save_model=True)
    utils.dump(f'./results/{alg_name}.pkl', return_list)
    utils.show(f'./results/{alg_name}.pkl', alg_name)
