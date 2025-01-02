import math
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
        Attributes:
            input_dim (int): input size of linear module
            output_dim (int): output size of linear module
            std_init (float): initial std value
            weight_mu (nn.Parameter): mean value weight parameter
            weight_sigma (nn.Parameter): std value weight parameter
            bias_mu (nn.Parameter): mean value bias parameter
            bias_sigma (nn.Parameter): std value bias parameter

        """

    def __init__(self, input_dim, output_dim, std_init=0.5):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.weight_sigma = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.register_buffer('weight_epsilon', torch.Tensor(output_dim, input_dim))

        self.bias_mu = nn.Parameter(torch.Tensor(output_dim))
        self.bias_sigma = nn.Parameter(torch.Tensor(output_dim))
        self.register_buffer('bias_epsilon', torch.Tensor(output_dim))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.input_dim)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.input_dim)
        )

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.fill_(
            self.std_init / math.sqrt(self.output_dim)
        )

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.input_dim)
        epsilon_out = self.scale_noise(self.output_dim)

        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def scale_noise(size):
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        if self.training:
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon
            )
        else:
            return F.linear(
                x,
                self.weight_mu,
                self.bias_mu
            )


class NoisyQnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.nl1 = NoisyLinear(hidden_dim, hidden_dim)
        self.nl2 = NoisyLinear(hidden_dim, action_dim)

    def reset_noise(self):
        """Reset all noisy layers."""
        self.nl1.reset_noise()
        self.nl2.reset_noise()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.nl1(x))
        return self.nl2(x)


class DQN:
    ''' DQN算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

        self.q_net = NoisyQnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = NoisyQnet(state_dim, hidden_dim, action_dim).to(device)
        self.epsilon = 0

        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标

        dqn_loss = F.mse_loss(q_values, q_targets.detach())  # 均方误差损失函数
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        # NoisyNet reset noise
        self.q_net.reset_noise()
        self.target_q_net.reset_noise()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alg_name = 'NoisyNet'
lr = 1e-3
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
update_interval = 1

env_name = 'CartPole-v1'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

replay_buffer = utils.ReplayBuffer(buffer_size)
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)

if __name__ == '__main__':
    print(env_name)
    return_list = utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size,
                                               update_interval)
    utils.dump(f'./results/{alg_name}.pkl', return_list)
    utils.show(f'./results/{alg_name}.pkl', alg_name)
