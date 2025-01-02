import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class CategoricalQnet(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
            support: torch.Tensor,
            atom_size: int = 51
    ):
        super().__init__()

        self.support = support
        self.out_dim = output_dim
        self.atom_size = atom_size

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * atom_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def dist(self, x):
        """Get distribution for atoms."""
        q_atoms = self.layers(x).view(-1, self.out_dim, self.atom_size)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        return dist


class DQN:

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device,
                 v_min=0.0,
                 v_max=200.0,
                 atom_size=51):
        self.action_dim = action_dim
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(v_min, v_max, atom_size).to(device)

        self.q_net = CategoricalQnet(
            state_dim, action_dim, hidden_dim, self.support, atom_size).to(device)
        self.target_q_net = CategoricalQnet(
            state_dim, action_dim, hidden_dim, self.support, atom_size).to(device)

        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
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
        batch_size = len(states)

        delta_z = (self.v_max - self.v_min) / (self.atom_size - 1)
        with torch.no_grad():
            max_next_action = self.target_q_net(next_states).argmax(1)
            next_dist = self.target_q_net.dist(next_states)
            max_next_dist = next_dist[range(batch_size), max_next_action]  # (B,atom_size)

            t_z = rewards + self.gamma * self.support * (1 - dones)
            t_z = t_z.clamp(self.v_min, self.v_max)

            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = torch.linspace(
                0, (batch_size - 1) * self.atom_size, batch_size
            ).long().unsqueeze(1).expand(batch_size, self.atom_size).to(self.device)

            proj_dist = torch.zeros(max_next_dist.size()).to(self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (max_next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (max_next_dist * (b - l.float())).view(-1)
            )

        dist = self.q_net.dist(states)
        max_dist = dist[range(batch_size), actions.squeeze()]  # (B,atom_size)

        dqn_loss = -(proj_dist * torch.log(max_dist)).sum(1).mean()

        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alg_name = 'C51'
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

v_min = -100.0
v_max = 0.0
atom_size = 51

env_name = 'CartPole-v1'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

replay_buffer = utils.ReplayBuffer(buffer_size)
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device, v_min, v_max, atom_size)

if __name__ == '__main__':
    print(env_name)
    return_list = utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size,
                                               update_interval)
    utils.dump(f'./results/{alg_name}.pkl', return_list)
    utils.show(f'./results/{alg_name}.pkl', alg_name)
