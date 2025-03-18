import collections
import os
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

import utils


class ReplayBuffer:
    def __init__(self, capacity, n_step=1):
        self._storage = []
        self._capacity = capacity
        self._next_idx = 0

        self.n_step = n_step
        self.n_step_buffer = collections.deque(maxlen=n_step)

    def size(self):
        return len(self._storage)

    def add(self, *data):
        self.n_step_buffer.append(data)
        if len(self.n_step_buffer) < self.n_step:
            return

        data = list(self.n_step_buffer)
        if self._next_idx < self.size():
            self._storage[self._next_idx] = data
        else:
            self._storage.append(data)
        self._next_idx = (self._next_idx + 1) % self._capacity

    def sample(self, batch_size):
        return random.choices(self._storage, k=batch_size)


class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN:
    ''' DQN算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate,
                 epsilon, target_update, device, n_step=1):
        self.action_dim = action_dim
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.n_step = n_step

        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)

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
        gamma = torch.tensor(transition_dict['gammas'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + gamma * max_next_q_values * (1 - dones)  # TD误差目标

        dqn_loss = F.mse_loss(q_values, q_targets.detach())  # 均方误差损失函数
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1


def get_n_step_data(batch_data, gamma, n_step):
    # bach_data (B,n_step,transition)
    if n_step == 1:
        b_s, b_a, b_r, b_ns, b_d = zip(*[d[0] for d in batch_data])
        b_g = [gamma for _ in batch_data]
        return b_s, b_a, b_r, b_ns, b_d, b_g

    b_s = [d[0][0] for d in batch_data]
    b_a = [d[0][1] for d in batch_data]
    b_r = [0.0] * len(batch_data)
    b_ns = [d[0][3] for d in batch_data]
    b_d = [0] * len(batch_data)
    b_g = [gamma] * len(batch_data)

    for i in range(len(batch_data)):
        for step in range(0, n_step):
            reward, next_state, done = batch_data[i][step][-3:]
            b_r[i] += gamma ** step * reward
            b_ns[i] = next_state
            b_g[i] = gamma ** (step + 1)
            if done: break

    return b_s, b_a, b_r, b_ns, b_d, b_g


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alg_name = 'MultiStepDQN'
lr = 1e-3
num_episodes = 1000
hidden_dim = 128
gamma = 0.99
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
update_interval = 1
n_step = 3

env_name = 'CartPole-v1'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
replay_buffer = ReplayBuffer(buffer_size, n_step)
agent = DQN(state_dim, hidden_dim, action_dim, lr, epsilon,
            target_update, device, n_step)

if __name__ == '__main__':
    os.makedirs(f'results/{alg_name}', exist_ok=True)
    print(env_name)
    total_step = 0
    return_list = []
    for i in range(10):
        with tqdm.tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset(seed=total_step)
                done = False
                truncated = False
                while not done and not truncated:
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done or truncated)

                    state = next_state
                    episode_return += reward
                    total_step += 1

                    if replay_buffer.size() > minimal_size and total_step % update_interval == 0:
                        batch_data = replay_buffer.sample(batch_size)
                        b_s, b_a, b_r, b_ns, b_d, b_g = get_n_step_data(batch_data, gamma, n_step)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns,
                                           'rewards': b_r, 'dones': b_d, 'gammas': b_g}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    utils.dump(f'results/{alg_name}/return.pkl', return_list)
    utils.show(f'results/{alg_name}/return.pkl', alg_name)
