import collections
import math
import os
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import utils
from segment_tree import SumSegmentTree, MinSegmentTree


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
        # only store transition when step reach n_step
        self.n_step_buffer.append(data)
        if len(self.n_step_buffer) < self.n_step:
            return

        data = list(self.n_step_buffer)
        if self._next_idx < self.size():
            self._storage[self._next_idx] = data
        else:
            self._storage.append(data)

        store_idx = self._next_idx
        self._next_idx = (self._next_idx + 1) % self._capacity
        return store_idx

    def sample(self, batch_size):
        return random.choices(self._storage, k=batch_size)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    Attributes:
        alpha: The exponent α determines how much prioritization is used
            with α = 0 corresponding to the uniform case.
        beta: Importance sampling weight β to correct bias
            annealing to 1 at the end of learning
        n_step: store multistep transition
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4, n_step=1):

        super().__init__(capacity, n_step)
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0 ** alpha

        tree_capacity = 1 << (capacity - 1).bit_length()
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def add(self, *data):
        """Store experience and priority."""
        if idx := super().add(*data):
            self.sum_tree[idx] = self.max_priority
            self.min_tree[idx] = self.max_priority

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        indices = self._sample_proportional(batch_size)
        data = [self._storage[i] for i in indices]
        weights = [self._calculate_importance_sampling_weight(i) for i in indices]
        return data, weights, indices

    def _sample_proportional(self, batch_size):
        """Sample indices based on proportions."""
        p_total = self.sum_tree.sum()
        segment = p_total / batch_size

        indices = []
        for i in range(batch_size):
            start = segment * i
            end = segment * (i + 1)
            sample = random.uniform(start, end)
            idx = self.sum_tree.retrieve(sample)
            indices.append(idx)
        return indices

    def _calculate_importance_sampling_weight(self, idx):
        """Calculate the weight of the experience at idx."""
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.size()) ** (-self.beta)

        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * self.size()) ** (-self.beta) / max_weight
        return weight

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        for idx, prior in zip(indices, priorities):
            prior = prior ** self.alpha
            self.sum_tree[idx] = prior
            self.min_tree[idx] = prior
            self.max_priority = max(self.max_priority, prior)


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


class Qnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, atom_size, support):
        super().__init__()

        self.support = support
        self.out_dim = output_dim
        self.atom_size = atom_size

        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.advantage_layer = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, self.out_dim * self.atom_size)
        )

        self.value_layer = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, self.atom_size)
        )

    def forward(self, x):
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def dist(self, x):
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        advantage = self.advantage_layer(F.relu(feature))
        value = self.value_layer(F.relu(feature))

        advantage = advantage.view(-1, self.out_dim, self.atom_size)
        value = value.view(-1, 1, self.atom_size)

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)

        return dist

    def reset_noise(self):
        """Reset all noisy layers."""
        for layer in (self.advantage_layer[0],
                      self.advantage_layer[2],
                      self.value_layer[0],
                      self.value_layer[2]):
            layer.reset_noise()


class DQN:
    """DQN Agent.
    Attributes:
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
    """

    def __init__(self, state_dim, hidden_dim, action_dim,
                 learning_rate, target_update, device,
                 replay_buffer, epsilon,
                 v_min, v_max, atom_size):
        self.target_update = target_update
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.replay_buffer = replay_buffer
        self.epsilon = epsilon

        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(v_min, v_max, atom_size).to(device)

        self.q_net = Qnet(state_dim, hidden_dim, action_dim, atom_size, self.support).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim, atom_size, self.support).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)

    def take_action(self, state):
        # if np.random.random() < self.epsilon:
        #     action = np.random.randint(self.action_dim)
        # else:
        #     state = torch.tensor([state], dtype=torch.float).to(self.device)
        #     action = self.q_net(state).argmax().item()

        # NoisyNet: no epsilon greedy action selection
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.int64).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        gamma = torch.tensor(transition_dict['gammas'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        weights = torch.tensor(transition_dict['weights'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        indices = transition_dict['indices']
        batch_size = len(states)

        # Categorical DQN
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # max_next_action = self.target_q_net(next_states).argmax(1)
            # Double DQN
            max_next_action = self.q_net(next_states).argmax(1)
            next_dist = self.target_q_net.dist(next_states)
            max_next_dist = next_dist[range(batch_size), max_next_action]  # (B,atom_size)

            # Categorical DQN compute target projection
            t_z = rewards + gamma * self.support * (1 - dones)
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z  # 获取target落在的区间, b是浮点数
            l = b.floor().long()
            u = b.ceil().long()

            offset = torch.linspace(
                0, (batch_size - 1) * self.atom_size, batch_size
            ).long().unsqueeze(1).expand(batch_size, self.atom_size).to(self.device)

            proj_dist = torch.zeros(max_next_dist.size()).to(self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (max_next_dist * (u - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (max_next_dist * (b - l)).view(-1)
            )
        dist = self.q_net.dist(states)
        max_dist = dist[range(batch_size), actions]
        # Categorical DQN cross-entropy loss
        elementwise_loss = - (proj_dist * torch.log(max_dist)).sum(1, keepdim=True)

        # PER weights * loss
        loss = torch.mean(weights * elementwise_loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # PER update priorities
        td_error = elementwise_loss.detach().cpu()
        new_prior = td_error + 1e-8
        self.replay_buffer.update_priorities(indices, new_prior.squeeze().tolist())

        # NoisyNet reset noise
        self.q_net.reset_noise()
        self.target_q_net.reset_noise()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1


def get_n_step_data(batch_data, gamma, n_step):
    """Return n step obs, rew, next_obs, and done.
    Args:
        batch_data: (B,n_step,transition)
        gamma: discount factor

    """
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

alg_name = 'Rainbow'
lr = 1e-3
num_episodes = 2000
hidden_dim = 128
gamma = 0.99
epsilon = 1  # deprecated
target_update = 100

# Multi-step learning
n_step = 3

# Replay buffer parameters
buffer_size = 10000
minimal_size = 500
batch_size = 64
update_interval = 1

# PER parameters
alpha = 0.5
beta = 0.4

# Categorical DQN parameters
v_min = -10
v_max = 10
atom_size = 51

env_name = 'MountainCar-v0'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"""
Learning rate: {lr}
Number of episodes: {num_episodes}
Hidden dimension: {hidden_dim}
Gamma: {gamma}
Epsilon (deprecated): {epsilon}
Target update interval: {target_update}
n-step: {n_step}
Buffer size: {buffer_size}
Minimal buffer size: {minimal_size}
Batch size: {batch_size}
Update interval: {update_interval}
Alpha (PER): {alpha}
Beta (PER): {beta}
v_min: {v_min}
v_max: {v_max}
Atom size: {atom_size}
Env name:{env_name}
State dimension: {state_dim}
Action dimension: {action_dim}
""")

replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha, beta, n_step)
agent = DQN(state_dim, hidden_dim, action_dim, lr,
            target_update, device, replay_buffer, epsilon,
            v_min, v_max, atom_size)

if __name__ == '__main__':
    os.makedirs(f'results/{alg_name}', exist_ok=True)
    print(env_name)
    total_step = 0
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                # agent.epsilon = max(0.01, agent.epsilon - 1e-4)

                # PER beta annealing
                replay_buffer.beta += (1.0 - beta) / num_episodes

                episode_return = 0
                state, _ = env.reset(seed=total_step)
                done, truncated = False, False
                while not done and not truncated:
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done or truncated)

                    state = next_state
                    episode_return += reward
                    total_step += 1

                    if replay_buffer.size() > minimal_size and total_step % update_interval == 0:
                        batch_data, b_w, b_i = replay_buffer.sample(batch_size)
                        b_s, b_a, b_r, b_ns, b_d, b_g = get_n_step_data(batch_data, gamma, n_step)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d, 'gammas': b_g, 'weights': b_w, 'indices': b_i}
                        agent.update(transition_dict)

                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    utils.dump(f'./results/{alg_name}/return.pkl', return_list)
    utils.show(f'./results/{alg_name}/return.pkl', alg_name)
