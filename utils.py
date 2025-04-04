import pickle
import random
import time
import operator
import gymnasium as gym
import numpy as np
from functools import wraps
from typing import Callable
from matplotlib import pyplot as plt
from tqdm import tqdm


def timeit(func):
    @wraps(func)
    def timed(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds", end='')
        return result

    return timed


class ReplayBuffer:
    def __init__(self, capacity):
        self._storage = []
        self._capacity = int(capacity)
        self._next_idx = 0

    def size(self):
        return len(self._storage)

    def add(self, *data):
        if self._next_idx < self.size():
            self._storage[self._next_idx] = data
        else:
            self._storage.append(data)
        self._next_idx = (self._next_idx + 1) % self._capacity

    def sample(self, batch_size):
        data = random.choices(self._storage, k=batch_size)
        return zip(*data)


class SegmentTree:
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    Attributes:
        capacity (int)
        tree (list)
        operation (function)

    """

    def __init__(self, capacity: int, operation: Callable, init_value: float):
        """Initialization.

        Args:
            capacity (int)
            operation (function)
            init_value (float)

        """
        assert (
                capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(
            self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying `self.operation`."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """Set value in tree."""
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """ Create SumSegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, init_value=0.0
        )

    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """Find the highest index `i` about upper bound in the tree"""
        # Check assert case and fix bug
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = left
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, init_value=float("inf")
        )

    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        super().__init__(capacity)
        # The exponent α determines how much prioritization is used
        # with α = 0 corresponding to the uniform case.
        self.alpha = alpha
        # Importance sampling weight β to correct bias
        # annealing to 1 at the end of learning
        self.beta = beta
        self.max_priority = 1.0 ** alpha

        tree_capacity = 1 << (capacity - 1).bit_length()
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def add(self, *data):
        """Store experience and priority."""
        self.sum_tree[self._next_idx] = self.max_priority
        self.min_tree[self._next_idx] = self.max_priority
        super().add(*data)

    def sample(self, batch_size):
        """Sample a batch of experiences."""

        indices = self._sample_proportional(batch_size)
        data = [self._storage[i] for i in indices]
        weights = [self._calculate_importance_sampling_weight(i) for i in indices]

        return zip(*data), weights, indices

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
            store_prior = prior ** self.alpha
            self.sum_tree[idx] = store_prior
            self.min_tree[idx] = store_prior
            self.max_priority = max(self.max_priority, store_prior)


class PrioritizedReplayBuffer_SIMPLE(ReplayBuffer):
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0 ** alpha
        self.priorities = []

    def add(self, *data):
        """Store experience and priority."""
        if self._next_idx < self.size():
            self._storage[self._next_idx] = data
            self.priorities[self._next_idx] = self.max_priority
        else:
            self._storage.append(data)
            self.priorities.append(self.max_priority)
        self._next_idx = (self._next_idx + 1) % self._capacity

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        indices = random.choices(list(range(self.size())),
                                 weights=self.priorities,
                                 k=batch_size)
        data = [self._storage[i] for i in indices]
        weights = [self._calculate_importance_sampling_weight(i) for i in indices]

        return zip(*data), weights, indices

    def _calculate_importance_sampling_weight(self, idx):
        """Calculate the weight of the experience at idx."""
        p_total = sum(self.priorities)
        p_min = min(self.priorities) / p_total
        max_weight = (p_min * self.size()) ** (-self.beta)

        p_sample = self.priorities[idx] / p_total
        weight = (p_sample * self.size()) ** (-self.beta) / max_weight

        return weight

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        for idx, prior in zip(indices, priorities):
            store_prior = prior ** self.alpha
            self.priorities[idx] = store_prior
            self.max_priority = max(self.max_priority, store_prior)


def train_on_policy_agent(env, agent, num_episodes, save_model=False):
    total_step = 0
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [],
                                   'rewards': [], 'dones': []}
                state, _ = env.reset(seed=total_step)
                done, truncated = False, False
                while not done and not truncated:
                    action = agent.take_action(state, eval=False)
                    next_state, reward, done, truncated, _ = env.step(action)

                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done or truncated)

                    state = next_state
                    episode_return += reward
                    total_step += 1
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'total_step': '%d' % total_step,
                                      'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
            if save_model: agent.save('./models')
    return return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size,
                           batch_size, update_interval, save_model=False):
    total_step = 0
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset(seed=total_step)
                done, truncated = False, False
                while not done and not truncated:
                    action = agent.take_action(state, eval=False)
                    next_state, reward, done, truncated, _ = env.step(action)

                    replay_buffer.add(state, action, reward, next_state, done or truncated)
                    if replay_buffer.size() > minimal_size and total_step % update_interval == 0:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns,
                                           'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)

                    state = next_state
                    episode_return += reward
                    total_step += 1
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'total_step': '%d' % total_step,
                                      'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
            if save_model: agent.save('./models')
    return return_list


def visualization(agent, env_name, num_episodes=10):
    agent.load_actor('models')
    env = gym.make(env_name, render_mode='human')
    for _ in range(num_episodes):
        state, info = env.reset()
        done, truncated = False, False
        episode_return = 0.
        while not done and not truncated:
            action = agent.take_action(state, eval=True)
            state, reward, done, truncated, info = env.step(action)
            episode_return += reward
        print(episode_return)
    env.close()


def dump(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def moving_average(a, window_size):
    return np.array([np.mean(a[i:i + window_size]) for i in range(len(a) - window_size + 1)])


def moving_std(a, window_size):
    return np.array([np.std(a[i:i + window_size]) for i in range(len(a) - window_size + 1)])


def moving_max(a, window_size):
    return np.array([np.max(a[i:i + window_size]) for i in range(len(a) - window_size + 1)])


def moving_min(a, window_size):
    return np.array([np.min(a[i:i + window_size]) for i in range(len(a) - window_size + 1)])


def show(data, title=None, x_label='Episodes', y_label='Rewards', window_size=9):
    if isinstance(data, str):
        with open(data, 'rb') as f:
            data_list = pickle.load(f)
    else:
        data_list = data

    rolling_mean = moving_average(data_list, window_size)
    rolling_max = moving_max(data_list, window_size)
    rolling_min = moving_min(data_list, window_size)
    # rolling_std = moving_std(data_list, window_size)
    x = np.arange(window_size - 1, len(data_list))

    plt.plot(x,
             rolling_mean,
             label='Rolling Mean',
             color='black')

    plt.fill_between(x,
                     rolling_min,
                     rolling_max,
                     alpha=0.3,
                     label='Rolling Min Max')

    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def compare(*files, title='compare', window_size=9):
    data_map = {}
    avg_lists = {}
    for file, name in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            data_map[name] = data
            avg_lists[name] = moving_average(data, window_size)

    for name, l in data_map.items():
        plt.plot(list(range(len(l))), l, label=name, linewidth=1.0)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.legend()
    plt.show()

    for name, l in avg_lists.items():
        plt.plot(list(range(len(l))), l, label=name, linewidth=1.0)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.legend()
    plt.show()
