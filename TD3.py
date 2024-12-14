import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class GaussianNoise:
    def __init__(self, size, mu=0, sigma=0.1, scale=1.):
        self.size = size
        self.mu = mu
        self.std = sigma
        self.scale = scale

    def sample(self):
        return self.scale * np.random.normal(self.mu, self.std, self.size)


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x)) * self.action_bound


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        sa = F.relu(self.fc1(sa))
        sa = F.relu(self.fc2(sa))
        return self.fc3(sa)


class TD3:
    """TD3Agent interacting with environment.

       Attribute:
           env (gym.Env): openAI Gym environment
           actor (nn.Module): actor model to select actions
           target_actor (nn.Module): actor model to predict next actions
           actor_optimizer (Optimizer): optimizer for training actor
           critic1 (nn.Module): critic model to predict state values
           critic2 (nn.Module): critic model to predict state values
           critic_target1 (nn.Module): target critic model to predict state values
           critic_target2 (nn.Module): target critic model to predict state values
           critic_optimizer (Optimizer): optimizer for training critic
           gamma (float): discount factor
           tau (float): parameter for soft target update
           exploration_noise (GaussianNoise): gaussian noise for policy
           target_policy_noise (GaussianNoise): gaussian noise for target policy
           target_policy_noise_clip (float): clip target gaussian noise
           policy_update_freq (int): update actor every time critic updates this times
       """

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, tau, gamma, device, initial_random_steps,
                 exploration_noise, target_policy_noise, target_policy_noise_clip, policy_update_freq):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic1 = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic1 = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())

        self.critic2 = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic2 = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr)

        self.gamma = gamma
        self.action_bound = action_bound
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device

        self.exploration_noise = exploration_noise
        self.target_policy_noise = target_policy_noise
        self.target_policy_noise_clip = target_policy_noise_clip
        self.policy_update_freq = policy_update_freq

        self.initial_random_steps = initial_random_steps
        self.count = 0  # update count
        self.total_step = 0  # env step

    def take_action(self, state, eval=False):
        if self.total_step < self.initial_random_steps and not eval:
            action = env.action_space.sample()
        else:
            state = torch.FloatTensor([state]).to(self.device)
            action = self.actor(state).squeeze(0).detach().cpu().numpy()
            if not eval:
                action = np.clip(action + self.exploration_noise.sample(),
                                 -self.action_bound, self.action_bound)
        self.total_step += 1
        return action

    def update(self, transition_dict):
        self.count += 1
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        with torch.no_grad():
            noise = torch.clamp(
                torch.FloatTensor(self.target_policy_noise.sample()).to(self.device),
                -self.target_policy_noise_clip,
                self.target_policy_noise_clip)

            next_actions = torch.clamp(
                self.target_actor(next_states) + noise,
                -self.action_bound,
                self.action_bound)

            next_q1 = self.target_critic1(next_states, next_actions)
            next_q2 = self.target_critic2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            q_target = rewards + self.gamma * next_q * (1 - dones)

        critic_loss = F.mse_loss(self.critic1(states, actions), q_target) + F.mse_loss(
            self.critic2(states, actions), q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.count % self.policy_update_freq == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.target_actor)
            self.soft_update(self.critic1, self.target_critic1)
            self.soft_update(self.critic2, self.target_critic2)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self, folder='models'):
        torch.save(self.actor.state_dict(), folder + "/td3_actor")
        torch.save(self.critic1.state_dict(), folder + "/td3_critic1")
        torch.save(self.critic2.state_dict(), folder + "/td3_critic2")

    def load_actor(self, folder='models'):
        self.actor.load_state_dict(torch.load(folder + "/td3_actor"))

    def visual(self):
        self.load_actor()
        utils.visualization(self, env_name)


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

actor_lr = 3e-4
critic_lr = 3e-4
num_episodes = 5000
hidden_dim = 512
gamma = 0.99
tau = 0.005  # Target network update rate
buffer_size = 1e6
initial_random_steps = 10000  # Time steps initial random policy is used
minimal_size = 10000  # Update begin step
batch_size = 256
update_interval = 1

env_name = 'Reacher-v5'
env_name = 'Hopper-v5'
env_name = 'Humanoid-v5'
env_name = 'Walker2d-v5'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

# TD3 parameters
exploration_noise = GaussianNoise(action_dim, mu=0, sigma=0.1, scale=action_bound)
target_policy_noise = GaussianNoise((batch_size, action_dim), mu=0, sigma=0.2, scale=action_bound)
target_policy_noise_clip = 0.5
policy_update_freq = 2

replay_buffer = utils.ReplayBuffer(buffer_size)
agent = TD3(state_dim, hidden_dim, action_dim, action_bound,
            actor_lr, critic_lr,
            tau, gamma, device, initial_random_steps,
            exploration_noise, target_policy_noise, target_policy_noise_clip, policy_update_freq)

if __name__ == '__main__':
    print(env_name)
    return_list = utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer,
                                               minimal_size, batch_size, update_interval,
                                               save_model=True)
    utils.dump('./results/td3.pkl', return_list)
    utils.show('./results/td3.pkl', 'td3')
