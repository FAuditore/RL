import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import utils


class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        action = torch.tanh(normal_sample)  # (B,action_dim)

        # 计算tanh_normal分布的对数概率密度
        log_prob = dist.log_prob(normal_sample).sum(dim=1, keepdim=True)
        log_prob -= torch.log(1 - action ** 2 + 1e-8).sum(dim=1, keepdim=True)
        action = action * self.action_bound
        return action, log_prob


class QValueNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        x = F.relu(self.fc1(torch.cat([x, a], dim=1)))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class SAC:
    ''' 处理连续动作的SAC算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device, initial_random_steps):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim,
                                         action_bound).to(device)  # 策略网络
        self.critic1 = QValueNetContinuous(state_dim, hidden_dim,
                                           action_dim).to(device)  # 第一个Q网络
        self.critic2 = QValueNetContinuous(state_dim, hidden_dim,
                                           action_dim).to(device)  # 第二个Q网络
        self.target_critic1 = QValueNetContinuous(state_dim,
                                                  hidden_dim, action_dim).to(
            device)  # 第一个目标Q网络
        self.target_critic2 = QValueNetContinuous(state_dim,
                                                  hidden_dim, action_dim).to(
            device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(),
                                                  lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(),
                                                  lr=critic_lr)
        # automatic entropy tuning
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(1, requires_grad=True,
                                      dtype=torch.float, device=device)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.initial_random_steps = initial_random_steps
        self.total_step = 0

    def take_action(self, state, eval=False):
        if self.total_step < self.initial_random_steps and not eval:
            action = env.action_space.sample()
        else:
            state = torch.FloatTensor([state]).to(self.device)
            action = self.actor(state)[0].squeeze(0).detach().cpu().numpy()

        self.total_step += 1
        return action

    def update(self, transition_dict):
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
            next_actions, next_log_probs = self.actor(next_states)
            next_q1 = self.target_critic1(next_states, next_actions)
            next_q2 = self.target_critic2(next_states, next_actions)
            # log_prob作为熵E_a[log(pi(a|s))]的估计值
            next_value = torch.min(next_q1, next_q2) - self.log_alpha.exp() * next_log_probs
            q_target = rewards + self.gamma * next_value * (1 - dones)

        # 更新两个Q网络
        # Minimize r + γ * (1-d) * (min(next_Q1, next_Q2) - α * next_log_prob) - curr_Q
        critic1_loss = F.mse_loss(self.critic1(states, actions), q_target)
        critic2_loss = F.mse_loss(self.critic2(states, actions), q_target)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 更新策略网络
        # Maximize min(Q_1,Q_2) - α * log_prob
        new_actions, log_probs = self.actor(states)
        q1 = self.critic1(states, new_actions)
        q2 = self.critic2(states, new_actions)
        actor_loss = -(torch.min(q1, q2) - self.log_alpha.exp() * log_probs).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        # Minimize - alpha * (log_prob + target_entropy)
        alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self, folder='models'):
        torch.save(self.actor.state_dict(), folder + "/sac_actor")
        torch.save(self.critic1.state_dict(), folder + "/sac_critic1")
        torch.save(self.critic2.state_dict(), folder + "/sac_critic2")
        torch.save(self.log_alpha, folder + "/sac_log_alpha")

    def load_actor(self, folder='models'):
        self.actor.load_state_dict(torch.load(folder + "/sac_actor"))

    def visual(self):
        self.load_actor()
        utils.visualization(self, env_name)


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alg_name = 'SAC'
actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 3e-4
num_episodes = 5000
gamma = 0.99
tau = 0.005  # 软更新参数
hidden_dim = 256
buffer_size = 1e6
initial_random_steps = 10000  # Time steps initial random policy is used
minimal_size = 10000  # Update begin step
batch_size = 256
update_interval = 1

env_name = 'Pendulum-v1'
env_name = 'Walker2d-v5'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值
target_entropy = -action_dim

replay_buffer = utils.ReplayBuffer(buffer_size)
agent = SAC(state_dim, hidden_dim, action_dim, action_bound,
            actor_lr, critic_lr, alpha_lr, target_entropy, tau,
            gamma, device, initial_random_steps)

if __name__ == '__main__':
    print(env_name)
    return_list = utils.train_off_policy_agent(env, agent, num_episodes,
                                               replay_buffer, minimal_size,
                                               batch_size, update_interval,
                                               save_model=True)

    utils.dump(f'./results/{alg_name}.pkl', return_list)
    utils.show(f'./results/{alg_name}.pkl', alg_name)
