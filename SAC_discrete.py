import math
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SAC:
    ''' 处理离散动作的SAC算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 alpha_lr, target_entropy, tau, gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic1 = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic2 = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic1 = ValueNet(state_dim, hidden_dim,
                                       action_dim).to(device)  # 第一个目标Q网络
        self.target_critic2 = ValueNet(state_dim, hidden_dim,
                                       action_dim).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(),
                                                  lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(),
                                                  lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(1, requires_grad=True,
                                      dtype=torch.float, device=device)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.total_step = 0

    def take_action(self, state, eval=False):
        if self.total_step <= initial_random_steps and not eval:
            action = env.action_space.sample()
        else:
            state = torch.FloatTensor([state]).to(self.device)
            probs = self.actor(state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample().item()

        self.total_step += 1
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']
                               ).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        with torch.no_grad():
            next_probs = self.actor(next_states)
            next_log_probs = torch.log(next_probs + 1e-8)
            next_q1 = self.target_critic1(next_states)
            next_q2 = self.target_critic2(next_states)

            # 由于离散情况下网络直接输出动作分布
            # 不再需要熵和Q的估计值, 可以直接直接计算期望
            # 下一个状态的熵 E_a'[Q(s',a') - log(pi(a'|s'))]
            next_entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
            # 下一个状态Q值的期望 E_a'[Q(s',a')]
            next_q = torch.sum(next_probs * torch.min(next_q1, next_q2), dim=1, keepdim=True)
            next_value = next_q + self.log_alpha.exp() * next_entropy
            q_target = rewards + self.gamma * next_value * (1 - dones)

        # 更新两个Q网络
        # Minimize r + γ * (1-d) * pi(s_t+1)^T *(min(next_Q1, next_Q2) - next_log_prob) - curr_Q
        critic1_q_values = self.critic1(states).gather(1, actions)
        critic1_loss = F.mse_loss(critic1_q_values, q_target)
        critic2_q_values = self.critic2(states).gather(1, actions)
        critic2_loss = F.mse_loss(critic2_q_values, q_target)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 更新策略网络
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        q1_value = self.critic1(states)
        q2_value = self.critic2(states)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        actor_loss = -(q + self.log_alpha.exp() * entropy).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = -(self.log_alpha.exp() * (entropy + target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self, folder='results'):
        torch.save(self.actor.state_dict(), folder + "/sac_d_actor")
        torch.save(self.critic1.state_dict(), folder + "/sac_d_critic1")
        torch.save(self.critic2.state_dict(), folder + "/sac_d_critic2")
        torch.save(self.log_alpha, folder + "/sac_d_log_alpha")

    def load_actor(self, folder='models'):
        self.actor.load_state_dict(torch.load(folder + "/sac_d_actor"))

    def visual(self):
        self.load_actor()
        utils.visualization(self, env_name)


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
batch_size = 64
update_interval = 1

env_name = 'CartPole-v1'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
# target_entropy = 0.98 * math.log(action_dim)
target_entropy = -action_dim

agent = SAC(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr,
            target_entropy, tau, gamma, device)

replay_buffer = utils.ReplayBuffer(buffer_size)
if __name__ == '__main__':
    print(env_name)
    return_list = utils.train_off_policy_agent(env, agent, num_episodes,
                                               replay_buffer, minimal_size,
                                               batch_size, update_interval,
                                               save_model=True)

    utils.dump('./results/sac_d.pkl', return_list)
    utils.show('./results/sac_d.pkl', 'sac')
