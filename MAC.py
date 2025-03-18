import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import utils


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return self.fc2(x)


class ValueNet(nn.Module):
    def __init__(self, critic_input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(critic_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MAC:

    def __init__(self, env, n_agents, observation_dim, state_dim, action_dim,
                 hidden_dim, actor_lr, critic_lr, tau, gamma, device):
        self.actors = [PolicyNet(observation_dim, hidden_dim, action_dim).to(device) for _ in range(n_agents)]

        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.target_critic = ValueNet(state_dim, hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.env = env
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.total_step = 0

        print({key: value for key, value in locals().items() if key not in ['self']})

    def take_action(self, observations, eval=False):
        actions = []
        with torch.no_grad():
            for i in range(self.n_agents):
                logits = self.actors[i](torch.FloatTensor([observations[i]]).to(self.device)).squeeze()
                avail_actions = torch.tensor(self.env.get_avail_agent_actions(i)).to(self.device)
                logits[avail_actions == 0] = float('-inf')
                if not eval:
                    action_dist = torch.distributions.Categorical(logits=logits)
                    actions.append(action_dist.sample().item())
                else:
                    actions.append(logits.argmax().item())
        if not eval:
            self.total_step += 1
        return actions

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device)
        observations = torch.tensor(transition_dict['observations']).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.target_critic(next_states) * (1 - dones)
        td_target = td_target.detach()
        v = self.critic(states)

        critic_loss = F.mse_loss(td_target.detach(), v)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        td_delta = (td_target - v).detach()
        for i in range(self.n_agents):
            log_probs = torch.log(self.actors[i](observations[:, i, :]).gather(1, actions[:, i].unsqueeze(1)))
            actor_loss = torch.mean(-log_probs * td_delta)
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        self.soft_update(self.critic, self.target_critic)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self, folder='models'):
        torch.save([a.state_dict() for a in self.actors], folder + "/mac_actors")
        torch.save(self.critic, folder + "/mac_critic")

    def load(self, folder='models'):
        actor_state_dicts = torch.load(folder + '/mac_actors', map_location='cpu')
        for a, sd in zip(self.actors, actor_state_dicts):
            a.load_state_dict(sd)
        critic_state_dict = torch.load(folder + '/mac_critic', map_location='cpu')
        self.critic.load_state_dict(critic_state_dict)


def evaluate(agent, eval_env, eval_episodes=20):
    avg_reward = 0.
    won = 0.
    for episode in range(eval_episodes):
        observations, state = eval_env.reset()
        terminated, info = False, None
        while not terminated:
            action = agent.take_action(observations, eval=True)
            reward, terminated, info = eval_env.step(action)
            avg_reward += reward
            observations = eval_env.get_obs()
        if info.get('battle_won', False):
            won += 1

    avg_reward /= eval_episodes
    win_rate = won / eval_episodes
    print(f"Evaluation over {eval_episodes} episodes,"
          f"Avg_reward: {avg_reward:.3f}",
          f"Win_rate: {win_rate * 100:.2f}%")
    return avg_reward, win_rate


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alg_name = 'MAC'
actor_lr = 1e-3
critic_lr = 1e-3
num_episodes = 10000
hidden_dim = 128
gamma = 0.99
tau = 1e-2
save_model = True

# 星际争霸2 -- SMAC 环境介绍 https://zhuanlan.zhihu.com/p/595500237
from smac.env import StarCraft2Env

env = StarCraft2Env(map_name="3m", seed=0, obs_instead_of_state=True)
env_info = env.get_env_info()
print('env_info: ', env_info)

n_agents = env_info['n_agents']
observation_dim = env.get_obs_size()
state_dim = env.get_state_size()
# 0:noop 1:stop 2:north 3:south 4:east 5:west 6-N:enemy id
action_dim = env.get_total_actions()

agent = MAC(env, n_agents, observation_dim, state_dim, action_dim,
            hidden_dim, actor_lr, critic_lr, tau, gamma, device)

if __name__ == '__main__':
    os.makedirs(f'results/{alg_name}', exist_ok=True)
    return_list, eval_list, win_rate_list = [], [], []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):

                episode_return, terminated = 0, False
                transition_dict = {'states': [], 'observations': [], 'actions': [],
                                   'rewards': [], 'next_states': [], 'dones': []}

                observations, state = env.reset()
                while not terminated:
                    actions = agent.take_action(observations, eval=False)
                    reward, terminated, info = env.step(actions)
                    next_state = env.get_state()
                    next_observations = env.get_obs()

                    transition_dict['states'].append(state)
                    transition_dict['observations'].append(observations)
                    transition_dict['actions'].append(actions)
                    transition_dict['rewards'].append(reward)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['dones'].append(terminated)

                    state = next_state
                    observations = next_observations
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'total_step': '%d' % agent.total_step,
                                      'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                if (i_episode + 1) % 100 == 0:
                    eval_return, win_rate = evaluate(agent, env, eval_episodes=20)
                    eval_list.append(eval_return)
                    win_rate_list.append(win_rate)
                pbar.update(1)
            if save_model: agent.save()
    utils.dump(f'results/{alg_name}/return.pkl', return_list)
    utils.dump(f'results/{alg_name}/eval.pkl', eval_list)
    utils.dump(f'results/{alg_name}/win_rate.pkl', win_rate_list)
    utils.show(f'results/{alg_name}/return.pkl', alg_name)
    utils.show(f'results/{alg_name}/eval.pkl', f'{alg_name} eval')
    utils.show(f'results/{alg_name}/win_rate.pkl', f'{alg_name} win rate')
