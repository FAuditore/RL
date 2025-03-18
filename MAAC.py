import os
import random

import matplotlib.pyplot as plt
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
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)


class ValueNet(nn.Module):
    def __init__(self, n_agents, observation_dim, action_dim, hidden_dim, attn_heads):
        super().__init__()
        self.n_head = attn_heads
        self.head_dim = hidden_dim // attn_heads
        assert (self.head_dim * attn_heads == hidden_dim)

        self.o_embeds = nn.ModuleList([nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LeakyReLU()
        ) for _ in range(n_agents)])

        self.o_a_embeds = nn.ModuleList([nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim),
            nn.LeakyReLU()
        ) for _ in range(n_agents)])

        # Q(embed(s), ∑α*attn(embed(s,a)))
        self.critics = nn.ModuleList([nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, action_dim)
        ) for _ in range(n_agents)])

        self.queries = nn.Linear(hidden_dim, hidden_dim)
        self.keys = nn.Linear(hidden_dim, hidden_dim)
        self.values = nn.Linear(hidden_dim, hidden_dim)

        mask = torch.zeros(n_agents, n_agents)
        torch.diagonal(mask).fill_(-1e10)  # attention without self
        self.register_buffer('mask', mask.unsqueeze(0).unsqueeze(0))

    def forward(self, observations, actions):
        B, N, _ = observations.size()

        # (B,N,hidden_dim)
        o_embeds = torch.stack(
            [self.o_embeds[i](observations[:, i]) for i in range(N)]).transpose(0, 1)

        # (B,N,hidden_dim)
        o_a_embeds = torch.stack(
            [self.o_a_embeds[i](torch.cat([observations[:, i], actions[:, i]], dim=1))
             for i in range(N)]).transpose(0, 1)

        # (B,nh,N,hd)
        Q = self.queries(o_embeds).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        K = self.keys(o_a_embeds).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        V = self.values(o_a_embeds).view(B, N, self.n_head, self.head_dim).transpose(1, 2)

        # (B, nh, N, N)
        attn_score = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        masked_score = attn_score + self.mask.clone().expand(*attn_score.size())

        attn_weights = F.softmax(masked_score, dim=-1)
        attn_out = attn_weights @ V  # (B, nh, N, N) x (B, nh, N, hd) -> (B, nh, N, hd)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, -1)

        # (B,N,action_dim)
        q_values = torch.stack([critic(
            torch.cat([o_embeds[:, i], attn_out[:, i]], dim=1))
            for i, critic in enumerate(self.critics)]).transpose(0, 1)
        return q_values


class MAAC:
    def __init__(self, env, n_agents, observation_dim, state_dim, action_dim,
                 hidden_dim, attn_heads, alpha, actor_lr, critic_lr, tau, gamma, device):
        self.actors = [PolicyNet(observation_dim, hidden_dim, action_dim).to(device) for _ in range(n_agents)]
        self.target_actors = [PolicyNet(observation_dim, hidden_dim, action_dim).to(device) for _ in range(n_agents)]

        self.critic = ValueNet(n_agents, observation_dim, action_dim, hidden_dim, attn_heads).to(device)
        self.target_critic = ValueNet(n_agents, observation_dim, action_dim, hidden_dim, attn_heads).to(device)
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
        self.alpha = alpha

        print({key: value for key, value in locals().items() if key not in ['self']})

    def take_action(self, observations, eval=False):
        actions = []
        with torch.no_grad():
            for i in range(self.n_agents):
                logits = self.actors[i](torch.tensor([observations[i]]).to(self.device)).squeeze()
                avail_actions = torch.tensor(self.env.get_avail_agent_actions(i)).to(self.device)
                logits[avail_actions == 0] = -1e10
                if not eval:
                    action_dist = torch.distributions.Categorical(logits=logits)
                    actions.append(action_dist.sample().item())
                else:
                    actions.append(logits.argmax().item())
        if not eval:
            self.total_step += 1
        return actions

    def update(self, transition_dict):
        batch_size = len(transition_dict['states'])
        states = torch.tensor(transition_dict['states']).to(self.device)
        observations = torch.tensor(transition_dict['observations']).to(self.device)
        avail_actions = torch.tensor(transition_dict['avail_actions']).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device)
        rewards = torch.tensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states']).to(self.device)
        next_observations = torch.tensor(transition_dict['next_observations']).to(self.device)
        next_avail_actions = torch.tensor(transition_dict['next_avail_actions']).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        with torch.no_grad():
            next_logits = torch.stack(
                [ta(next_observations[:, i])
                 for i, ta in enumerate(self.target_actors)]).transpose(0, 1)  # (B,N,action_dim)
            next_logits[next_avail_actions == 0] = -1e10
            next_one_hot_actions = F.one_hot(next_logits.argmax(dim=-1), num_classes=self.action_dim)

            next_qs = self.target_critic(next_observations, next_one_hot_actions)  # (B,N,action_dim)
            next_qs[next_avail_actions == 0] = -1e10
            next_probs = F.softmax(next_qs, dim=-1)
            next_q = torch.sum(next_probs * next_qs, dim=-1)  # 每个智能体下一个状态的期望Q值
            next_entropy = -torch.sum(next_probs * torch.log(next_probs + 1e-8), dim=-1)  # (B,N)
            next_value = next_q + self.alpha * next_entropy  # (B,N)
            q_target = rewards.expand(batch_size, self.n_agents) + self.gamma * next_value * (
                    1 - dones.expand(batch_size, self.n_agents))

        qs = self.critic(observations, F.one_hot(actions, num_classes=self.action_dim))
        q = qs.gather(2, actions.unsqueeze(2)).squeeze()
        critic_loss = F.mse_loss(q, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        logits = torch.stack([a(observations[:, i])
                              for i, a in enumerate(self.actors)]).transpose(0, 1)
        logits[avail_actions == 0] = -1e10
        cur_actions = logits.argmax(dim=-1)
        cur_qs = self.critic(observations, F.one_hot(cur_actions, num_classes=self.action_dim))  # (B,N,action_dim)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # (B,N)

        counterfactual_baseline = torch.sum(probs * qs, dim=-1)  # 其他智能体动作不变情况下Q值的期望
        cur_q = cur_qs.gather(2, cur_actions.unsqueeze(2)).squeeze()  # 当前策略q值

        log_probs = log_probs.gather(2, cur_actions.unsqueeze(2)).squeeze()
        actor_loss = -(log_probs * (cur_q - counterfactual_baseline).detach() + self.alpha * entropy).mean()

        for i in range(self.n_agents):
            self.actor_optimizers[i].zero_grad()
        actor_loss.backward()

        for i in range(self.n_agents):
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optimizers[i].step()
            self.soft_update(self.actors[i], self.target_actors[i])

        self.soft_update(self.critic, self.target_critic)

        return actor_loss.item(), critic_loss.item()

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self, folder='models'):
        torch.save([a.state_dict() for a in self.actors], folder + "/maac_actors")
        torch.save(self.critic.state_dict(), folder + "/maac_critic")

    def load(self, folder='models'):
        actor_state_dicts = torch.load(folder + '/maac_actors', map_location='cpu')
        for a, sd in zip(self.actors, actor_state_dicts):
            a.load_state_dict(sd)
        critic_state_dict = torch.load(folder + '/maac_critic', map_location='cpu')
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

alg_name = 'MAAC'
actor_lr = 1e-3
critic_lr = 1e-3
num_episodes = 50000
hidden_dim = 128
gamma = 0.99
tau = 0.005
save_model = True
buffer_size = 1e6
initial_random_steps = 400  # Time steps initial random policy is used
minimal_size = 400  # Update begin step
batch_size = 1024
update_interval = 100
alpha = 1e-3
attn_heads = 4

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

replay_buffer = utils.ReplayBuffer(buffer_size)
agent = MAAC(env, n_agents, observation_dim, state_dim, action_dim,
             hidden_dim, attn_heads, alpha, actor_lr, critic_lr, tau, gamma, device)

if __name__ == '__main__':
    os.makedirs(f'results/{alg_name}', exist_ok=True)
    return_list, eval_list, win_rate_list = [], [], []
    actor_losses, critic_losses = [], []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                observations, state = env.reset()
                episode_return, terminated = 0, False
                while not terminated:
                    avail_actions = env.get_avail_actions()
                    actions = agent.take_action(observations, eval=False)
                    reward, terminated, info = env.step(actions)
                    next_state = env.get_state()
                    next_observations = env.get_obs()
                    next_avail_actions = env.get_avail_actions()

                    replay_buffer.add(state, observations, avail_actions, actions, reward,
                                      next_state, next_observations, next_avail_actions, terminated)
                    if replay_buffer.size() > minimal_size and agent.total_step % update_interval == 0:
                        b_s, b_o, b_aa, b_a, b_r, b_ns, b_no, b_naa, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'observations': b_o, 'avail_actions': b_aa,
                                           'actions': b_a, 'rewards': b_r, 'next_states': b_ns,
                                           'next_observations': b_no, 'next_avail_actions': b_naa,
                                           'dones': b_d}
                        actor_loss, critic_loss = agent.update(transition_dict)
                        actor_losses.append(actor_loss)
                        critic_losses.append(critic_loss)

                    state = next_state
                    observations = next_observations
                    episode_return += reward
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'total_step': '%d' % agent.total_step,
                                      'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:]),
                                      'actor_loss': '%.3f' % np.mean(actor_losses[-10:]),
                                      'critic_loss': '%.3f' % np.mean(critic_losses[-10:])})
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

    plt.plot(actor_losses, label='actor_loss')
    plt.plot(critic_losses, label='critic_loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()
