import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import utils


class QNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QMixer(nn.Module):
    def __init__(self, state_dim, n_agents, hyper_net_hidden_dim, mix_net_hidden_dim):
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.mix_net_hidden_dim = mix_net_hidden_dim

        # (B,n_agents) -> (B,mix_hidden_dim)
        self.hyper_W1 = nn.Sequential(
            nn.Linear(state_dim, hyper_net_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_net_hidden_dim, n_agents * mix_net_hidden_dim))
        self.hyper_B1 = nn.Linear(state_dim, mix_net_hidden_dim)

        # (B,mix_hidden_dim) -> (B,1)
        self.hyper_W2 = nn.Sequential(nn.Linear(state_dim, hyper_net_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hyper_net_hidden_dim, mix_net_hidden_dim * 1))
        # output V(s)
        self.hyper_B2 = nn.Sequential(
            nn.Linear(state_dim, mix_net_hidden_dim),
            nn.ReLU(),
            nn.Linear(mix_net_hidden_dim, 1)
        )

    def forward(self, qs, s):
        qs = qs.view(-1, 1, self.n_agents)
        w1 = torch.abs(self.hyper_W1(s)).view(-1, self.n_agents, self.mix_net_hidden_dim)
        b1 = self.hyper_B1(s).view(-1, 1, self.mix_net_hidden_dim)

        hidden = F.elu(qs @ w1 + b1)

        w2 = torch.abs(self.hyper_W2(s)).view(-1, self.mix_net_hidden_dim, 1)
        b2 = self.hyper_B2(s).view(-1, 1, 1)

        q_total = hidden @ w2 + b2

        return q_total.view(-1, 1)


class QMIX:

    def __init__(self, env, n_agents, observation_dim, state_dim, action_dim,
                 hidden_dim, hyper_net_hidden_dim, mix_net_hidden_dim, lr, target_update,
                 begin_epsilon, end_epsilon, epsilon_anneal_time, gamma, device):
        self.q_net = QNet(observation_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = QNet(observation_dim, hidden_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.mixer = QMixer(state_dim, n_agents, hyper_net_hidden_dim, mix_net_hidden_dim).to(device)
        self.target_mixer = QMixer(state_dim, n_agents, hyper_net_hidden_dim, mix_net_hidden_dim).to(device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.optimizer = optim.Adam(list(self.q_net.parameters()) +
                                    list(self.mixer.parameters()), lr=lr)

        self.env = env
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.gamma = gamma
        self.device = device
        self.target_update = target_update
        self.epsilon_delta = (end_epsilon - begin_epsilon) / epsilon_anneal_time
        self.epsilon = begin_epsilon
        self.end_epsilon = end_epsilon
        self.device = device
        self.total_step = 0
        self.count = 0

        print({key: value for key, value in locals().items() if key not in ['self']})

    def take_action(self, observations, eval=False):
        actions = []
        with torch.no_grad():
            for i in range(self.n_agents):
                if np.random.random() > self.epsilon or eval:
                    logits = self.q_net(torch.FloatTensor(observations[i]).to(self.device)).squeeze()
                    avail_actions = torch.tensor(self.env.get_avail_agent_actions(i)).to(self.device)
                    logits[avail_actions == 0] = float('-inf')
                    actions.append(logits.argmax().item())
                else:
                    actions.append(np.random.choice(np.nonzero(env.get_avail_agent_actions(i))[0]))

        if not eval:
            self.epsilon = max(self.epsilon + self.epsilon_delta, self.end_epsilon)
            self.total_step += 1
        return actions

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device)
        observations = torch.tensor(transition_dict['observations'],
                                    dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        next_observations = torch.tensor(transition_dict['next_observations'],
                                         dtype=torch.float).to(self.device)
        next_avail_actions = torch.tensor(transition_dict['next_avail_actions']).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        with torch.no_grad():
            # (B,n_agents,obs_dim) -> (B,n_agents,action_dim) -> (B,n_agents) -> (B,1)
            # Double DQN
            next_qs = self.q_net(next_observations)
            next_qs[next_avail_actions == 0] = float('-inf')
            max_action = next_qs.argmax(dim=-1, keepdim=True)
            max_next_qs = self.target_q_net(next_observations).gather(2, max_action).squeeze(2)
            max_next_q_total = self.target_mixer(max_next_qs, next_states)
            q_total_target = rewards + self.gamma * max_next_q_total * (1 - dones)

        qs = self.q_net(observations).gather(2, actions.unsqueeze(2)).squeeze(2)
        q_total = self.mixer(qs, states)

        loss = F.mse_loss(q_total_target, q_total)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.count += 1

    def save(self, folder='models'):
        torch.save(self.q_net.state_dict(), folder + "/qmix_q_net")
        torch.save(self.mixer.state_dict(), folder + "/qmix_mixer")

    def load(self, folder='models'):
        self.q_net.load_state_dict(torch.load(folder + '/qmix_q_net', map_location='cpu'))
        self.mixer.load_state_dict(torch.load(folder + '/qmix_mixer', map_location='cpu'))


def evaluate(agent, eval_env, eval_episodes=20):
    avg_reward = 0.
    won = 0.
    for episode in range(eval_episodes):
        obs, _ = eval_env.reset()
        obs = concat_agent_id(obs)
        terminated, info = False, None
        while not terminated:
            action = agent.take_action(obs, eval=True)
            reward, terminated, info = eval_env.step(action)
            avg_reward += reward
            obs = concat_agent_id(eval_env.get_obs())
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

alg_name = 'QMIX'
lr = 5e-4
num_episodes = 5000
hidden_dim = 128
hyper_net_hidden_dim = 128
mix_net_hidden_dim = 64
gamma = 0.99
buffer_size = 5e4
target_update = 500
minimal_size = 1000
batch_size = 64
update_interval = 1
begin_epsilon = 1.0
end_epsilon = 0.05
epsilon_anneal_time = 10000
save_model = True

# 星际争霸2 -- SMAC 环境介绍 https://zhuanlan.zhihu.com/p/595500237
from smac.env import StarCraft2Env

env = StarCraft2Env(map_name="3m", seed=0, obs_instead_of_state=True)
env_info = env.get_env_info()
print('env_info: ', env_info)

n_agents = env_info['n_agents']
observation_dim = env.get_obs_size() + n_agents
state_dim = env.get_state_size()
# 0:noop 1:stop 2:north 3:south 4:east 5:west 6-N:enemy id
action_dim = env.get_total_actions()


def concat_agent_id(observations):
    one_hot_ids = np.eye(len(observations), dtype=np.float32)
    return [np.concatenate((observations[i], one_hot_ids[i])) for i in range(len(observations))]


replay_buffer = utils.ReplayBuffer(buffer_size)
agent = QMIX(env, n_agents, observation_dim, state_dim, action_dim,
             hidden_dim, hyper_net_hidden_dim, mix_net_hidden_dim, lr, target_update,
             begin_epsilon, end_epsilon, epsilon_anneal_time,
             gamma, device)

if __name__ == '__main__':
    return_list = []
    eval_list = []
    win_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):

                observations, state = env.reset()
                observations = concat_agent_id(observations)
                episode_return, terminated = 0, False
                while not terminated:
                    # avail_actions = env.get_avail_actions()
                    actions = agent.take_action(observations, eval=False)
                    reward, terminated, info = env.step(actions)
                    next_state = env.get_state()
                    next_observations = env.get_obs()
                    next_observations = concat_agent_id(next_observations)
                    next_avail_actions = env.get_avail_actions()

                    replay_buffer.add(state, observations, actions, reward,
                                      next_state, next_observations, next_avail_actions, terminated)
                    if replay_buffer.size() > minimal_size and agent.total_step % update_interval == 0:
                        b_s, b_o, b_a, b_r, b_ns, b_no, b_naa, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'observations': b_o, 'actions': b_a, 'rewards': b_r,
                                           'next_states': b_ns, 'next_observations': b_no, 'next_avail_actions': b_naa,
                                           'dones': b_d}
                        agent.update(transition_dict)

                    state = next_state
                    observations = next_observations
                    episode_return += reward
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'total_step': '%d' % agent.total_step,
                                      'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                if (i_episode + 1) % 100 == 0:
                    eval_return, win_rate = evaluate(agent, env, eval_episodes=20)
                    eval_list.append(eval_return)
                    win_list.append(win_rate)
                pbar.update(1)
            if save_model: agent.save()
    utils.dump(f'./results/{alg_name}.pkl', return_list)
    utils.show(f'./results/{alg_name}.pkl', alg_name)
    utils.dump(f'./results/{alg_name}_eval.pkl', eval_list)
    utils.show(f'./results/{alg_name}_eval.pkl', f'{alg_name} eval')
    utils.dump(f'./results/{alg_name}_win.pkl', win_list)
    utils.show(f'./results/{alg_name}_win.pkl', f'{alg_name} win rate')
