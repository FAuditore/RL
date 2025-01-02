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
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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


class MADDPG:

    def __init__(self, env, n_agents, observation_dims, action_dims, critic_input_dim,
                 hidden_dim, actor_lr, critic_lr, initial_random_steps,
                 tau, gamma, device):
        self.actors, self.critics, self.target_actors, self.target_critics = [], [], [], []
        for i in range(n_agents):
            self.actors.append(PolicyNet(observation_dims[i], hidden_dim, action_dims[i]).to(device))
            self.critics.append(ValueNet(critic_input_dim, hidden_dim).to(device))

            self.target_actors.append(PolicyNet(observation_dims[i], hidden_dim, action_dims[i]).to(device))
            self.target_critics.append(ValueNet(critic_input_dim, hidden_dim).to(device))

            self.target_critics[i].load_state_dict(self.critics[i].state_dict())
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())

        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=critic_lr) for critic in self.critics]

        self.env = env
        self.n_agents = n_agents
        self.action_dims = action_dims
        self.observation_dims = observation_dims
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.initial_random_steps = initial_random_steps
        self.total_step = 0

        print({key: value for key, value in locals().items() if key not in ['self']})

    def take_action(self, observations, eval=False):
        # observations: dict{agent_0:state,agent_1:state,...}
        actions = {}
        with torch.no_grad():
            observations = [torch.FloatTensor([observations[agent]]).to(self.device) for agent in self.env.agents]
            logits = [self.actors[i](observations[i]) for i in range(self.n_agents)]
            actions = {agent:
                           F.gumbel_softmax(logits[i], hard=True).argmax().item()
                           if self.total_step < self.initial_random_steps and not eval
                           else logits[i].argmax().item()
                       for i, agent in enumerate(self.env.agents)}

        self.total_step += 1
        return actions

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        with torch.no_grad():
            next_one_hot_actions = torch.cat(
                [F.one_hot(
                    ta(next_states[:, i]).argmax(dim=1),
                    num_classes=self.action_dims[i]) for i, ta in enumerate(self.target_actors)], dim=1)
            next_critic_input = torch.cat(
                [next_states.view(-1, sum(self.observation_dims)), next_one_hot_actions], dim=1)
            next_qs = [tc(next_critic_input) for tc in self.target_critics]
            qs_target = [rewards[:, i].unsqueeze(1) +
                         self.gamma * (1 - dones) * next_qs[i] for i in range(self.n_agents)]

        one_hot_actions = [F.one_hot(actions[:, i], num_classes=self.action_dims[i]) for i in range(self.n_agents)]
        critic_input = torch.cat(
            [states.view(-1, sum(self.observation_dims)), *one_hot_actions], dim=1)
        qs = [c(critic_input) for c in self.critics]

        for i, co in enumerate(self.critic_optimizers):
            critic_loss = F.mse_loss(qs[i], qs_target[i])
            co.zero_grad()
            critic_loss.backward()
            # nn.utils.clip_grad_value_(self.critic.parameters(), clip_value=.5)
            co.step()

        qs = []
        logits = [a(states[:, i]) for i, a in enumerate(self.actors)]
        # one_hot_actions = [F.one_hot(logit.argmax(dim=1), num_classes=self.action_dims[i]).detach()
        #                    for i, logit in enumerate(logits)]
        gumbel_actions = [F.gumbel_softmax(logit, hard=True)
                          for i, logit in enumerate(logits)]
        for i_agent in range(self.n_agents):
            all_actions = []
            for i in range(self.n_agents):
                if i == i_agent:
                    all_actions.append(gumbel_actions[i])
                else:
                    all_actions.append(one_hot_actions[i])  # other agents' action do not change
            critic_input = torch.cat(
                [states.view(-1, sum(self.observation_dims)), *all_actions], dim=1)
            qs.append(self.critics[i_agent](critic_input))

        for i, ao in enumerate(self.actor_optimizers):
            actor_loss = -qs[i].mean() + (logits[i] ** 2).mean() * 1e-3
            ao.zero_grad()
            # nn.utils.clip_grad_value_(self.actor.parameters(), clip_value=.5)
            actor_loss.backward()
            ao.step()

        for i in range(self.n_agents):
            self.soft_update(self.actors[i], self.target_actors[i])
            self.soft_update(self.critics[i], self.target_critics[i])

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self, folder='models'):
        torch.save([a.state_dict() for a in self.actors], folder + "/maddpg_actors")
        torch.save([c.state_dict() for c in self.critics], folder + "/maddpg_critics")

    def load(self, folder='models'):
        actor_state_dicts = torch.load(folder + '/maddpg_actors', map_location='cpu')
        critic_state_dicts = torch.load(folder + '/maddpg_critics', map_location='cpu')
        for a, sd, c, cd in zip(self.actors, actor_state_dicts,
                                self.critics, critic_state_dicts):
            a.load_state_dict(sd)
            c.load_state_dict(cd)


def evaluate(agent, eval_env, eval_episodes=10):
    avg_reward = 0.
    for episode in range(eval_episodes):
        state, info = eval_env.reset()
        while eval_env.agents:
            action = agent.take_action(state, eval=True)
            state, reward, done, truncated, info = eval_env.step(action)
            avg_reward += sum(rewards.values()) / agent.n_agents

    avg_reward /= eval_episodes

    print(f"Evaluation over {eval_episodes} episodes,"
          f"Avg_reward: {avg_reward:.3f}")
    return avg_reward


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alg_name = 'MADDPG'
actor_lr = 1e-2
critic_lr = 1e-2
num_episodes = 25000
hidden_dim = 128
gamma = 0.95
tau = 1e-2
buffer_size = 1e6
initial_random_steps = 0  # Time steps initial random policy is used
minimal_size = 1000  # Update begin step
batch_size = 64
update_interval = 100
save_model = True

from pettingzoo.mpe import simple_spread_v3

env = simple_spread_v3.parallel_env(max_cycles=25, continuous_actions=False)

n_agents = env.max_num_agents
observation_dims = []
action_dims = []
for observation_space in env.observation_spaces.values():
    observation_dims.append(observation_space.shape[0])
for action_space in env.action_spaces.values():
    action_dims.append(action_space.n)
critic_input_dim = sum(observation_dims) + sum(action_dims)

replay_buffer = utils.ReplayBuffer(buffer_size)
agent = MADDPG(env, n_agents, observation_dims, action_dims, critic_input_dim,
               hidden_dim, actor_lr, critic_lr,
               initial_random_steps, tau, gamma, device)

if __name__ == '__main__':
    return_list = []
    eval_list = []
    total_step = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):

                episode_return = 0
                obs, info = env.reset()
                while env.agents:
                    actions = agent.take_action(obs, eval=False)
                    next_obs, rewards, done, truncated, info = env.step(actions)

                    replay_buffer.add(
                        [o for o in obs.values()],
                        [a for a in actions.values()],
                        [r for r in rewards.values()],
                        [no for no in next_obs.values()],
                        True in done.values() or True in truncated.values())
                    if replay_buffer.size() > minimal_size and total_step % update_interval == 0:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s,
                                           'actions': b_a,
                                           'next_states': b_ns,
                                           'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)

                    episode_return += sum(rewards.values()) / n_agents
                    total_step += 1
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'total_step': '%d' % total_step,
                                      'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                if (i_episode + 1) % 100 == 0:
                    eval_list.append(evaluate(agent, env, eval_episodes=10))
                pbar.update(1)
            if save_model: agent.save()
    utils.dump(f'./results/{alg_name}.pkl', return_list)
    utils.show(f'./results/{alg_name}.pkl', alg_name)
    utils.dump(f'./results/{alg_name}_eval.pkl', eval_list)
    utils.show(f'./results/{alg_name}_eval.pkl', f'{alg_name} eval')
