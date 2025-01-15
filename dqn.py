import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product

matplotlib.use("Agg")


class Net_linear_donut(nn.Module):
    def __init__(self, states, actions, device):
        super(Net_linear_donut, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(states, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, actions),
        )

    def forward(self, x):
        action_prob = self.model(x)
        return action_prob


class Net_linear_lending(nn.Module):

    def __init__(self, states, actions, device):
        super(Net_linear_lending, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(states, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, actions),
        )

    def forward(self, x):
        action_prob = self.model(x)
        return action_prob


class Net_RNN(nn.Module):
    def __init__(self, states, actions, device, batch_size=256, hidden_size=16):
        super(Net_RNN, self).__init__()
        self.layer1 = nn.Linear(states, 32)
        self.layer2 = nn.Linear(32, 16)
        self.rnn = nn.GRU(16, hidden_size, batch_first=True)
        self.out_layer = nn.Linear(16, actions)
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = device

    def forward(self, x, prev_hidden=None):
        if prev_hidden == None:
            prev_hidden = torch.zeros([1, self.hidden_size]).to(self.device)

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x, hidden = self.rnn(x, prev_hidden)
        x = self.out_layer(x)
        return x, hidden


class DQN:
    def __init__(
        self,
        num_states,
        num_actions,
        memory_capacity,
        learning_rate,
        device,
        env_type,
        net_type,
        args,
    ):
        super(DQN, self).__init__()

        self.device = device
        self.env_type = env_type
        self.net_type = net_type

        # Choose the network based on the environment and network type
        net_class = Net_linear_donut if self.env_type == "donut" else Net_linear_lending
        if self.net_type == "rnn":
            net_class = Net_RNN

        self.eval_net, self.target_net = net_class(
            num_states, num_actions, device
        ), net_class(num_states, num_actions, device)
        self.eval_net.to(self.device)
        self.target_net.to(self.device)

        def init_weights(m):
            if hasattr(m, "weight"):
                nn.init.orthogonal_(m.weight.data)
            if hasattr(m, "bias"):
                nn.init.constant_(m.bias.data, 0)

        def init_weights_rnn(m):
            classname = m.__class__.__name__
            if classname.find("Linear") != -1:
                m.weight.data.normal_(0, 1)
                m.weight.data *= 1 / torch.sqrt(
                    m.weight.data.pow(2).sum(1, keepdim=True)
                )
                if m.bias is not None:
                    m.bias.data.fill_(0)

        if self.net_type == "linear":
            self.eval_net.apply(init_weights)
        else:
            self.eval_net.apply(init_weights_rnn)

        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.num_states = num_states
        self.num_actions = num_actions
        self.memory_capacity = memory_capacity
        self.args = args

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = torch.zeros((memory_capacity, num_states * 2 + 2)).to(self.device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, hidden=None, greedy=False):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)
        hidden = None if hidden is None else hidden.to(self.device)

        with torch.no_grad():
            if hidden is not None:
                action_value, hidden = self.eval_net.forward(state, hidden)
                action_value = action_value.cpu()
            else:
                action_value = self.eval_net.forward(state).cpu()

        if greedy or np.random.uniform() >= self.args.epsilon:  # greedy policy
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]

        else:  # random policy
            action = np.random.randint(0, self.num_actions)
            action = action

        return action, hidden

    def store_transition(
        self, state, action, reward, next_state, memory=None, next_memory=None
    ):
        if memory is not None and next_memory is not None:
            transition = np.hstack(
                (state, memory, [action, reward], next_state, next_memory)
            )
        else:
            transition = np.hstack((state, [action, reward], next_state))
        transition = torch.as_tensor(transition).to(self.device, non_blocking=True)
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.args.q_network_iterations == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.memory_capacity, self.args.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_state = batch_memory[:, : self.num_states]
        batch_action = batch_memory[:, self.num_states : self.num_states + 1].to(
            torch.long
        )
        batch_reward = batch_memory[:, self.num_states + 1 : self.num_states + 2]
        batch_next_state = batch_memory[:, -self.num_states :]

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=True,
        ):
            if self.net_type == "rnn":
                q_eval, _ = self.eval_net(batch_state)
                q_eval = q_eval.gather(1, batch_action)
            else:
                q_eval = self.eval_net(batch_state).gather(1, batch_action)

            with torch.no_grad():
                if self.net_type == "rnn":
                    q_next, _ = self.target_net(batch_next_state)
                    q_next = q_next.detach()
                else:
                    q_next = self.target_net(batch_next_state).detach()
                max_q_next = q_next.max(1)[0].view(self.args.batch_size, 1)

            q_target = batch_reward + self.args.gamma * max_q_next
            loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def counterfactual_update(
        self,
        fair_env,
        state,
        action,
        prev_reward,
        next_state,
        actual_memory,
        max_ep_len,
        state_mode="binary",
        num_updates=2,
    ):
        if self.env_type == "donut":
            all_possible = []
            for i in range(len(actual_memory)):
                tmp = []
                ed = min(max_ep_len, actual_memory[i] + num_updates + 1)
                for j in range(actual_memory[i] + 1, ed):
                    tmp.append(j)
                all_possible.append(tmp)
            possible_memories = list(product(*all_possible))

            for i in range(len(possible_memories)):
                curr = list(possible_memories[i])
                reward = 0
                if curr[action] == max_ep_len:
                    continue
                next_memories = curr.copy()
                next_memories[action] += 1
                for j in range(len(curr)):
                    reward += np.log(float(next_memories[j]) + 1)
                if prev_reward == 0:
                    reward = 0

                if state_mode == "binary":
                    memory = fair_env.binary_state(curr)
                    next_memory = fair_env.binary_state(next_memories)
                else:
                    memory = curr
                    next_memory = next_memories
                if self.net_type == "linear":
                    self.store_transition(
                        state, action, reward, next_state, memory, next_memory
                    )
                else:
                    self.store_transition(state, action, reward, next_state)
        else:
            for i in range(len(actual_memory)):
                for k in range(1, num_updates + 1):
                    possible_memory = actual_memory.copy()
                    possible_memory[i] += k
                    if possible_memory[i] >= max_ep_len + 1:
                        break

                    subg = 0
                    if action > 1:
                        subg = 1
                    if possible_memory[subg] == max_ep_len:
                        continue

                    next_memories = possible_memory.copy()
                    next_memories[subg] += 1
                    reward = -1.0 * abs(possible_memory[1] - possible_memory[0])
                    if prev_reward < -1 * max_ep_len:
                        reward = prev_reward

                    if state_mode == "reset":
                        mn = min(possible_memory)
                        for j in range(2):
                            possible_memory[j] = possible_memory[j] - mn
                        mn = min(next_memories)
                        for j in range(2):
                            next_memories[j] = next_memories[j] - mn

                    memory = fair_env.binary_state(possible_memory, max_ep_len + 1)
                    next_memory = fair_env.binary_state(next_memories, max_ep_len + 1)
                    if self.net_type == "linear":
                        self.store_transition(
                            state, action, reward, next_state, memory, next_memory
                        )
                    else:
                        self.store_transition(state, action, reward, next_state)
