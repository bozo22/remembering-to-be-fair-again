import argparse
from itertools import product

import gym
import torch
import numpy as np
from torch import nn


class CovidSEIREnv(gym.Env):
    """
    A multi-region COVID SEIR environment following the OpenAI Gym interface.

    State representation (for k regions, flattened into one vector of length 4*k):
      Region 1: S1, E1, I1, R1, death_rate, population
      Region 2: S2, E2, I2, R2, death_rate, population
      ...
      Region k: Sk, Ek, Ik, Rk

    At each timestep:
      1. A known number of vaccines is produced (vaccine_schedule[t]).
      2. The agent allocates these vaccines among k regions using a continuous vector action:
         a[i] in [0,1], sum(a[i]) = 1.
      3. We move vaccinated individuals from S -> R (assuming perfect vaccine efficacy).
      4. We perform the SEIR update for each region.

    The reward is (by default) the negative sum of all infected fractions: we want to minimize the total infection across regions.

    You can customize:
      - vaccine allocation logic (efficacy, partial take, etc.)
      - infection dynamics (beta, sigma, gamma can be region-specific or time-varying)
      - reward function (cost of high infection, cost of unused vaccines, etc.)
      - done condition (stop when infection dies out, or after a fixed horizon, etc.)
    """

    metadata = {"render.modes": ["human"]}

    # TODO: Model death rate and try to make death proportions similar to each other
    def __init__(
            self,
            k=3,
            population=None,  # array-like of length k, or a single scalar
            beta=0.3,  # can be scalar or array-like of length k
            sigma=0.2,  # can be scalar or array-like of length k
            gamma=0.1,  # can be scalar or array-like of length k
            vaccine_schedule=None,  # array-like of length max_steps with # vaccines available each step
            max_steps=500,
            allocation_step=0.2,  # granularity of vaccine allocation fractions
            init_states=None,
    ):
        super(CovidSEIREnv, self).__init__()

        # Number of regions
        self.k = k

        # Handle population
        if population is None:
            self.population = np.array([1_000_000] * k, dtype=np.float32)
        else:
            population = np.array(population, dtype=np.float32)
            if population.shape == ():  # single scalar
                self.population = np.array([population] * k, dtype=np.float32)
            else:
                self.population = population

        # We hold these constant for now but these can be changed
        # Contact rate(β) changes as people alter their behavior(e.g., lockdowns, masking, vaccination
        # coverage, new variants).

        # Incubation rate(σ), which governs the transition from Exposed to Infected, can vary slightly
        # if a new variant has a different incubation period.

        # Recovery rate (γ) can shift if healthcare interventions improve or if new variants affect severity.

        # Handle beta (contact rate), sigma (incubation), gamma (recovery)
        # If user provides a single scalar, convert to array of length k.
        # Otherwise, assume it is already array-like of length k.
        def _param_to_array(param):
            param_array = np.array(param, dtype=np.float32)
            if param_array.shape == ():
                # scalar
                return np.array([param_array] * k, dtype=np.float32)
            else:
                return param_array

        # SEIR parameters
        self.beta = _param_to_array(beta)
        self.sigma = _param_to_array(sigma)
        self.gamma = _param_to_array(gamma)

        # Vaccine schedule
        if vaccine_schedule is None:
            self.vaccine_schedule = np.zeros(max_steps, dtype=np.float32)
        else:
            self.vaccine_schedule = np.array(vaccine_schedule, dtype=np.float32)
            if len(self.vaccine_schedule) < max_steps:
                padding = np.zeros(max_steps - len(self.vaccine_schedule), dtype=np.float32)
                self.vaccine_schedule = np.concatenate([self.vaccine_schedule, padding])
        self.max_steps = max_steps

        # Define discrete allocation fractions
        self.allocation_fractions = np.arange(0.0, 1.01, allocation_step)

        # Define all possible allocations that sum to 1
        all_possible_allocations = [
            allocation for allocation in product(self.allocation_fractions, repeat=self.k)
            if np.isclose(sum(allocation), 1.0, atol=1e-6)
        ]

        self.allocation_mapping = np.array(all_possible_allocations, dtype=np.float32)

        # Action space: Discrete index into the allocation mapping
        self.action_space = gym.spaces.Discrete(len(self.allocation_mapping))

        # Observation space: 4 compartments per region -> shape (4*k,)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4 * k,), dtype=np.float32
        )

        self.current_step = 0
        self.state = None
        # Make it so it stores the previous states
        self.memory = np.zeros((self.k,))
        self.init_states = init_states
        self.reset(self.init_states)

    def step(self, action):
        """
        Take one step in the environment:
          1. Allocate vaccines among k regions.
          2. Move vaccinated individuals from S -> R (assuming perfect efficacy).
          3. Apply SEIR updates in each region.
          4. Compute reward, done, info.
        """
        # -- 1. Distribute vaccines --

        # Map action index to allocation vector
        allocation = self.allocation_mapping[action]

        # Proceed with vaccine allocation and SEIR updates
        total_vaccines = self.vaccine_schedule[self.current_step]
        allocation = allocation * total_vaccines  # Scale allocation by available vaccines

        # Current state is shape (4*k,)
        # We'll reshape it into (k,4) for clarity in calculations
        # region_state[i] = [S, E, I, R] for region i
        region_state = self.state.reshape((self.k, 4))

        # -- 2. Vaccinate (move from S -> R) --
        for i in range(self.k):
            pop_i = self.population[i]
            # The fraction of the region's population that can be vaccinated
            # (just naive: min of S[i]*pop_i or # of allocated vaccines).
            # Then convert that number of vaccines to fraction of the region's pop.
            max_possible_vaccines = region_state[i, 0] * pop_i  # S fraction * population
            used_vaccines = min(allocation[i], max_possible_vaccines)
            # Convert used_vaccines to fraction out of population
            used_vaccines_frac = used_vaccines / pop_i

            # Move that fraction from S to R
            region_state[i, 0] -= used_vaccines_frac  # S
            region_state[i, 3] += used_vaccines_frac  # R

            # Clip to ensure no numeric drift
            region_state[i, 0] = np.clip(region_state[i, 0], 0.0, 1.0)
            region_state[i, 3] = np.clip(region_state[i, 3], 0.0, 1.0)


        # -- 3. SEIR updates for each region --
        # Basic compartmental update (Euler discrete approximation)
        S_new = np.zeros(self.k, dtype=np.float32)
        E_new = np.zeros(self.k, dtype=np.float32)
        I_new = np.zeros(self.k, dtype=np.float32)
        R_new = np.zeros(self.k, dtype=np.float32)

        for i in range(self.k):
            S_i, E_i, I_i, R_i = region_state[i]

            beta_i = self.beta[i]
            sigma_i = self.sigma[i]
            gamma_i = self.gamma[i]

            dS = -beta_i * S_i * I_i
            dE = sigma_i * S_i * I_i - sigma_i * E_i
            dI = sigma_i * E_i - gamma_i * I_i
            dR = gamma_i * I_i

            # Update
            S_new[i] = S_i + dS
            E_new[i] = E_i + dE
            I_new[i] = I_i + dI
            R_new[i] = R_i + dR

        # Ensure fractions stay in [0, 1]
        S_new = np.clip(S_new, 0.0, 1.0)
        E_new = np.clip(E_new, 0.0, 1.0)
        I_new = np.clip(I_new, 0.0, 1.0)
        R_new = np.clip(R_new, 0.0, 1.0)

        # Recombine
        region_state = np.stack([S_new, E_new, I_new, R_new], axis=1)
        self.state = region_state.flatten()

        # -- 4. Reward: e.g., negative sum of suscepted fractions across all k regions --
        total_infected = np.sum(I_new)
        # print(f"prod{np.sum(S_new*self.population)}")
        sum_sus = np.sum((S_new+E_new)*self.population)
        # print(f"reward: {min(self.memory)}")
        reward = -sum_sus  # we want to minimize suscepted population
        # Negative product of suscepted (nash)
        # Episode termination condition
        self.current_step += 1
        done = (self.current_step >= self.max_steps)

        # Info dictionary for debugging
        info = {
            "total_infected": float(total_infected),
            "vaccines_allocated": allocation
        }
        self.memory += allocation
        # print(f"allocation: {allocation}")
        return self.state, self.memory, float(reward), done, info

    def reset(self, init_states=[]):
        """
        Resets the environment to an initial state.
        For example, each region starts with:
          80% susceptible, 10% exposed, 10% infected, 0% recovered.
        """
        self.current_step = 0
        if len(init_states) == 0:
            print("nonooosdfsdf")
            init_states = np.repeat(np.array([0.8, 0.1, 0.1, 0.0])[np.newaxis, :], self.k, axis=0)

        # Build initial (k,4) array
        region_init = []
        for i in range(self.k):
            region_init.append(init_states[i])

        region_init = np.array(region_init, dtype=np.float32)
        self.state = region_init.flatten()
        self.memory = np.zeros((self.k, ))
        # print("----------------Env reset--------------------")
        return self.state, self.memory

    def render(self, mode="human"):
        """
        Render each region's SEIR fractions.
        """
        region_state = self.state.reshape((self.k, 4))
        print(f"Step {self.current_step}")
        for i in range(self.k):
            S_i, E_i, I_i, R_i = region_state[i]
            print(f"  Region {i}: S={S_i:.4f}, E={E_i:.4f}, I={I_i:.4f}, R={R_i:.4f}, Vaccines allocated so far={self.memory[i]}")
        print()

    def close(self):
        pass

class Net(nn.Module):
    def __init__(self, states, actions):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(states, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, actions))

    def forward(self, x):
        action_prob = self.model(x)
        return action_prob
class DQN():
    def __init__(self, num_states, num_actions, memory_capacity, learning_rate, args):
        super(DQN, self).__init__()

        self.eval_net, self.target_net = Net(num_states, num_actions), Net(num_states, num_actions)

        def init_weights(m):
            if hasattr(m, 'weight'):
                nn.init.orthogonal_(m.weight.data)
            if hasattr(m, 'bias'):
                nn.init.constant_(m.bias.data, 0)

        self.eval_net.apply(init_weights)

        self.target_net.load_state_dict(
            self.eval_net.state_dict())

        self.num_states = num_states
        self.num_actions = num_actions
        self.memory_capacity = memory_capacity
        self.args = args

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((memory_capacity, num_states * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()


    def choose_action(self, state, greedy=False):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if greedy:
            with torch.no_grad():
                action_value = self.target_net.forward(state)
                action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]

        elif np.random.uniform() >= self.args.epsilon:  # greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]

        else:  # random policy
            action = np.random.randint(0, self.num_actions)
            action = action
        return action

    def store_transition(self, state, memory, action, reward, next_state, next_memory):
        transition = np.hstack((state, memory, [action, reward], next_state, next_memory))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.args.q_network_iterations == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.memory_capacity, self.args.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.num_states])
        batch_action = torch.LongTensor(batch_memory[:, self.num_states:self.num_states + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.num_states + 1:self.num_states + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.num_states:])

        q_eval = self.eval_net(batch_state).gather(1, batch_action)

        with torch.no_grad():
            q_next = self.target_net(batch_next_state).detach()
            max_q_next = q_next.max(1)[0].view(self.args.batch_size, 1)

            # # This implements double DQN, which will stabilize training?
            # # Action selection using eval_net
            # max_action_selection = self.eval_net(batch_next_state).max(1)[1].view(-1, 1)
            # # Q-value evaluation using target_net
            # max_q_next = self.target_net(batch_next_state).gather(1, max_action_selection)

        q_target = batch_reward + self.args.gamma * max_q_next
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def counterfactual_update(
            self,
            env,
            state,
            action,
            prev_reward,
            next_state,
            actual_memory,
            max_ep_len,
            state_mode='binary',
            num_updates=2
    ):
        pass


def run(k, max_ep_len, memory_capacity, args, seed=42):
    max_steps = 24
    # Suppose at each step we produce 50,000 vaccines for 10 steps
    # vaccine_schedule = [1_000] * max_steps
    #approximate vaccine production schedule from https://www.ifpma.org/news/as-covid-19-vaccine-output-estimated-to-reach-over-12-billion-by-year-end-and-24-billion-by-mid-2022-innovative-vaccine-manufacturers-renew-commitment-to-support-g20-efforts-to-address-remaining-barr/
    vaccine_schedule = (np.arange(0,max_steps)**2 * 0.08)*3_000_000
    infected_records = np.ones((args.episodes, max_steps, k))
    init_state_0 = [0.99, 0.01, 0.0, 0.0]
    init_state_1 = [0.8, 0.1, 0.1, 0.0]
    init_state_2 = [0.75, 0.1, 0.15, 0.0]
    init_states = np.array([init_state_0, init_state_1, init_state_2])
    env = CovidSEIREnv(k=k, population=[700_000_000, 200_000_000, 100_000_000], vaccine_schedule=vaccine_schedule,
                       max_steps=max_steps, beta=[0.33, 0.22, 0.18], gamma=[0.262, 0.085, 0.087], sigma=0.2,
                       init_states=init_states)

    num_actions = env.action_space.n
    state, memory = env.reset(init_states)
    num_states = len(state) + len(memory)

    dqn = DQN(num_states, num_actions, memory_capacity, args.lr, args)

    episodes = args.episodes
    print("Collecting Experience....")
    reward_list = []
    donuts_list = []
    all_rewards = np.zeros((episodes, max_steps))
    print(all_rewards.shape)
    for i in range(episodes):
        state, memory = env.reset(init_states)
        ep_reward = 0
        ep_donuts = 0

        while True:
            state_input = state.copy()
            state_input = np.concatenate((state_input, memory))
            action = dqn.choose_action(state_input)
            actual_memory = env.memory.copy()
            next_state, next_memory, reward, done, info = env.step(action)
            dqn.store_transition(state, memory, action, reward, next_state, next_memory)
            # if args.counterfactual:
            #     dqn.counterfactual_update(env, state, action, reward, next_state, actual_memory, max_ep_len,
            #                               args.state_mode, args.num_updates)
            ep_reward += reward
            if reward != 0:
                ep_donuts += 1

            if dqn.memory_counter >= memory_capacity:
                dqn.learn()
                if done and i % 100 == 0:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                break
            state = next_state
            memory = next_memory


        if dqn.args.epsilon > 0.2:
            dqn.args.epsilon = dqn.args.epsilon * 0.999
        ep_reward = 0
        ep_donuts = 0

        state, memory = env.reset(init_states)
        state_input = state.copy()
        state_input = np.concatenate((state_input, memory))
        ep_reward = 0
        # print("-------------------Evaluating-------------------")

        curr_step = 0

        while True:
            action = dqn.choose_action(state_input, True)

            next_state, next_memory, reward, done, info = env.step(action)
            reshaped_next_state = next_state.reshape((k, 4))
            infected_records[i][curr_step] = (reshaped_next_state[:, 0]+reshaped_next_state[:, 1])*env.population
            print(f"reward: {reward}")
            all_rewards[i][curr_step] = reward
            curr_step += 1
            ep_reward += reward
            if reward != 0:
                ep_donuts += 1
            if done:
                print("------------------")
                break
            state = next_state
            memory = next_memory
            state_input = state.copy()
            state_input = np.concatenate((state_input, memory))
            env.render()
        # if i % 10 == 0:
            # print(i, "-------------------------------")
        reward_list.append(ep_reward)
        donuts_list.append(ep_donuts)
    env.close()
    return reward_list, donuts_list, all_rewards, infected_records


import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt

def plot_mean_and_std(rewards_array, name="example"):
    """
    Plot the rewards at each step over all episodes for each element and overlay the mean with standard deviation.

    Args:
        rewards_array (np.ndarray): A 3D numpy array of shape [num_episodes, num_steps, num_elements]
                                     containing rewards for multiple episodes and multiple elements.
    """
    # Get the number of episodes, steps, and elements
    num_episodes, num_steps, num_elements = rewards_array.shape

    # Create a plot
    plt.figure(figsize=(12, 8))
    steps = np.arange(num_steps)

    # Plot each individual element's mean reward per episode
    for element in range(num_elements):
        element_rewards = rewards_array[:, :, element]
        mean_rewards = np.mean(element_rewards, axis=0)
        std_rewards = np.std(element_rewards, axis=0)

        # Plot mean reward and standard deviation for this element
        plt.plot(steps, mean_rewards, label=f"Element {element + 1} Mean Reward")
        plt.fill_between(steps, mean_rewards - std_rewards, mean_rewards + std_rewards,
                         alpha=0.2, label=f"Element {element + 1} Std Dev")

    # Add labels, title, and legend
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.title("Rewards at Each Step for Each Element")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(f"{name}.png")



if __name__ == "__main__":
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Fair Covid""")
    prs.add_argument("-ep", dest="episodes", type=int, default=300, required=False, help="episodes.\n")
    prs.add_argument("-lr", dest="lr", type=float, default=0.0001, required=False, help="learning rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=1.0, required=False, help="Exploration rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.95, required=False, help="Discount factor\n")
    prs.add_argument("-sm", dest="state_mode", type=str, default='deep', required=False,
                     help="State representation mode\n")
    prs.add_argument("-cf", dest="counterfactual", type=bool, default=False, required=False,
                     help="Counterfactual Update\n")
    prs.add_argument("-bs", dest="batch_size", type=int, default=64, required=False,
                     help="Batch Size\n")
    prs.add_argument("-qiter", dest="q_network_iterations", type=int, default=1000, required=False,
                     help="Q network iterations\n")
    prs.add_argument("-nexp", dest="num_exps", type=int, default=1, required=False,
                     help="Number of Experiments\n")
    prs.add_argument("-nupds", dest="num_updates", type=int, default=2, required=False,
                     help="Number of updates to look forward in to generate counterfactual experiences\n")
    args = prs.parse_args()

    reward_t, donut_t, rewards_to_plot, infected_records = run(k=3, max_ep_len=10, memory_capacity=400, args=args)
    plot_mean_and_std(np.expand_dims(rewards_to_plot, axis=-1), "rewards")
    plot_mean_and_std(infected_records, "infected_records")

    print(infected_records[-1][-1])
    # k = 3
    # max_steps = 10
    # # Suppose at each step we produce 50,000 vaccines for 10 steps
    # vaccine_schedule = [5_000] * max_steps
    #
    # env = CovidSEIREnv(k=k, population=[10_000, 100_000, 890_000], vaccine_schedule=vaccine_schedule, max_steps=max_steps)

    # done = False
    # while not done:
    #     # Action: random allocation in [0,1]
    #     action = env.action_space.sample()
    #     obs, reward, done, info = env.step(action)
    #     env.render()

