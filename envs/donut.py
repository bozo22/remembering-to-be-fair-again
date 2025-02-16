import gym
import random
import numpy as np
from numpy.typing import NDArray
from gym.spaces import Discrete, MultiBinary
from itertools import product
from core.aggregations import Aggregation, NSW


class Donut(gym.Env):
    def __init__(
        self,
        people: int,
        episode_length: int,
        state_mode: str = "full",
        seed: int = 42,
        aggregation: Aggregation | None = None,
        p: NDArray | None = None,
        distribution: str | None = None,
        binarize_memory: bool = True,
        dynamic_prob: bool = False,
    ) -> None:
        # full: number of donuts for each person so far as a list [d1, d2, ...]
        # compact: full but as one number
        # binary: binary state of full
        # reset: number of donuts for person i - min number of donuts

        self.people = people
        self.episode_length = episode_length
        self.state_mode = state_mode
        self.aggregation = aggregation if aggregation is not None else NSW()
        self.distribution = distribution
        self.d_param1 = [50, 50, 50, 75, 25]
        self.d_param2 = [0.9, -0.9, 0.1, 0.6, 0.5]
        self.current_step = 0

        self.dynamic_prob = dynamic_prob
        if self.dynamic_prob:
            print("Dynamic probability is enabled.")
        self.prob_tracker = {}
        self.initial_prob = {}

        # Action space: Discrete choice between customers
        self.action_space = Discrete(self.people, seed=seed)

        # Observation space:
        # No memory: Binary representation of customers at the counter -> shape (people,)
        # With memory: Binary representation of customers at the counter + number of donuts given to each -> shape (2 * people,)
        memory_size = people
        if binarize_memory:
            memory_size = self.people * int(np.ceil(np.log2(self.episode_length + 1)))
        if state_mode == "none":
            memory_size = 0
        self.observation_space = Discrete(people + memory_size, seed=seed)

        # Set the initial state and memory
        self.state = np.zeros(self.people, dtype=np.int32)
        self.memory = np.array([0 for _ in range(people)], dtype=np.int32)

        # Set customer probabilities
        if p is None:
            self.prob = [0.8 for _ in range(self.people)]
        else:
            self.prob = p

        for i in range(self.people):
            self.initial_prob[i] = self.prob[i]

        # Set running values
        self.running_values = ["donuts_allocated"]
        self.running_values_done = []

        # Reset the environment
        self.reset()

    def logistic_prob(self, t, t_mid, steepness):
        """Logistic probability function."""

        return 1.0 / (1.0 + np.exp(-steepness * (t - t_mid)))

    def bell_prob(self, t, mu, sigma):
        """Gaussian probability function."""

        return np.exp(-((t - mu) ** 2) / (2.0 * sigma**2))

    def uniform_interval_prob(self, t, start, end):
        """Uniform probability function."""

        if t >= start and t <= end:
            prob = 1.0
        else:
            prob = 0.0
        return prob

    def binarize_memory(self, memory: NDArray) -> NDArray:
        zero_fill = int(np.ceil(np.log2(self.episode_length)))
        ans = ""
        for i in memory.astype(int):
            ans += bin(i)[2:].zfill(zero_fill)

        # print(s, "***", ans)
        int_ans = []
        for t in ans:
            int_ans.append(int(t))
        return np.array(int_ans, dtype=np.float32)

    def get_transformed_memory(self, memory: NDArray | None = None) -> NDArray:
        """Transform memory based on state mode.

        Args:
            memory (NDArray | None, optional): The memory to transform. If None, self.memory is used. Defaults to None.

        Returns:
            NDArray: The transformed memory.
        """

        # Use self.memory if needed
        self_memory = False
        if memory is None:
            memory = self.memory
            self_memory = True
        memory = memory.copy()

        # Full: no processing needed
        if self.state_mode == "full":
            pass

        # Min: subtract min value
        elif self.state_mode == "min":
            memory -= np.min(memory)
            assert memory is not None

        # Reset: reset to 0 when elements are equal
        elif self.state_mode == "reset":
            if self_memory and np.all(memory == memory[0]):
                self.memory = np.zeros_like(memory)

        # None: no memory
        elif self.state_mode == "none":
            memory = np.array([])

        # if memory.sum() > 0:
        #     memory /= memory.sum()
        #     assert memory is not None

        if self.binarize_memory:
            memory = self.binarize_memory(memory)

        return memory

    def get_transition(
        self,
        state: NDArray,
        memory: NDArray,
        action: int | NDArray,
        episode: int,
    ) -> tuple[NDArray, NDArray, float, dict]:
        """Get a transition. Simulate the donut distribution and calculate the reward.

        Args:
            state (NDArray): The current state.
            memory (NDArray): The current memory.
            action (int | NDArray): The action taken.
            episode (int): The current episode.

        Returns:
            tuple[NDArray, NDArray, float, dict]: The next state, next memory, the reward, and the info dictionary.
        """

        # Simulate donut distribution
        drop = True
        new_memory = memory.copy()
        if state[action]:
            drop = False
            new_memory[action] += 1


            if self.dynamic_prob:
                self.prob[action] = min(self.prob[action] + 0.1, 1.0) 
                self.prob_tracker[action] = 0  # Reset timer for recovery

        # Update probability tracker (increment counters for all tracked people)
        for i in list(self.prob_tracker.keys()):
            self.prob_tracker[i] += 1  # Increment step counter

            # If 2 steps have passed, restore probability to initial value
            if self.prob_tracker[i] > 2:
                self.prob[i] = self.initial_prob[i]  # Restore probability
                del self.prob_tracker[i] 

        # Get next state
        new_state = np.zeros_like(state)
        for i in range(self.people):
            p = random.random()
            if self.distribution == "logistic":
                self.prob[i] = self.logistic_prob(
                    episode,
                    self.d_param1[i],  # middle point
                    self.d_param2[i],  # steepness
                )
            elif self.distribution == "bell":
                self.prob[i] = self.bell_prob(
                    episode,
                    self.d_param1[i],  # mean
                    self.d_param2[i],  # std
                )
            elif self.distribution == "uniform-interval":
                self.prob[i] = self.uniform_interval_prob(
                    episode,
                    self.d_param1[i],  # start
                    self.d_param2[i],  # end
                )
            if p <= self.prob[i]:
                new_state[i] = 1
            else:
                new_state[i] = 0

        utilities = new_memory.copy()
        reward = 0 if drop else self.aggregation(utilities)

        info = {}
        info["donuts_allocated"] = 0 if drop else 1

        return new_state, new_memory, reward, info

    def get_counterfactual_transitions(
        self,
        state: NDArray,
        actual_state: NDArray,
        action: int | NDArray,
        actual_memory: NDArray,
        schedule_step: int,
        n_counterfactuals: int,
    ) -> list[tuple[NDArray, NDArray, int, float, NDArray, NDArray]]:
        
        all_possible = []
        actual_memory = actual_memory.astype(np.int32)

        for i in range(len(actual_memory)):
            tmp = []
            ed = min(self.episode_length, actual_memory[i] + 2)
            for j in range(actual_memory[i] + 1, ed):
                tmp.append(j)
            all_possible.append(tmp)
        possible_memories = list(product(*all_possible))

        original_prob = self.prob.copy()
        original_tracker = self.prob_tracker.copy()
        
        transitions = []
        n_counterfactuals = min(n_counterfactuals, len(possible_memories))
        
        for i in range(n_counterfactuals):
            
            cf_memory = np.array(list(possible_memories[i]), dtype=np.float32)
            
            if self.dynamic_prob:
                temp_prob = original_prob.copy()  
                temp_tracker = original_tracker.copy()

                for j in range(self.people):
                    if cf_memory[j] > actual_memory[j]:  
                        temp_prob[j] = min(temp_prob[j] + 0.1, 1.0)
                        temp_tracker[j] = 0 
                
                for j in list(temp_tracker.keys()):
                    temp_tracker[j] += 1
                    if temp_tracker[j] > 2:
                        temp_prob[j] = self.initial_prob[j]
                        del temp_tracker[j]
                backup_prob = self.prob.copy()
                self.prob = temp_prob.copy()
                


            new_state, new_memory, reward, _ = self.get_transition(
                actual_state, cf_memory, action, schedule_step
            )
            if self.dynamic_prob:
                self.prob = backup_prob.copy()

            cf_memory = self.get_transformed_memory(cf_memory)
            new_memory = self.get_transformed_memory(new_memory)

            transitions.append(
                (state, cf_memory, action, reward, new_state, new_memory)
            )
        
        self.prob = original_prob.copy()
        self.prob_tracker = original_tracker.copy()

        return transitions

    def step(self, action: int) -> tuple[NDArray, float, bool, bool, dict]:
        self.current_step += 1
        state = self.state
        memory = self.memory

        new_state, new_memory, reward, info = self.get_transition(
            state, memory, action, self.current_step
        )

        done = self.current_step >= self.episode_length
        self.state = new_state
        self.memory = new_memory

        new_memory = self.get_transformed_memory()
        obs = np.concatenate((new_state, new_memory))
        info["state"] = new_state.copy()
        info["memory"] = new_memory.copy()

        return obs, reward, done, False, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[NDArray, dict]:

        self.memory = np.zeros(self.people, dtype=np.float32)
        self.state = np.zeros(self.people, dtype=np.float32)
        self.current_step = 0

        for i in range(self.people):
            p = random.random()
            if p <= self.prob[i]:
                self.state[i] = 1
            else:
                self.state[i] = 0

        memory = self.get_transformed_memory()
        obs = np.concatenate((self.state, memory))
        info = {
            "state": self.state.copy(),
            "memory": memory.copy(),
            "donuts_allocated": 0,
        }

        return obs, info
