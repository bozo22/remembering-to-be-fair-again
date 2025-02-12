import gym
from gym.spaces import Discrete, MultiBinary
import random
import numpy as np
from numpy.typing import NDArray
from core.aggregations import Aggregation, RDP


class Lending(gym.Env):
    def __init__(
        self,
        people: int,
        episode_length: int,
        seed: int,
        state_mode: str = "full",
        p: list[float] | None = None,
        aggregation: Aggregation | None = None,
        binarize: bool = True,
    ):
        self.people = people
        self.seed = seed
        self.episode_length = episode_length
        self.state_mode = state_mode
        self.binarize_obs = binarize

        # Action space: Discrete choice between customers
        self.action_space = Discrete(self.people, seed=seed)

        # Observation space:
        # No memory: Binary representation of customers at the counter + success + credit -> shape (k + 1 + k,)
        # With memory: Binary representation of customers at the counter + success + credit + memory -> shape (k + 1 + k + k // 2,)
        self.success_size = 1
        self.credit_size = people
        self.memory_size = people
        if binarize:
            self.success_size = int(np.ceil(np.log2((episode_length + 1) * 2)))
            self.credit_size = people * int(np.ceil(np.log2(7)))
            self.memory_size = (people // 2) * int(np.ceil(np.log2(episode_length + 1)))
        if state_mode == "none":
            self.memory_size = 0
        self.observation_space = Discrete(
            people + self.success_size + self.credit_size + self.memory_size, seed=seed
        )

        self.aggregation = aggregation if aggregation is not None else RDP()

        self.default_credit = np.array([4, 4, 7, 7], dtype=np.float32)
        self.credit = self.default_credit.copy()

        self.memory = np.zeros(people // 2, dtype=np.float32)
        self.success = self.episode_length

        self.current_step = 0
        self.state = np.zeros(2 * people + 1, dtype=np.float32)

        if p is None:
            self.prob = [0.9 for _ in range(self.people)]
        else:
            self.prob = p

        # Set running values
        self.running_values = []
        self.running_values_done = []

        self.reset()

    def binarize(self, s: NDArray, length: int) -> NDArray:
        zero_fill = int(np.ceil(np.log2(length)))
        ans = ""
        for i in s.astype(int):
            ans += bin(i)[2:].zfill(zero_fill)

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

        if self.binarize_obs:
            memory = self.binarize(memory, self.episode_length + 1)

        return memory

    def get_transition(
        self,
        state: NDArray,
        memory: NDArray,
        action: int | NDArray,
        episode: int,
    ) -> tuple[NDArray, NDArray, float, dict]:

        # possible actions are 0, 1, 2, 3
        done = episode >= self.episode_length
        state = state.copy()
        customers = state[: self.people]
        success = state[self.people : self.people + 1]
        credit = state[self.people + 1 :]

        wrong_action = customers[action] == 0

        subg = 0 if action <= 1 else 1

        new_memory = memory.copy()
        if not wrong_action:
            new_memory[subg] += 1
            repayment = random.random()
            if repayment <= ((credit[action] + 2) / 10):
                success += 1
                credit[action] = min(credit[action] + 1, 7)
            else:
                success -= 1
                credit[action] = max(credit[action] - 1, 0)

        customers = np.zeros_like(customers, dtype=np.float32)
        for i in range(self.people):
            p = random.random()
            if p <= self.prob[i]:
                customers[i] = 1
            else:
                customers[i] = 0

        reward = self.aggregation(new_memory)
        if wrong_action:
            reward = -1 * self.episode_length
        if done and success[0] < self.episode_length + int(self.episode_length / 10):
            reward = -10 * self.episode_length

        new_state = np.concatenate([customers, success, credit], dtype=np.float32)

        return new_state, new_memory, reward, {}

    def get_counterfactual_transitions(
        self,
        state: NDArray,
        actual_state: NDArray,
        action: int | NDArray,
        actual_memory: NDArray,
        schedule_step: int,
        n_counterfactuals: int,
    ) -> list[tuple[NDArray, NDArray, int, float, NDArray, NDArray]]:
        transitions = []
        for i in range(len(actual_memory)):
            for k in range(1, n_counterfactuals // 2 + 1):
                cf_memory = actual_memory.copy()
                cf_memory[i] += k
                if cf_memory[i] >= self.episode_length + 1:
                    break

                new_state, new_memory, reward, _ = self.get_transition(
                    actual_state, cf_memory, action, schedule_step
                )
                cf_memory = self.get_transformed_memory(cf_memory)
                new_memory = self.get_transformed_memory(new_memory)

                customers = new_state[: self.people]
                success = new_state[self.people : self.people + 1]
                credit = new_state[self.people + 1 :]

                if self.binarize_obs:
                    success = self.binarize(success, (self.episode_length + 1) * 2)
                    credit = self.binarize(credit, 7)

                new_state = np.concatenate(
                    [customers, success, credit], dtype=np.float32
                )

                transitions.append(
                    (state, cf_memory, action, reward, new_state, new_memory)
                )

        return transitions

    def step(self, action: int | NDArray) -> tuple[NDArray, float, bool, bool, dict]:
        self.current_step += 1
        state = self.state
        memory = self.memory

        new_state, new_memory, reward, info = self.get_transition(
            state, memory, action, self.current_step
        )
        done = self.current_step >= self.episode_length
        self.state = new_state
        self.memory = new_memory

        customers = new_state[: self.people]
        success = new_state[self.people : self.people + 1]
        credit = new_state[self.people + 1 :]

        if self.binarize_obs:
            success = self.binarize(success, (self.episode_length + 1) * 2)
            credit = self.binarize(credit, 7)

        new_state = np.concatenate([customers, success, credit], dtype=np.float32)
        new_memory = self.get_transformed_memory()

        obs = np.concatenate([new_state, new_memory], dtype=np.float32)
        info = {
            "state": new_state.copy(),
            "memory": new_memory.copy(),
        }
        return obs, reward, done, False, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[NDArray, dict]:
        self.memory = np.zeros(self.people // 2, dtype=np.float32)
        self.current_step = 0
        self.success = self.episode_length
        self.credit = self.default_credit.copy()

        customers = np.zeros(self.people, dtype=np.float32)
        for i in range(self.people):
            p = random.random()
            if p <= self.prob[i]:
                customers[i] = 1
            else:
                customers[i] = 0
        self.state = np.concatenate(
            [customers, np.array([self.success], dtype=np.float32), self.credit]
        )
        memory = self.get_transformed_memory()
        success = np.array([self.success], dtype=np.float32)
        credit = self.credit
        if self.binarize_obs:
            success = self.binarize(success, (self.episode_length + 1) * 2)
            credit = self.binarize(credit, 7)

        state = np.concatenate([customers, success, credit], dtype=np.float32)
        obs = np.concatenate([state, memory], dtype=np.float32)
        info = {
            "state": state.copy(),
            "memory": memory.copy(),
        }

        return obs, info
