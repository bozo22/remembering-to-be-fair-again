import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch


class ReplayMemory:
    def __init__(self, env, min_size, max_size, obs_length, device, storage_devie):
        self.device = device
        self.storage_device = storage_devie
        self.memory = torch.zeros((max_size, obs_length * 2 + 2)).to(
            self.storage_device
        )

        self.num_states = obs_length
        self.min_size = min_size
        self.max_size = max_size
        self.memory_counter = 0
        self.full = False
        self._initialize(env)

    # Run random policy for min_size steps to fill up the buffer
    def _initialize(self, env):
        _, info = env.reset()
        state = info["state"]
        memory = info["memory"]

        for _ in range(self.min_size):
            action = env.action_space.sample()
            _, reward, done, _, info = env.step(action)
            next_state = info["state"]
            next_memory = info["memory"]
            self.store_transition(
                state, memory, action, reward, next_state, next_memory
            )

            state = next_state
            memory = next_memory

            if done:
                _, info = env.reset()
                state = info["state"]
                memory = info["memory"]

    def store_transition(
        self,
        state: NDArray,
        memory: NDArray,
        action: int,
        reward: float,
        next_state: NDArray,
        next_memory: NDArray,
    ) -> None:
        """Store a transition in the replay buffer.

        Args:
            state (NDArray): The current state.
            memory (NDArray): The current memory.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (NDArray): The next state.
            next_memory (NDArray): The next memory.
        """

        transition = np.hstack(
            (state, memory, [action, reward], next_state, next_memory)
        )
        transition_t = torch.as_tensor(transition).to(
            self.storage_device, non_blocking=True
        )
        index = self.memory_counter % self.max_size
        self.memory[index, :] = transition_t
        self.memory_counter += 1
        self.full = self.memory_counter >= self.max_size

    # Sample batch_size random transitions from the buffer
    def sample(self, batch_size):
        size = self.max_size if self.full else self.memory_counter
        sample_index = np.random.choice(size, batch_size)
        batch_memory = self.memory[sample_index, :].to(self.device)
        batch_obs = batch_memory[:, : self.num_states]
        batch_action = batch_memory[:, self.num_states : self.num_states + 1].to(
            torch.long
        )
        batch_reward = batch_memory[:, self.num_states + 1 : self.num_states + 2]
        batch_next_obs = batch_memory[:, -self.num_states :]

        return batch_obs, batch_action, batch_reward, batch_next_obs
