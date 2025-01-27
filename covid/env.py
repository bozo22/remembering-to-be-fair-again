import gym
from gym.spaces import Discrete, Box
import numpy as np
from itertools import product
from numpy.typing import ArrayLike, NDArray


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
        render_mode: str | None = None,
        state_mode: str = "full",
        k: int = 3,
        population: ArrayLike | None = None,
        beta: ArrayLike = 0.3,
        sigma: ArrayLike = 0.2,
        gamma: ArrayLike = 0.1,
        vaccine_schedule: NDArray | None = None,
        max_steps: int = 500,
        allocation_step: float = 0.1,  # granularity of vaccine allocation fractions
        init_states: NDArray | None = None,
        normalize_reward: bool = False,
        normalize_obs: bool = False,
        novax: bool = False,
        continuous_actions: bool = False,
    ) -> None:
        super(CovidSEIREnv, self).__init__()

        # Set state mode
        self.state_mode = state_mode

        # Set render mode
        self.render_mode = render_mode

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
        def _param_to_array(param: ArrayLike | float) -> NDArray:
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
                padding = np.zeros(
                    max_steps - len(self.vaccine_schedule), dtype=np.float32
                )
                self.vaccine_schedule = np.concatenate([self.vaccine_schedule, padding])
        self.max_steps = max_steps

        # Define discrete allocation fractions
        self.allocation_fractions = np.arange(0.0, 1.01, allocation_step)

        # Define all possible allocations that sum to 1
        all_possible_allocations = [
            allocation
            for allocation in product(self.allocation_fractions, repeat=self.k)
            if np.isclose(sum(allocation), 1.0, atol=1e-6)
        ]

        self.allocation_mapping = np.array(all_possible_allocations, dtype=np.float32)

        # Action space: Discrete index into the allocation mapping
        self.continuous_actions = continuous_actions
        if continuous_actions:
            self.action_space = Box(
                low=0.0, high=1.0, shape=(self.k,), dtype=np.float32
            )
        else:
            self.action_space = Discrete(len(self.allocation_mapping))

        # Observation space:
        # No memory: 4 compartments per region + vaccines to allocate -> shape (4*k + 1,)
        # With memory: 4 compartments per region + vaccines to allocate + memory -> shape (4*k + 1 + k,)
        memory_size = k if state_mode != "none" else 0
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(4 * k + 1 + memory_size,), dtype=np.float32
        )

        self.normalize_reward = normalize_reward
        self.normalize_obs = normalize_obs
        self.novax = novax
        self.current_step = 0

        # Make it so it stores the previous states
        self.memory = np.zeros((self.k,), dtype=np.float32)
        self.init_states = (
            np.repeat(np.array([0.8, 0.1, 0.1, 0.0])[np.newaxis, :], self.k, axis=0)
            if init_states is None
            else init_states
        )

        self.reset()

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
        if self.state_mode == "min":
            memory -= np.min(memory)
            assert memory is not None

        # Reset: reset to 0 when elements are equal
        elif self.state_mode == "reset":
            if self_memory and np.all(memory == 0):
                self.memory = memory

        # None: no memory
        elif self.state_mode == "none":
            memory = np.array([])

        if self.normalize_obs and memory.sum() != 0:
            # memory /= memory.sum()
            memory /= np.sum(self.population)
            assert memory is not None

        return memory

    def normalize_state(self, state: NDArray) -> NDArray:
        """Normalize the state.

        Args:
            state (NDArray): The state to normalize.

        Returns:
            NDArray: The normalized state.
        """

        return state / np.sum(self.population)

        # Scale each region's compartments by the population
        region_state = state[:-1].reshape((self.k, 4))
        region_state_normalized = (
            region_state / self.population[:, np.newaxis]
        ).flatten()

        # Scale number of vaccines by population
        vaccines_normalized = state[-1] / np.sum(self.population)

        return np.concatenate([region_state_normalized, [vaccines_normalized]])

    def get_reward(
        self, state: NDArray, memory: NDArray, new_infected: NDArray
    ) -> float:
        """Get the reward.

        Args:
            state (NDArray): The state.
            memory (NDArray): The memory.
            new_infected (NDArray): # of newly infected people since last step.
        """

        reward = -np.sum(new_infected)
        if self.normalize_reward:
            reward /= np.sum(self.population)
        return reward

    def get_transition(
        self,
        state: NDArray,
        memory: NDArray,
        action: int | NDArray,
        schedule_step: int,
    ) -> tuple[NDArray, NDArray, float, dict]:
        """
        Calculate a transition in the environment:
          1. Allocate vaccines among k regions.
          2. Move vaccinated individuals from S -> R (assuming perfect efficacy).
          3. Apply SEIR updates in each region.
          4. Compute reward, done, info.

        Args:
            state (NDArray): Current state of the environment.
            memory (NDArray): Current memory of the environment.
            action (int | NDArray): Index of the action in the allocation mapping (if discrete) or the continuous allocation vector.
            schedule_step (int): Current timestep in the vaccine schedule.

        Returns:
            tuple[NDArray, NDArray, float, dict]: New state, new memory, reward, info dictionary.
        """

        # -- 1. Distribute vaccines --

        # Map action index to allocation vector
        allocation: NDArray
        if self.continuous_actions:
            assert isinstance(
                action, np.ndarray
            ), f"Expected np.ndarray, got {type(action)}"
            if np.sum(action) == 0:
                allocation = np.array([1.0 / self.k] * self.k)
            else:
                allocation = action / np.sum(action)
            # e_x = np.exp(action - np.max(action))
            # allocation = e_x / e_x.sum()
        else:
            allocation = self.allocation_mapping[action]

        assert np.isclose(
            np.sum(allocation), 1.0, atol=1e-6
        ), f"Allocation does not sum to 1: {allocation}"

        # Proceed with vaccine allocation and SEIR updates
        total_vaccines = self.vaccine_schedule[schedule_step]
        allocation = (
            allocation * total_vaccines
        ).round()  # Scale allocation by available vaccines

        # Current state is shape (4*k,)
        # We'll reshape it into (k,4) for clarity in calculations
        # region_state[i] = [S, E, I, R] for region i
        region_state = state[:-1].reshape((self.k, 4))

        # allocation = best_alloc * total_vaccines

        # -- 2. Vaccinate (move from S -> R) --
        if not self.novax:
            for i in range(self.k):
                max_possible_vaccines = region_state[i, 0]  # S compartment
                used_vaccines = min(allocation[i], max_possible_vaccines)

                # Move from S -> R
                region_state[i, 0] -= used_vaccines  # S
                region_state[i, 3] += used_vaccines  # R

                # Clip to ensure no numeric drift
                region_state[i, 0] = np.clip(
                    region_state[i, 0], 0.0, self.population[i]
                )
                region_state[i, 3] = np.clip(
                    region_state[i, 3], 0.0, self.population[i]
                )

        # -- 3. SEIR updates for each region --
        # Basic compartmental update (Euler discrete approximation)
        S_new = np.zeros(self.k, dtype=np.float32)
        E_new = np.zeros(self.k, dtype=np.float32)
        I_new = np.zeros(self.k, dtype=np.float32)
        R_new = np.zeros(self.k, dtype=np.float32)
        infected = region_state[:, 2].copy()

        for i in range(self.k):
            S_i, E_i, I_i, R_i = region_state[i]

            beta_i = self.beta[i]
            sigma_i = self.sigma[i]
            gamma_i = self.gamma[i]
            pop_i = self.population[i]

            dS = -beta_i * S_i * I_i / pop_i
            dE = beta_i * S_i * I_i / pop_i - sigma_i * E_i
            dI = sigma_i * E_i - gamma_i * I_i
            dR = gamma_i * I_i

            # Update
            S_new[i] = S_i + dS
            E_new[i] = E_i + dE
            I_new[i] = I_i + dI
            R_new[i] = R_i + dR

        # Ensure fractions stay in [0, population]
        S_new = np.clip(S_new, 0.0, self.population)
        E_new = np.clip(E_new, 0.0, self.population)
        I_new = np.clip(I_new, 0.0, self.population)
        R_new = np.clip(R_new, 0.0, self.population)
        infected = I_new - infected

        # Recombine
        region_state = np.stack([S_new, E_new, I_new, R_new], axis=1).flatten()
        vaccines = (
            0
            if schedule_step >= self.max_steps - 1
            else self.vaccine_schedule[schedule_step + 1]
        )
        new_state = np.concatenate([region_state, [vaccines]])
        new_memory = memory + allocation

        # -- 4. Reward: e.g., negative sum of suscepted fractions across all k regions --
        reward = self.get_reward(new_state, memory, infected)

        # Info dictionary for debugging
        info = {
            "new_infected": np.sum(infected),
            "vaccines_allocated": allocation,
        }

        return new_state, new_memory, reward, info

    def get_counterfactual_transitions(
        self,
        state: NDArray,
        action: int | NDArray,
        memory: NDArray,
        schedule_step: int,
        n_counterfactuals: int,
    ) -> list[tuple[NDArray, NDArray, int, float, NDArray, NDArray]]:
        """Generate counterfactual transitions.

        Args:
            state (NDArray): The state.
            action (int): The action.
            memory (NDArray): The memory.
            schedule_step (int): The vaccine production schedule step.
            n_counterfactuals (int): The number of counterfactuals to generate.

        Returns:
            list[tuple[NDArray, NDArray, int, float, NDArray, NDArray]]: The generated counterfactual transitions, a list of (state, memory, action, reward, new_state, new_memory) tuples.
        """
        ...

    def step(self, action: int | NDArray) -> tuple[NDArray, float, bool, bool, dict]:
        """
        Take one step in the environment.

        Args:
            action (int | NDArray): Index of the action in the allocation mapping (if discrete) or the continuous allocation vector.

        Returns:
            tuple[NDArray, float, bool, bool, dict]: Observation, reward, terminated, truncated, info dictionary.
        """

        # Get transition
        schedule_step = self.current_step
        new_state, new_memory, reward, info = self.get_transition(
            self.state, self.memory, action, schedule_step
        )

        # Update state and memory
        self.state = new_state.copy()
        self.memory = new_memory

        # Episode termination condition
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Transform memory
        memory = self.get_transformed_memory()
        state = new_state
        if self.normalize_obs:
            state = self.normalize_state(state)

        # Save separate state and memory
        info["state"] = state.copy()
        info["memory"] = memory

        # Concat state and memory for observation
        obs = np.concatenate((state, memory))

        return obs, float(reward), done, False, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[NDArray, dict]:
        """
        Resets the environment to an initial state.
        For example, each region starts with:
          80% susceptible, 10% exposed, 10% infected, 0% recovered.

        Args:
            seed (int | None, optional): Random seed. Defaults to None.
            options (dict | None, optional): Additional options. Defaults to None.

        Returns:
            tuple[NDArray, dict]: Observation, info dictionary
        """

        init_states = self.init_states
        self.current_step = 0

        # Build initial (k,4) array
        region_init = []
        for i in range(self.k):
            region_init.append(init_states[i] * self.population[i])

        region_init = np.array(region_init, dtype=np.float32).flatten()
        self.state = np.concatenate(
            [region_init, [self.vaccine_schedule[self.current_step]]]
        )
        self.memory = np.zeros((self.k,), dtype=np.float32)
        memory = self.get_transformed_memory()
        state = self.state.copy()
        if self.normalize_obs:
            state = self.normalize_state(state)

        # Save separate state and memory
        info = {"state": state, "memory": memory}

        # Concatenate state and memory for observation
        obs = np.concatenate((state, memory))

        return obs, info

    def render(self, mode: str = "human") -> None:
        """
        Render each region's SEIR fractions.

        Args:
            mode (str, optional): Rendering mode. Defaults to "human".
        """

        if self.render_mode is not None:
            region_state = self.state[:-1].reshape((self.k, 4))
            print(f"Step {self.current_step}")
            for i in range(self.k):
                S_i, E_i, I_i, R_i = region_state[i]
                print(
                    f"  Region {i}: S={S_i:,.0f}, E={E_i:,.0f}, I={I_i:,.0f}, R={R_i:,.0f}, Vaccines allocated so far={self.memory[i]}"
                )
            print()

    def close(self) -> None:
        """Close the environment."""
        pass
