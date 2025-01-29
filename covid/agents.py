import torch
import torch.nn as nn
import numpy as np
from numpy.typing import ArrayLike, NDArray
from stable_baselines3 import SAC as SB3SAC
from policies import MLPPolicy
from argparse import Namespace
from abc import ABC, abstractmethod
from env import CovidSEIREnv
from utils import ReplayMemory
import gym


class Agent(ABC):

    @abstractmethod
    def choose_action(self, obs: NDArray, greedy: bool = False) -> int | NDArray:
        """Choose an action given an observation.

        Args:
            obs (NDArray): The observation.
            greedy (bool, optional): Whether to use a greedy policy. Defaults to False.

        Returns:
            int | np.ndarray: The chosen action.
        """
        ...

    @abstractmethod
    def store_transition(
        self,
        state: NDArray,
        memory: NDArray,
        action: int | NDArray,
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
        ...

    @abstractmethod
    def learn(self) -> float | None:
        """Take a learning step.

        Returns:
            float | None: The loss (if applicable).
        """
        ...

    # @abstractmethod
    # def counterfactual_update(
    #     self,
    #     env: CovidSEIREnv,
    #     state: NDArray,
    #     action: int,
    #     memory: NDArray,
    #     next_state: NDArray,
    #     max_ep_len: int,
    #     schedule_step: int,
    #     magnitude: int = 25_000_000,
    #     n_counterfactuals: int = 10,
    #     distribution: str = "uniform",
    # ) -> None:
    #     """Perform a counterfactual update.

    #     Args:
    #         env (CovidSEIREnv): The environment.
    #         state (NDArray): The current state.
    #         action (int): The action taken.
    #         memory (NDArray): The current memory.
    #         next_state (NDArray): The next state.
    #         max_ep_len (int): The maximum episode length.
    #         schedule_step (int): The current vaccine schedule step.
    #         magnitude (int, optional): The magnitude of the counterfactual update. Defaults to 25_000_000.
    #         n_counterfactuals (int, optional): The number of counterfactuals to generate. Defaults to 10.
    #         distribution (str, optional): The distribution to sample from. Defaults to "uniform".
    #     """
    #     ...


class DQN(Agent):
    def __init__(
        self,
        env: CovidSEIREnv,
        num_states: int,
        num_actions: int,
        memory_capacity: int,
        learning_rate: float,
        device: torch.device | str,
        args: Namespace,
        normalize: bool = False,
    ):
        super(DQN, self).__init__()

        # Initialize the policy networks
        self.device = device
        self.eval_net, self.target_net = MLPPolicy(num_states, num_actions), MLPPolicy(
            num_states, num_actions
        )
        self.eval_net.to(self.device)
        self.target_net.to(self.device)

        # Initialize the weights of the policy networks
        def init_weights(m: nn.Module) -> None:
            if type(m) == nn.Linear:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                m.bias.data.fill_(0.0)

        self.eval_net.apply(init_weights)

        self.target_net.load_state_dict(self.eval_net.state_dict())

        # Save hyperparameters
        self.num_states = num_states
        self.num_actions = num_actions
        self.memory_capacity = memory_capacity
        self.epsilon = args.epsilon
        self.q_network_iterations = args.q_network_iterations
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.normalize = normalize

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.replay_memory = ReplayMemory(
            env,
            memory_capacity // 4,
            memory_capacity,
            num_states,
            device,
            device,
        )
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=4_000, gamma=0.5
        )
        self.loss_func = nn.SmoothL1Loss()

    def choose_action(self, obs: NDArray, greedy: bool = False) -> int:
        obs_t = torch.unsqueeze(torch.FloatTensor(obs), 0).to(self.device)
        if self.normalize:
            obs_t = nn.functional.normalize(obs_t, p=2, dim=1)

        if greedy or np.random.uniform() >= self.epsilon:  # greedy policy
            with torch.no_grad():
                action_value = self.eval_net.forward(obs_t)
                action = torch.max(action_value.cpu(), 1)[1].data.numpy()
            action = action[0]

        else:  # random policy
            action = np.random.randint(self.num_actions)
        return action

    def store_transition(
        self,
        state: NDArray,
        memory: NDArray,
        action: int | NDArray,
        reward: float,
        next_state: NDArray,
        next_memory: NDArray,
    ) -> None:
        action = int(action)
        self.replay_memory.store_transition(
            state, memory, action, reward, next_state, next_memory
        )

    def learn(self) -> float:
        if self.learn_step_counter % self.q_network_iterations == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        batch_obs, batch_action, batch_reward, batch_next_obs = (
            self.replay_memory.sample(self.batch_size)
        )

        if self.normalize:
            batch_obs = nn.functional.normalize(batch_obs, p=2, dim=1)
        q_eval = self.eval_net(batch_obs).gather(1, batch_action)

        with torch.no_grad():
            if self.normalize:
                batch_next_obs = nn.functional.normalize(batch_next_obs, p=2, dim=1)
            q_next = self.target_net(batch_next_obs).detach()
            max_q_next = q_next.max(1)[0].view(self.batch_size, 1)

            # # This implements double DQN, which will stabilize training?
            # # Action selection using eval_net
            # max_action_selection = self.eval_net(batch_next_state).max(1)[1].view(-1, 1)
            # # Q-value evaluation using target_net
            # max_q_next = self.target_net(batch_next_state).gather(1, max_action_selection)

        q_target = batch_reward + self.gamma * max_q_next
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def counterfactual_update(
        self,
        env,
        state,
        action,
        memory,
        next_state,
        max_ep_len,
        schedule_step,
        magnitude=25_000_000,
        n_counterfactuals=10,
        distribution="uniform",
    ):
        # Don't do anything if we're at the end of the episode
        if schedule_step == max_ep_len - 1:
            return
        for _ in range(n_counterfactuals):
            # Generate counterfactual memory (clip to ensure non-negative)
            cf_memory = memory
            assert distribution in ["normal", "uniform"], "Invalid distribution type"
            if distribution == "normal":
                cf_memory = (
                    np.random.normal(memory, magnitude).round().astype(np.float32)
                )
            elif distribution == "uniform":
                cf_memory = (
                    np.random.uniform(memory - magnitude, memory + magnitude)
                    .round()
                    .astype(np.float32)
                )
            cf_memory = np.clip(cf_memory, a_min=0, a_max=None)

            # Get the transition using the counterfactual memory
            new_state, new_memory, reward, _ = env.get_transition(
                state, cf_memory, action, schedule_step
            )
            new_memory = env.get_transformed_memory(new_memory)

            # Store the counterfactual experience
            # NOTE: Should we use next_state (from the actual transition) or new_state (from the counterfactual transition)?
            # Paper says next_state, but new_state seems more logical to me.
            self.store_transition(
                state, cf_memory, action, reward, new_state, new_memory
            )


class SAC(Agent):

    def __init__(
        self,
        env: CovidSEIREnv,
        args: Namespace,
        memory_capacity: int,
        learning_rate: float,
        device: torch.device | str,
    ):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = args.batch_size
        # policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[32, 32])
        self.model = SB3SAC(
            "MlpPolicy",
            env,
            verbose=0,
            # policy_kwargs=policy_kwargs,
            device=device,
            buffer_size=memory_capacity,
            learning_rate=learning_rate,
            target_update_interval=args.q_network_iterations,
        )
        self.model._setup_learn(env.max_steps * args.episodes)

    def choose_action(self, obs: NDArray, greedy: bool = False) -> NDArray:
        return self.model.predict(obs, deterministic=greedy)[0]

    def learn(self) -> None:
        self.model.train(gradient_steps=1, batch_size=self.batch_size)

    def store_transition(
        self,
        state: NDArray,
        memory: NDArray,
        action: int | NDArray,
        reward: float,
        next_state: NDArray,
        next_memory: NDArray,
    ) -> None:
        assert self.model.replay_buffer is not None, "Replay buffer is not initialized"
        obs = np.concatenate([state, memory])
        next_obs = np.concatenate([next_state, next_memory])
        action_a = np.array([action])
        reward_a = np.array([reward])
        done_a = np.array([False])
        self.model.replay_buffer.add(obs, next_obs, action_a, reward_a, done_a, [{}])
        pass


class Random(Agent):
    def __init__(self, env: CovidSEIREnv):
        self.env = env

    def choose_action(self, obs: NDArray, greedy: bool = False) -> int:
        return self.env.action_space.sample()

    def store_transition(
        self,
        state: NDArray,
        memory: NDArray,
        action: int | NDArray,
        reward: float,
        next_state: NDArray,
        next_memory: NDArray,
    ) -> None:
        pass

    def learn(self) -> float:
        return 0
