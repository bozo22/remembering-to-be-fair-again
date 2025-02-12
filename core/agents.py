import torch
import torch.nn as nn
import numpy as np
from gym import Env
from numpy.typing import NDArray
from stable_baselines3 import SAC as SB3SAC
from argparse import Namespace
from abc import ABC, abstractmethod
from core.utils import ReplayMemory
from core.policies import MLPPolicy, RNNPolicy
from envs.covid import CovidSEIREnv


class Agent(ABC):

    @abstractmethod
    def choose_action(
        self, obs: NDArray, greedy: bool = False, hidden: torch.Tensor | None = None
    ) -> tuple[int | NDArray, torch.Tensor | None]:
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


class DQN(Agent):
    def __init__(
        self,
        env: Env,
        num_states: int,
        num_actions: int,
        memory_capacity: int,
        learning_rate: float,
        device: torch.device | str,
        args: Namespace,
        net_arch: list[int],
        normalize: bool = False,
    ):
        super(DQN, self).__init__()

        # Initialize the policy networks
        self.device = device
        self.eval_net: nn.Module
        self.target_net: nn.Module
        if args.net_type == "linear":
            self.eval_net, self.target_net = MLPPolicy(
                num_states, num_actions, net_arch
            ), MLPPolicy(num_states, num_actions, net_arch)
        elif args.net_type == "rnn":
            self.eval_net, self.target_net = RNNPolicy(
                num_states, num_actions, net_arch, args.hidden_size
            ), RNNPolicy(num_states, num_actions, net_arch, args.hidden_size)

        assert self.eval_net is not None
        assert self.target_net is not None

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
            memory_capacity,
            memory_capacity,
            num_states,
            device,
            device,
        )
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    def choose_action(
        self, obs: NDArray, greedy: bool = False, hidden: torch.Tensor | None = None
    ) -> tuple[int, torch.Tensor | None]:
        obs_t = torch.unsqueeze(torch.FloatTensor(obs), 0).to(self.device)
        if self.normalize:
            obs_t = nn.functional.normalize(obs_t, p=2, dim=1)

        if greedy or np.random.uniform() >= self.epsilon:  # greedy policy
            with torch.no_grad():
                action_value, hidden = self.eval_net.forward(obs_t, prev_hidden=hidden)
                action = torch.max(action_value.cpu(), 1)[1].data.numpy()
            action = action[0]

        else:  # random policy
            action = np.random.randint(self.num_actions)
        return action, None

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
        q_eval, _ = self.eval_net(batch_obs)
        q_eval = q_eval.gather(1, batch_action)

        with torch.no_grad():
            if self.normalize:
                batch_next_obs = nn.functional.normalize(batch_next_obs, p=2, dim=1)
            q_next, _ = self.target_net(batch_next_obs)
            q_next = q_next.detach()
            max_q_next = q_next.max(1)[0].view(self.batch_size, 1)

        q_target = batch_reward + self.gamma * max_q_next
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class SAC(Agent):

    def __init__(
        self,
        env: Env,
        args: Namespace,
        memory_capacity: int,
        learning_rate: float,
        device: torch.device | str,
        net_arch: list[int],
    ):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = args.batch_size
        policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=net_arch, n_critics=2)
        self.model = SB3SAC(
            "MlpPolicy",
            env,
            verbose=0,
            policy_kwargs=policy_kwargs,
            device=device,
            buffer_size=memory_capacity,
            learning_rate=learning_rate,
            gamma=args.gamma,
            tau=0.000001,
            ent_coef=0.000,
            # target_entropy=0.0,
            # use_sde=True,
            # action_noise=NormalActionNoise(0, 0.1),
            # replay_buffer_class=HerReplayBuffer,
        )
        self.model._setup_learn(env.max_steps * args.episodes)

    def choose_action(
        self, obs: NDArray, greedy: bool = False, hidden: torch.Tensor | None = None
    ) -> tuple[NDArray, torch.Tensor | None]:
        self.model.policy.set_training_mode(False)
        with torch.no_grad():
            return self.model.predict(obs, deterministic=greedy)[0], None

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


class Random(Agent):
    def __init__(self, env: Env):
        self.env = env

    def choose_action(
        self, obs: NDArray, greedy: bool = False, hidden: torch.Tensor | None = None
    ) -> tuple[int, torch.Tensor | None]:
        return self.env.action_space.sample(), None

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
