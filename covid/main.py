from collections import deque
import argparse
import numpy as np
from tqdm import tqdm
import torch
import numpy as np
import csv
import random
from argparse import Namespace
from gym.spaces import Discrete, Box
from env import CovidSEIREnv
from agents import Agent, DQN, SAC, Random
import pickle
import warnings

warnings.filterwarnings("ignore")  # Suppress stable_baselines3 gym wrapper warnings


def set_seed(seed: int, env: CovidSEIREnv) -> None:
    """Set the random seed for reproducibility.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        env.seed(seed)
    except AttributeError:
        pass


def run(
    k: int,
    max_ep_len: int,
    memory_capacity: int,
    device: torch.device | str,
    args: Namespace,
    seed=42,
) -> tuple[list, list, list]:
    """Run the training loop.

    Args:
        k (int): Number of regions.
        max_ep_len (int): Maximum episode length.
        memory_capacity (int): Memory capacity.
        device (torch.device | str): Device to run on.
        args (Namespace): Arguments.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        tuple[list, list]: List of rewards, list of infected counts and list of memories.
    """
    # Suppose at each step we produce 50,000 vaccines for 10 steps
    # vaccine_schedule = [1_000] * max_steps
    # approximate vaccine production schedule from https://www.ifpma.org/news/as-covid-19-vaccine-output-estimated-to-reach-over-12-billion-by-year-end-and-24-billion-by-mid-2022-innovative-vaccine-manufacturers-renew-commitment-to-support-g20-efforts-to-address-remaining-barr/
    vaccine_schedule = (np.arange(1, max_ep_len + 1) ** 2 * 0.08) * 3_000_000
    # vaccine_schedule = [1_000_000_000 / max_ep_len] * max_ep_len
    infected_records = np.ones((args.episodes, max_ep_len, k))
    init_state_0 = [0.8, 0.2, 0.0, 0.0]
    init_state_1 = [0.9, 0.1, 0.0, 0.0]
    init_state_2 = [0.99, 0.01, 0.0, 0.0]
    init_states = np.array([init_state_0, init_state_1, init_state_2])

    # Values from https://arxiv.org/pdf/2005.12777
    beta = [0.33, 0.22, 0.18]
    gamma = [0.262, 0.085, 0.087]
    sigma = 0.2

    env = CovidSEIREnv(
        render_mode="human",
        state_mode=args.state_mode,
        k=k,
        population=[700_000_000, 200_000_000, 100_000_000],
        vaccine_schedule=vaccine_schedule,
        max_steps=max_ep_len,
        beta=beta,
        gamma=gamma,
        sigma=sigma,
        init_states=init_states,
        normalize_reward=True,
        normalize_obs=True,
        novax=args.novax,
        continuous_actions=args.agent_type in ["sac", "random_cont"],
    )

    set_seed(seed, env)

    assert type(env.observation_space) == Box
    num_states = env.observation_space.shape[0]

    num_actions: int | None = None
    if type(env.action_space) == Discrete:
        num_actions = env.action_space.n
    if type(env.action_space) == Box:
        num_actions = env.action_space.shape[0]
    assert num_actions is not None

    agent: Agent | None = None
    if args.agent_type == "dqn":
        agent = DQN(
            env,
            num_states,
            num_actions,
            memory_capacity,
            args.lr,
            device,
            args,
            # normalize=True,
        )
    elif args.agent_type == "sac":
        agent = SAC(
            env,
            args,
            memory_capacity,
            args.lr,
            device,
        )
    elif args.agent_type in ["random", "random_cont"]:
        agent = Random(env)

    assert agent is not None

    episodes = args.episodes
    reward_list = []
    infected_list = []
    memory_list = []
    all_rewards = np.zeros((episodes, max_ep_len))
    reward_buffer = deque(maxlen=100)
    loss_buffer = deque(maxlen=100)

    for i in (t := tqdm(range(episodes))):
        obs, info = env.reset()
        state = info["state"].copy()
        memory = info["memory"].copy()
        step = 0

        while True:
            # Store info for CF update
            schedule_step = env.current_step
            actual_state = env.state.copy()
            actual_memory = env.memory.copy()

            # Take step
            action = agent.choose_action(obs)
            next_obs, reward, done, _, info = env.step(action)
            next_state = info["state"].copy()
            next_memory = info["memory"].copy()

            # Store actual experience
            agent.store_transition(
                state, memory, action, reward, next_state, next_memory
            )

            # Store counterfactual experiences
            if args.counterfactual:
                cf_transitions = env.get_counterfactual_transitions(
                    state,
                    actual_state,
                    action,
                    memory,
                    actual_memory,
                    schedule_step,
                    args.num_counterfactuals,
                )
                for transition in cf_transitions:
                    agent.store_transition(*transition)

            # Learn
            if step % 5 == 0:
                loss = agent.learn()
                loss_buffer.append(loss)
            if done:
                break

            # Transition to next state
            obs = next_obs
            state = next_state
            memory = next_memory
            step += 1

        # Update epsilon
        if type(agent) == DQN and agent.epsilon > 0.01:
            agent.epsilon = agent.epsilon * 0.999

        # Evaluate
        obs, info = env.reset()
        state = info["state"]
        memory = info["memory"]
        ep_reward = 0
        curr_step = 0
        ep_infected = 0
        ep_memory = np.zeros_like(memory)

        while True:
            # Take step
            action = agent.choose_action(obs, True)
            next_obs, reward, done, _, info = env.step(action)
            next_state = info["state"]
            next_memory = info["memory"]

            # Update rewards
            all_rewards[i][curr_step] = reward

            # Transition to next state
            curr_step += 1
            ep_reward += reward
            ep_infected += info["new_infected"]
            obs = next_obs
            state = next_state
            memory = next_memory
            if done:
                utility_vaccines = (
                    (memory / memory.sum()) - (env.population / env.population.sum())
                    if memory.sum() > 0
                    else np.zeros_like(memory)
                )
                ep_memory = utility_vaccines
                break

        reward_list.append(ep_reward)
        infected_list.append(ep_infected)
        memory_list.append(ep_memory)
        reward_buffer.append(ep_reward)

        # Set tqdm description
        description = f"[EP {i+1}/{episodes}] Reward: {np.mean(reward_buffer):,.4f}"
        if type(agent) == DQN:
            description += f" | Loss: {np.mean(loss_buffer):.4f}"
        if type(agent) == DQN:
            description += f" | Epsilon: {agent.epsilon:.2f}"
        if type(agent) == DQN:
            description += f" | LR: {agent.optimizer.param_groups[0]['lr']:.7f}"
        if type(agent) == SAC:
            description += f" | Buffer: {agent.model.replay_buffer.size():,}"
        t.set_description(description)
        t.refresh()

    env.close()

    return reward_list, infected_list, memory_list


def save_data(
    num_exps: int,
    reward_list_all: list,
    infected_list_all: list,
    memory_list_all: list,
    args: Namespace,
) -> None:
    """Save the training curves to a CSV file.

    Args:
        num_exps (int): Number of experiments.
        reward_list_all (list): List of rewards for each experiment.
        infected_list_all (list): List of infected counts for each experiment.
        memory_list_all (list): List of final step memories for each experiment.
        args (Namespace): Arguments.
    """

    name = args.state_mode.capitalize()
    if args.counterfactual:
        name = f"FairSCM ({name})"
    if args.agent_type == "random":
        name = "Random"
    if args.agent_type == "random_cont":
        name = "Random"
    if args.novax:
        name = f"NoVax"
    rewards_dataset_path = f"./datasets/covid/{name}_reward.csv"
    infected_dataset_path = f"./datasets/covid/{name}_infected.csv"
    memory_dataset_path = f"./datasets/covid/{name}_memory.pkl"

    with open(rewards_dataset_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        for i in range(num_exps):
            csv_writer.writerow(reward_list_all[i])

    with open(infected_dataset_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        for i in range(num_exps):
            csv_writer.writerow(infected_list_all[i])

    if name == "Full":
        memory_list_all_a = np.array(memory_list_all)
        pickle.dump(memory_list_all_a, open(memory_dataset_path, "wb"))


if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Fair Covid""",
    )
    prs.add_argument(
        "-ep",
        dest="episodes",
        type=int,
        default=1000,
        required=False,
        help="episodes.\n",
    )
    prs.add_argument(
        "-lr",
        dest="lr",
        type=float,
        default=0.0001,
        required=False,
        help="learning rate.\n",
    )
    prs.add_argument(
        "-e",
        dest="epsilon",
        type=float,
        default=1.0,
        required=False,
        help="Exploration rate.\n",
    )
    prs.add_argument(
        "-g",
        dest="gamma",
        type=float,
        default=0.95,
        required=False,
        help="Discount factor\n",
    )
    prs.add_argument(
        "-sm",
        dest="state_mode",
        type=str,
        default="full",
        required=False,
        help="State representation mode\n",
    )
    prs.add_argument(
        "-cf",
        dest="counterfactual",
        type=bool,
        default=False,
        required=False,
        help="Counterfactual Update\n",
    )
    prs.add_argument(
        "-bs",
        dest="batch_size",
        type=int,
        default=64,
        required=False,
        help="Batch Size\n",
    )
    prs.add_argument(
        "-qiter",
        dest="q_network_iterations",
        type=int,
        default=1000,
        required=False,
        help="Q network iterations\n",
    )
    prs.add_argument(
        "-nexp",
        dest="num_exps",
        type=int,
        default=1,
        required=False,
        help="Number of Experiments\n",
    )
    prs.add_argument(
        "-nupds",
        dest="num_updates",
        type=int,
        default=2,
        required=False,
        help="Number of updates to look forward in to generate counterfactual experiences\n",
    )
    prs.add_argument(
        "-ncf",
        dest="num_counterfactuals",
        type=int,
        default=10,
        required=False,
        help="Number of counterfactual experiences to generate per step\n",
    )
    prs.add_argument(
        "-agent",
        dest="agent_type",
        type=str,
        default="dqn",
        required=False,
        help="Agent type\n",
    )
    prs.add_argument(
        "-novax",
        dest="novax",
        type=bool,
        default=False,
        required=False,
        help="Disable vaccines\n",
    )
    prs.add_argument(
        "-device",
        dest="device",
        type=str,
        default="cpu",
        required=False,
        help="Device (cpu or cuda)\n",
    )
    args = prs.parse_args()

    # reward_t, donut_t, rewards_to_plot, infected_records = run(
    #     k=3, max_ep_len=10, memory_capacity=400, args=args
    # )
    # plot_mean_and_std(np.expand_dims(rewards_to_plot, axis=-1), "rewards")
    # plot_mean_and_std(infected_records, "infected_records")

    seed = 2025
    num_exps = args.num_exps
    reward_list = []
    infected_list = []
    memory_list = []
    memory_capacity = 5_000 * (args.num_counterfactuals if args.counterfactual else 1)
    device = args.device
    for i in range(num_exps):
        print(f"Experiment {i+1}/{num_exps}")
        experiment_seed = seed + i + 1
        reward_t, infected_t, memory_t = run(
            k=3,
            max_ep_len=24,
            memory_capacity=memory_capacity,
            device=device,
            args=args,
            seed=experiment_seed,
        )
        reward_list.append(reward_t)
        infected_list.append(infected_t)
        memory_list.append(memory_t)
    save_data(num_exps, reward_list, infected_list, memory_list, args)

    # print(infected_records[-1][-1])
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
