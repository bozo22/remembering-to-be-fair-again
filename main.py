import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from envs.donut import Donut
from envs.lending import Lending
import argparse
from datetime import datetime
import csv
from itertools import product
from dqn import DQN

matplotlib.use("Agg")

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")


def run(num_people, max_ep_len, memory_capacity, args, seed):
    if args.env_type == "donut":
        env = Donut(
            people=num_people,
            episode_length=max_ep_len,
            seed=seed,
            state_mode=args.state_mode,
            # p will need to be added as an argument for reproducibility
            p=[0.8, 0.8, 0.8, 0.8, 0.8],
            distribution=args.distribution,
            d_param1=args.d_param1,
            d_param2=args.d_param2,
            zero_memory=args.zero_memory,
            reward_type=args.reward_type
        )
    else:
        env = Lending(
            people=num_people,
            episode_length=max_ep_len,
            seed=seed,
            state_mode=args.state_mode,
            p=[0.9, 0.9, 0.9, 0.9, 0.9],
        )

    num_actions = env.action_space.n
    if args.net_type == "rnn":
        state = env.reset()
    else:
        state, memory = env.reset()
    num_states = len(state) + (0 if args.net_type == "rnn" else len(memory))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    dqn = DQN(
        num_states,
        num_actions,
        memory_capacity,
        args.lr,
        device,
        args.env_type,
        args.net_type,
        args,
    )

    episodes = args.episodes
    print("Collecting Experience....")
    reward_list = []
    donuts_list = []

    for i in range(episodes):
        if args.net_type == "rnn":
            state = env.reset()
        else:
            state, memory = env.reset()
        hidden = torch.zeros([1, 16])
        ep_reward = 0
        ep_donuts = 0
        while True:
            state_input = state.copy()
            if args.net_type == "linear":
                state_input.extend(memory)
            if args.net_type == "rnn":
                action, new_hidden = dqn.choose_action(state_input, hidden=hidden)
            else:
                action, new_hidden = dqn.choose_action(state_input)

            if args.env_type == "donut":
                actual_memory = env.memory.copy()
            else:
                actual_memory = env.loans.copy()
            if args.net_type == "linear":
                next_state, next_memory, reward, done, info = env.step(action)
            else:
                next_state, reward, done, info = env.step(action)
                next_memory = None

            if args.net_type == "rnn":
                dqn.store_transition(state, action, reward, next_state)
            else:
                dqn.store_transition(
                    state, action, reward, next_state, memory, next_memory
                )
            if args.counterfactual:
                dqn.counterfactual_update(
                    env,
                    state,
                    action,
                    reward,
                    next_state,
                    actual_memory,
                    max_ep_len,
                    args.state_mode,
                )
            ep_reward += reward
            if reward != 0:
                ep_donuts += 1

            if dqn.memory_counter >= memory_capacity:
                dqn.learn()
                if done and i % 1000 == 0:
                    print(
                        "episode: {} , the episode reward is {}".format(
                            i, round(ep_reward, 3)
                        )
                    )
            if done:
                break
            state = next_state
            hidden = new_hidden
            memory = next_memory

        if dqn.args.epsilon > 0.2:
            dqn.args.epsilon = dqn.args.epsilon * 0.999
        ep_reward = 0
        ep_donuts = 0

        if args.net_type == "linear":
            state, memory = env.reset()
        else:
            state = env.reset()
        state_input = state.copy()
        if args.net_type == "linear":
            state_input.extend(memory)
        ep_reward = 0
        hidden = torch.zeros([1, 16])

        while True:
            if args.net_type == "rnn":
                action, hidden = dqn.choose_action(
                    state_input, hidden=hidden, greedy=True
                )
            else:
                action, hidden = dqn.choose_action(state_input, greedy=True)

            if args.net_type == "linear":
                next_state, next_memory, reward, done, info = env.step(action)
            else:
                next_state, reward, done, info = env.step(action)
                next_memory = None

            ep_reward += reward
            if reward != 0:
                ep_donuts += 1
            if done:
                break
            state = next_state
            memory = next_memory
            hidden = new_hidden
            state_input = state.copy()
            if args.net_type == "linear":
                state_input.extend(memory)
        if i % 10 == 0:
            if args.env_type == "donut":
                print("done", ep_reward, env.donuts.copy())
            else:
                print("done", ep_reward, env.loans, env.success)
        reward_list.append(ep_reward)
        donuts_list.append(ep_donuts)
    return reward_list, donuts_list


def main():
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Fair Donut""",
    )
    prs.add_argument(
        "-ep",
        dest="episodes",
        type=int,
        default=50000,
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
        default="deep",
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
        "-env",
        dest="env_type",
        type=str,
        choices=["donut", "lending"],
        required=True,
        help="Environment Type\n",
    )
    prs.add_argument(
        "-net",
        dest="net_type",
        type=str,
        choices=["linear", "rnn"],
        required=True,
        help="Network Type\n",
    )
    prs.add_argument(
        "-dis",
        dest="distribution",
        type=str,
        default=None,
        required=False,
        choices=["logistic", "bell", "uniform-interval"],
        help="Distribution\n",
    )
    prs.add_argument(
        "-d1",
        dest="d_param1",
        type=str,
        default=None,
        required=False,
        help="Distribution parameter 1, comma-separated list of numbers\n",
    )
    prs.add_argument(
        "-d2",
        dest="d_param2",
        type=str,
        default=None,
        required=False,
        help="Distribution parameter 2, comma-separated list of numbers\n",
    )
    prs.add_argument(
        "-nomem",
        dest="zero_memory",
        type=bool,
        default=False,
        required=False,
        help="Force zero memory\n",
    )
    prs.add_argument(
        "-des",
        dest="description",
        type=str,
        default="",
        required=False,
        help="Output file description\n",
    )
    prs.add_argument(
        "-rt", "--reward_type",
        type=str,
        default="nsw",
        choices=["nsw", "utilitarian", "rawlsian", "egalitarian", "gini"],
        help="Select the reward function to use: 'nsw' for Nash Social Welfare, "
            "'utilitarian' for Utilitarian Welfare, 'rawlsian' for Rawlsian Welfare, "
            "'egalitarian' for Egalitarian Welfare, "
            "or 'gini' for Gini Coefficient based Social Welfare."
    )
    args = prs.parse_args()

    num_people = 5 if args.env_type == "donut" else 4
    seed = 2024
    max_ep_len = 100 if args.env_type == "donut" else 40
    memory_capacity = 5000 if args.env_type == "donut" else 1000

    if args.counterfactual:
        args.batch_size = args.batch_size * np.power(2, num_people)
        memory_capacity *= np.power(2, num_people - 1)

    if args.d_param1 and args.d_param2:
        args.d_param1 = [float(x) for x in args.d_param1.split(",")]
        args.d_param2 = [float(x) for x in args.d_param2.split(",")]

    num_exps = args.num_exps
    reward_list = []
    donut_list = []
    for i in range(num_exps):
        random.seed(seed)
        np.random.seed(seed + i + 1)
        reward_t, donut_t = run(num_people, max_ep_len, memory_capacity, args, seed + i) # what if we run lending?
        reward_list.append(reward_t)
        donut_list.append(donut_t) # what if we run lending?
    save_plot_avg(reward_list, donut_list, args, num_exps, num_people, max_ep_len)


def save_plot_avg(
    reward_list_all, donuts_list_all, args, num_exps, num_people, max_ep_len
):

    pathprefix = (
        "./datasets/" + args.net_type + "-" + args.env_type + "-dqn/" + args.state_mode
    )
    rewards_dataset_paths = (
        pathprefix
        + "-des"
        + str(args.description)
        + "-people"
        + str(num_people)
        + "-cf"
        + str(args.counterfactual)
        + "-rt" 
        + args.reward_type
        + "-"
        + current_time
        + ".csv"
    )

    with open(rewards_dataset_paths, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["episodes", "people", "maxeplen", "learning rate", "batch size"]
        )
        csv_writer.writerow(
            [
                str(args.episodes),
                str(num_people),
                str(max_ep_len),
                str(args.lr),
                str(args.batch_size),
            ]
        )
        for i in range(num_exps):
            csv_writer.writerow(reward_list_all[i])
            csv_writer.writerow([""])
            if args.env_type == "donut":
                csv_writer.writerow(donuts_list_all[i])
            else:
                csv_writer.writerow(" ")

    reward_list_all = np.array(reward_list_all)
    donuts_list_all = np.array(donuts_list_all)

    interv = 10
    reward_list = []
    donuts_list = []
    for k in range(len(reward_list_all)):
        reward_list_t = []
        donuts_list_t = []
        for j in range(0, len(reward_list_all[k]), interv):
            end = j + interv
            end = min(end, len(reward_list_all[k]))
            mn = np.mean(reward_list_all[k][j:end], axis=0)
            mn_d = np.mean(donuts_list_all[k][j:end], axis=0)
            reward_list_t.append(mn)
            donuts_list_t.append((mn_d))
        reward_list.append(reward_list_t)
        donuts_list.append(donuts_list_t)
    reward_list = np.array(reward_list)
    donuts_list = np.array(donuts_list)

    mean_rewards = np.mean(reward_list, axis=0)
    mean_donuts = np.mean(donuts_list, axis=0)

    std_rewards = np.std(reward_list, axis=0)
    std_donuts = np.std(donuts_list, axis=0)

    x = [i * 10 for i in range(len(mean_rewards))]

    if args.env_type == "donut":
        fig, ax = plt.subplots(1, 2)

        ax[0].plot(x, mean_rewards, label="Reward")
        ci = 1.96 * std_rewards / np.sqrt(num_exps)
        ax[0].fill_between(x, (mean_rewards - ci), (mean_rewards + ci), alpha=0.3)

        ax[1].plot(x, mean_donuts, label="Reward")
        ci = 1.96 * std_donuts / np.sqrt(num_exps)
        ax[1].fill_between(x, (mean_donuts - ci), (mean_donuts + ci), alpha=0.3)

        # ax[0].set_ylabel("Sum of NSW")
        if args.reward_type == 'nsw':
            ylabel_text = "Sum of Nash Social Welfare"
        elif args.reward_type == 'utilitarian':
            ylabel_text = "Sum of Utilitarian Welfare"
        elif args.reward_type == 'rawlsian':
            ylabel_text = "Sum of Rawlsian Welfare"
        elif args.reward_type == 'egalitarian':
            ylabel_text = "Sum of Egalitarian Welfare"
        elif args.reward_type == 'gini':
            ylabel_text = "Sum of Gini Coefficient Welfare"
        ax[0].set_ylabel(ylabel_text)
        ax[1].set_ylabel("Number of allocated donuts")
    else:
        fig, ax = plt.subplots(1, 1)

        ax.plot(x, mean_rewards, label="Reward")
        ci = 1.96 * std_rewards / np.sqrt(num_exps)
        ax.fill_between(x, (mean_rewards - ci), (mean_rewards + ci), alpha=0.3)

        ax.set_ylabel("Sum of NSW")

    title = args.state_mode
    if args.counterfactual:
        title += " with Counterfactuals"
    plt.suptitle(title, fontsize=16)
    plt.savefig(
        "./"
        + args.env_type
        + "/"
        + args.net_type
        + "-DQN"
        + args.state_mode
        + "-des"
        + str(args.description)
        + "-cf"
        + str(args.counterfactual)
        + "-rt" 
        + args.reward_type
        + "-"
        + current_time
        + ".png"
    )
    # plt.show()


if __name__ == "__main__":
    main()
