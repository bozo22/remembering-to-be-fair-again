import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import argparse

matplotlib.use("Agg")

def plot_data(env, ax, donuts=False):
    # Define the mapping of names to colors and legend names
    name_mapping = {
        "Fullprob": ("tab:blue", "Full"),
        "Fulldonuts": ("tab:blue", "Full"),
        "Full": ("tab:blue", "Full"),
        "FairQCMprob": ("tab:orange", "FairQCM"),
        "FairQCMdonuts": ("tab:orange", "FairQCM"),
        "FairQCM": ("tab:orange", "FairQCM"),
        "Minprob": ("tab:green", "Min"),
        "Mindonuts": ("tab:green", "Min"),
        "Min": ("tab:green", "Min"),
        "Resetprob": ("tab:red", "Reset"),
        "Resetdonuts": ("tab:red", "Reset"),
        "Reset": ("tab:red", "Reset"),
        "RNNprob": ("tab:purple", "RNN"),
        "RNNdonuts": ("tab:purple", "RNN"),
        "RNN": ("tab:purple", "RNN"),
    }

    for filename in os.listdir(f"datasets/{env}"):

        if (
            filename.endswith("donuts.csv")
            and not donuts
            or not filename.endswith("donuts.csv")
            and donuts
        ):
            continue

        data = np.genfromtxt(f"datasets/{env}/" + filename, delimiter=",")

        def average_over_episodes(data, window_size=10):
            num_points = data.shape[1]
            aggregated_data = []
            for i in range(0, num_points, window_size):
                aggregated_data.append(data[:, i:i+window_size].mean(axis=1))
            return np.array(aggregated_data).T

        data_avg = average_over_episodes(data, window_size=10)

        original_x = np.arange(data.shape[1])
        averaged_x = original_x[::10] + 5  

        name = filename.split(".")[0]
        if donuts:
            name = name.split("_")[0]

        color, label = name_mapping.get(name, ("gray", name))  # Default to gray if not in mapping
        ax.plot(averaged_x, data_avg.mean(axis=0), label=label, color=color) 
        print("data_avg.shape", data_avg.shape)
        ax.fill_between(
            averaged_x,
            data_avg.mean(axis=0) - data_avg.std(axis=0),
            data_avg.mean(axis=0) + data_avg.std(axis=0),
            color=color,
            alpha=0.2,
            linewidth=0
        )

    ax.set_xticks(np.linspace(0, 1000, 6))  # 0, 200, 400, ..., 1000
    ax.set_xticklabels([f"{int(tick)}" for tick in np.linspace(0, 1000, 6)])



def create_plots(env):

    if env == "lending":
        ax = plt.subplot(111)
        plot_data(env, ax)
        ax.legend()
        ax.set_xlabel("Number of Episodes")
        ax.set_ylabel("Accumulated Relaxed DP")
        plt.savefig("lending.png")

    else:
        _, axs = plt.subplots(1, 2, figsize=(8, 4), layout="tight")
        plot_data(env, axs[0], donuts=False)
        plot_data(env, axs[1], donuts=True)

        for ax in axs:
            axs[1].legend()
            ax.set_xlabel("Number of Episodes")

        axs[0].set_ylabel("Accumulated NSW Scores")
        axs[1].set_ylabel("Number of allocated donuts")

    plt.savefig(f"{env}.png")


if __name__ == "__main__":
    prs = argparse.ArgumentParser()
    prs.add_argument("--env", type=str, default="donut")
    args = prs.parse_args()

    create_plots(args.env)

