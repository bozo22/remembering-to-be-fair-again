import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import argparse

matplotlib.use("Agg")


def plot_data(env, ax, smooth, donuts=False):
    for filename in os.listdir(f"datasets/{env}"):
        if (
            filename.endswith("donuts.csv")
            and not donuts
            or not filename.endswith("donuts.csv")
            and donuts
        ):
            continue

        # Load data from file:
        data = np.genfromtxt(f"datasets/{env}/" + filename, delimiter=",")

        name = filename.split(".")[0]
        if donuts:
            name = name.split("_")[0]

        # Smooth data
        exps, vals = data.shape
        data = (
            data.reshape(exps, -1, smooth).mean(axis=2, keepdims=True)
            # .repeat(smooth, axis=2)
            .reshape(exps, -1)
        )

        # Plot
        xx = np.arange(0, vals, smooth)
        ax.plot(xx, data.mean(axis=0), label=name)
        ax.fill_between(
            xx,
            data.mean(axis=0) - data.std(axis=0),
            data.mean(axis=0) + data.std(axis=0),
            alpha=0.2,
        )


def create_plots(env, smooth):

    if env == "lending":
        ax = plt.subplot(111)
        plot_data(env, ax, smooth)
        ax.legend()
        ax.set_xlabel("Number of Episodes")
        ax.set_ylabel("Accumulated Relaxed DP")
        plt.savefig("lending.png")

    elif env == "covid":
        ax = plt.subplot(111)
        plot_data(env, ax, smooth)
        ax.legend()
        ax.set_xlabel("Number of Episodes")
        ax.set_ylabel("Cumulative Reward")
        plt.savefig("covid.png")

    else:
        _, axs = plt.subplots(1, 2, figsize=(8, 4), layout="tight")
        plot_data(env, axs[0], smooth, donuts=False)
        plot_data(env, axs[1], smooth, donuts=True)

        for ax in axs:
            ax.legend()
            ax.set_xlabel("Number of Episodes")

        axs[0].set_ylabel("Accumulated NSW Scores")
        axs[1].set_ylabel("Number of allocated donuts")

    plt.savefig(f"{env}.png")


if __name__ == "__main__":
    prs = argparse.ArgumentParser()
    prs.add_argument("--env", type=str, default="donut")
    prs.add_argument("--smooth", type=int, default=5)
    args = prs.parse_args()

    create_plots(args.env, args.smooth)
