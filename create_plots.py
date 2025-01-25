import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse

matplotlib.use("Agg")


def plot_data(env, ax, smooth, root, data_type="reward"):
    for filename in sorted(os.listdir(f"{root}/{env}")):
        if not filename.endswith(f"{data_type}.csv"):
            continue

        # Load data from file:
        data = np.genfromtxt(f"{root}/{env}/" + filename, delimiter=",")

        name = filename.split(".")[0].split("_")[0]

        # Smooth data
        if data.ndim == 1:
            data = data.reshape(1, -1)
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


def create_plots(env, smooth, root):

    if env == "lending":
        ax = plt.subplot(111)
        plot_data(env, ax, smooth, root)
        ax.legend()
        ax.set_xlabel("Number of Episodes")
        ax.set_ylabel("Accumulated Relaxed DP")

    elif env == "covid":
        _, axs = plt.subplots(1, 2, figsize=(8, 4), layout="tight")
        plot_data(env, axs[0], smooth, root, data_type="reward")
        plot_data(env, axs[1], smooth, root, data_type="infected")

        for ax in axs:
            ax.legend()
            ax.set_xlabel("Number of Episodes")

        axs[0].set_ylabel("Cumulative Reward")
        axs[1].set_ylabel("Number of Infected people")
        axs[1].ticklabel_format(useOffset=False, style="plain")
        axs[1].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

    else:
        _, axs = plt.subplots(1, 2, figsize=(8, 4), layout="tight")
        plot_data(env, axs[0], smooth, root, data_type="reward")
        plot_data(env, axs[1], smooth, root, data_type="donuts")

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
    prs.add_argument("--root", type=str, default="datasets")
    args = prs.parse_args()

    create_plots(args.env, args.smooth, args.root)
