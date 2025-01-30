import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import pickle

matplotlib.use("Agg")


def plot_data(env, ax, smooth, root, data_type="reward"):
    avg_stds = []
    for filename in sorted(os.listdir(f"{root}/{env}")):
        # Load data from file:
        data = np.genfromtxt(f"{root}/{env}/" + filename, delimiter=",")
        data = data[:,:500]
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


        avg_std = data.std(axis=0).mean()
        avg_stds.append((name, avg_std))

        # Plot
        xx = np.arange(0, vals, smooth)
        ax.plot(xx, data.mean(axis=0), label=name, linewidth=3, alpha=0.9)
        # ax.fill_between(
        #     xx,
        #     data.mean(axis=0) - data.std(axis=0),
        #     data.mean(axis=0) + data.std(axis=0),
        #     alpha=0.1,
        # )
    print("Average Standard Deviations per Baseline:")
    for name, avg_std in avg_stds:
        print(f"{name}: {avg_std:.4f}")
    


def create_plots(env, smooth, root):

    if env == "lending":
        ax = plt.subplot(111)
        plot_data(env, ax, smooth, root)
        ax.legend()
        ax.set_xlabel("Number of Episodes")
        ax.set_ylabel("Accumulated Relaxed DP")

    elif env == "covid":
        _, axs = plt.subplots(1, 3, figsize=(12, 4), layout="tight")
        plot_data(env, axs[0], smooth, root, data_type="reward")
        plot_data(env, axs[1], smooth, root, data_type="infected")
        plot_data(env, axs[2], smooth, root, data_type="memory")

        for ax in axs:
            ax.legend()
            ax.set_xlabel("Number of episodes")

        axs[0].set_ylabel("Cumulative reward")
        axs[1].set_ylabel("Number of infected people")
        axs[2].set_ylabel("% of vaccines allocated")
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
