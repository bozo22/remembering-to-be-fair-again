import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import pickle

matplotlib.use("Agg")


def plot_data(env, ax, smooth, root, data_type="reward"):
    for filename in sorted(os.listdir(f"{root}/{env}")):
        # Load data from file:
        if filename.endswith(f"{data_type}.csv"):
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
        elif filename.endswith(f"{data_type}.pkl"):
            data = pickle.load(open(f"{root}/{env}/" + filename, "rb"))
            name = filename.split(".")[0].split("_")[0]

            # Smooth data
            if data.ndim == 2:
                data = data.reshape(1, data.shape[0], data.shape[1])
            exps, eps, regions = data.shape
            data = data.transpose(2, 0, 1)

            # Plot
            for i in range(regions):

                data_i = (
                    data[i]
                    .reshape(exps, -1, smooth)
                    .mean(axis=2, keepdims=True)
                    # .repeat(smooth, axis=2)
                    .reshape(exps, -1)
                )

                xx = np.arange(0, eps, smooth)
                ax.plot(xx, data_i.mean(axis=0), label="Region " + str(i))
                ax.fill_between(
                    xx,
                    data_i.mean(axis=0) - data_i.std(axis=0),
                    data_i.mean(axis=0) + data_i.std(axis=0),
                    alpha=0.2,
                )
        else:
            continue


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
