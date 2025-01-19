import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import argparse

matplotlib.use("Agg")


def plot_data(env, ax, donuts=False):
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
        ax.plot(data.mean(axis=0), label=name)
        ax.fill_between(
            np.arange(data.shape[1]),
            data.mean(axis=0) - data.std(axis=0),
            data.mean(axis=0) + data.std(axis=0),
            alpha=0.2,
        )


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
            ax.legend()
            ax.set_xlabel("Number of Episodes")

        axs[0].set_ylabel("Accumulated NSW Scores")
        axs[1].set_ylabel("Number of allocated donuts")

    plt.savefig(f"{env}.png")


if __name__ == "__main__":
    prs = argparse.ArgumentParser()
    prs.add_argument("--env", type=str, default="donut")
    args = prs.parse_args()

    create_plots(args.env)
