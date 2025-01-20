import numpy as np
import os
import matplotlib
matplotlib.use("Agg")  # Use the 'Agg' backend when running without a GUI
import matplotlib.pyplot as plt
import argparse

def plot_data(env, rt, ax, donuts=False):
    directory = f"datasets/{env}/{rt}"
    for filename in os.listdir(directory):
        if donuts and "donuts" not in filename or not donuts and "donuts" in filename:
            continue

        data_path = os.path.join(directory, filename)
        try:
            data = np.genfromtxt(data_path, delimiter=",")
        except UnicodeDecodeError as e:
            print(f"Unicode decode error in file {filename}: {e}")
            continue  # Skip to the next file

        name = filename.split(".")[0]

        ax.plot(data.mean(axis=0), label=name)
        ax.fill_between(
            np.arange(data.shape[1]),
            data.mean(axis=0) - data.std(axis=0),
            data.mean(axis=0) + data.std(axis=0),
            alpha=0.2,
        )

def create_plots(env, rt):
    _, axs = plt.subplots(1, 2, figsize=(8, 4), layout="tight")
    plot_data(env, rt, axs[0], donuts=False)
    plot_data(env, rt, axs[1], donuts=True)

    for ax in axs:
        ax.legend()
        ax.set_xlabel("Number of Episodes")
        axs[0].set_ylabel(f"Accumulated {rt} Scores")
        axs[1].set_ylabel("Number of allocated donuts")

    output_path = os.path.join("donut", f"{rt}.png")
    plt.savefig(output_path)
    plt.close()  # Close the plot to free memory

if __name__ == "__main__":
    prs = argparse.ArgumentParser()
    prs.add_argument("--env", type=str, required=True)
    prs.add_argument("--rt", type=str, required=True)
    args = prs.parse_args()

    create_plots(args.env, args.rt)
