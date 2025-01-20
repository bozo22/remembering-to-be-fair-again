import numpy as np
import os
import matplotlib
matplotlib.use("Agg")  # Use the 'Agg' backend when running without a GUI
import matplotlib.pyplot as plt
import argparse

def plot_data(env, rt, ax, donuts=False):
    directory = f"datasets/{env}/{rt}"
    method_colors = {
        'Full': '#1F77B4',  # blue
        'FairQCM': '#FF7F0F',  # orange
        'Min': '#2E9F2B',  # green
        'Reset': '#D62628',  # red 
        'RNN': '#9467BD'  # purple
    }


    for filename in os.listdir(directory):
        if donuts and "donuts" not in filename or not donuts and "donuts" in filename:
            continue

        # Extract method name from filename
        if "donuts" in filename:
            method = filename.split('_donuts')[0]  # Strip out the suffix to get the method
        else:
            method = filename.split('.csv')[0]  # Strip out the extension to get the method

        color = method_colors.get(method, 'gray')  # Use a default color if method not recognized

        data_path = os.path.join(directory, filename)
        data = np.genfromtxt(data_path, delimiter=",")
        name = method   # Adjust label based on data type

        ax.plot(data.mean(axis=0), label=name, color=color)
        ax.fill_between(
            np.arange(data.shape[1]),
            data.mean(axis=0) - data.std(axis=0),
            data.mean(axis=0) + data.std(axis=0),
            color=color,
            alpha=0.2,
        )

def create_plots(env, rt):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), layout="tight")
    plot_data(env, rt, axs[0], donuts=False)
    plot_data(env, rt, axs[1], donuts=True)

    # axs[0].set_title(f"{rt} Scores")
    # axs[1].set_title(f"{rt} Donuts Allocated")

    for ax in axs:
        ax.legend()
        ax.set_xlabel("Number of Episodes")
        axs[0].set_ylabel(f"Accumulated {rt} Scores")
        axs[1].set_ylabel("Number of allocated donuts")

    output_path = os.path.join("donut", f"{rt}.png")
    if not os.path.exists("donut"):
        os.makedirs("donut")
    plt.savefig(output_path)
    plt.close()  # Ensure memory is freed by closing the plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--rt", type=str, required=True)
    args = parser.parse_args()

    create_plots(args.env, args.rt)
