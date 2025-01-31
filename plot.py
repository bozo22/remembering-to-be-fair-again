import numpy as np
import os
import matplotlib
matplotlib.use("Agg")  # Use the 'Agg' backend when running without a GUI
import matplotlib.pyplot as plt
import argparse

def plot_data(env, rt, ax, smooth, donuts=False):
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

        # Smooth data
        exps, vals = data.shape
        data = (
            data.reshape(exps, -1, smooth).mean(axis=2, keepdims=True)
            # .repeat(smooth, axis=2)
            .reshape(exps, -1)
        )

        xx = np.arange(0, vals, smooth)
        ax.plot(xx, data.mean(axis=0), label=name, color=color)
        ax.fill_between(
            # np.arange(data.shape[1]),
            xx,
            data.mean(axis=0) - data.std(axis=0),
            data.mean(axis=0) + data.std(axis=0),
            color=color,
            alpha=0.2,
        )

def create_plots(env, rt, smooth):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), layout="tight")
    plot_data(env, rt, axs[0], smooth, donuts=False)
    plot_data(env, rt, axs[1], smooth, donuts=True)

    # axs[0].set_title(f"{rt} Scores")
    # axs[1].set_title(f"{rt} Donuts Allocated")

    for ax in axs:
        # ax.legend()
        ax.set_xlabel("Number of Episodes")
        axs[0].set_ylabel(f"Accumulated {rt} Scores")
        axs[1].set_ylabel("Number of allocated donuts")
    # 在图形外部添加统一的图例
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)

    output_path = os.path.join("donut", f"{rt}.png")
    if not os.path.exists("donut"):
        os.makedirs("donut")
    plt.savefig(output_path)
    plt.close()  # Ensure memory is freed by closing the plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--rt", type=str, required=True)
    parser.add_argument("--smooth", type=int, default=5)
    args = parser.parse_args()

    create_plots(args.env, args.rt, args.smooth)
