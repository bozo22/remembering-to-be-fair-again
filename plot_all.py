import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
plt.rcParams['text.usetex'] = True  # 启用TeX渲染

def plot_data_no_std(env, rt, ax, smooth, donuts=False):
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
        ax.plot(xx, data.mean(axis=0), label=name, color=color, linewidth=4)
        # ax.fill_between(
        #     # np.arange(data.shape[1]),
        #     xx,
        #     data.mean(axis=0) - data.std(axis=0),
        #     data.mean(axis=0) + data.std(axis=0),
        #     color=color,
        #     alpha=0.2,
        # )

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
        ax.plot(xx, data.mean(axis=0), label=name, color=color, linewidth=4)
        ax.fill_between(
            # np.arange(data.shape[1]),
            xx,
            data.mean(axis=0) - data.std(axis=0),
            data.mean(axis=0) + data.std(axis=0),
            color=color,
            alpha=0.2,
        )

def create_combined_plot(env, smooth):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # Adjust subplot layout size as needed
    reward_types = ['egalitarian', 'gini']
    for i, rt in enumerate(reward_types):
        plot_data_no_std(env, rt, axs[i, 0], smooth, donuts=False)
        axs[i, 0].set_ylabel(f"Accumulated {rt} Scores", fontsize=18)
        axs[i, 0].set_xlabel("Number of Episodes", fontsize=18)
        if i == 0:
            plot_data_no_std(env, rt, axs[i, 1], smooth, donuts=True)
        else:
            plot_data(env, rt, axs[i, 1], smooth, donuts=True)
        axs[i, 1].set_ylabel(f"Number of allocated donuts for {rt}", fontsize=18)
        axs[i, 1].set_xlabel("Number of Episodes", fontsize=18)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', ncol=5)
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize='18')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("all.png")
    plt.close()  # Close the plot to free up memory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--smooth", type=int, default=5)
    args = parser.parse_args()

    create_combined_plot(args.env, args.smooth)
