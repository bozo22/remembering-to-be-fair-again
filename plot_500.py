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
        'FairQCM': '#FF7F0E',  # orange
        'Min': '#2CA02C',  # green
        'Reset': '#D62728',  # red
        'RNN': '#9467BD'  # purple
    }

    for filename in os.listdir(directory):
        if donuts and "donuts" not in filename or not donuts and "donuts" in filename:
            continue

        method = filename.split('_donuts')[0] if "donuts" in filename else filename.split('.csv')[0]
        color = method_colors.get(method, 'gray')  # Default to gray if method not recognized

        data_path = os.path.join(directory, filename)
        try:
            # Load data and slice to only the first 500 columns if they exist
            data = np.genfromtxt(data_path, delimiter=",")
            if data.shape[1] > 500:  # Check if there are more than 500 columns
                data = data[:, :500]  # Slice to only the first 500 columns

            mean_data = data.mean(axis=0)
            std_data = data.std(axis=0)
            x_range = np.arange(mean_data.shape[0])
            
            ax.plot(x_range, mean_data, label=method, color=color)
            ax.fill_between(x_range, mean_data - std_data, mean_data + std_data, color=color, alpha=0.2)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def create_plots(env, rt):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), layout="tight")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--rt", type=str, required=True)
    args = parser.parse_args()

    create_plots(args.env, args.rt)
