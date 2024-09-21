# hyperparameter_experiments.py

# run exp on the number of layers
from typing import List
import os
import subprocess
import time
import csv

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the base command for running the training script
BASE_COMMAND = [
    "python3", "train.py", "config/train_shakespeare_char.py",
    "--device=cpu",
    "--compile=False",
    "--eval_iters=20",
    "--log_interval=1",
    "--block_size=64",
    "--batch_size=12",
    "--dropout=0.0"
]

# Set defaults for model arch
DEFAULT_N_HEAD: int = 4
DEFAULT_N_LAYER: int = 4
DEFAULT_N_EMBED: int = 128

def experiment(exp_var: str, 
               values: List[int], 
               experiment_config: dict):

    # Set Base Command
    base_command = experiment_config.get("base_command", BASE_COMMAND)

    # Set Max Iters
    max_iters = experiment_config.get("max_iters", 2000)
    print(max_iters)
    lr_decay_iters = max_iters

    command = base_command + [
        f"--max_iters={max_iters}",
        f"--lr_decay_iters={lr_decay_iters}"
    ]

    if exp_var == "n_head":
        # set arch vals
        n_layer = experiment_config.get("n_layer", DEFAULT_N_LAYER)
        n_embed = experiment_config.get("n_embed", DEFAULT_N_EMBED)

        # add to command
        command + [
            f"--n_layer={n_layer}",
            f"--n_embd={n_embed}"
        ]


    elif exp_var == "n_layer":
        n_embed = experiment_config.get("n_embed", DEFAULT_N_EMBED)
        n_head = experiment_config.get("n_head", DEFAULT_N_HEAD)

        command + [
            f"--n_head={n_head}",
            f"--n_embd={n_embed}"
        ]

    elif exp_var == "n_embed":
        n_layer = experiment_config.get("n_layer", DEFAULT_N_LAYER)
        n_head = experiment_config.get("n_head", DEFAULT_N_HEAD)

        command + [
            f"--n_layer={n_layer}",
            f"--n_head={n_head}"
        ]

    else:
        raise NotImplementedError

    results = []

    # Iterate over the configurations
    for val in values:
        c = command + [
            f"--{exp_var}={val}",
            f"--out_dir=model-out-{exp_var}-{val}"
        ]

        print(f"------------{exp_var}={val}------------")
        try:
            start = time.time()
            result = subprocess.run(c, capture_output=True, text=True)
            if result.returncode != 0:
                # If there is an error, print it and continue to the next configuration
                print(f"Error occurred for {exp_var} = {val}")
                print(result.stderr)
                break

            # If no error, extract the final loss from the output
            end = time.time()
            print(f"Training time (seconds): {end - start}, (mins) {(end - start)/60}")
            output_lines = result.stdout.splitlines()
            for line in output_lines:
                if line.startswith("step"):
                    step_num = int(line.split(" ")[1].strip(":"))
                    train_loss = float(line.split("train loss")[1].split(",")[0].strip())
                    val_loss = float(line.split("val loss")[1].strip())
                    results.append([exp_var, val, step_num, train_loss, val_loss])
                if line.startswith(f"step {max_iters}:"):
                    print(f"Final Results for {exp_var} = {val}: {line}")
        except Exception as e:
            print(f"An exception occurred for configuration {exp_var} = {val}: {str(e)}")

    # Define the CSV file path
    csv_file = f"loss_values_{exp_var}.csv"

    # Create and open the CSV file for writing
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow([exp_var, "value", "step", "train_loss", "val_loss"])

        # Write all collected data to the CSV file
        writer.writerows(results)

    print(f"Loss values have been written to {csv_file}.")

def graph_losses(csv_file: str,
                 exp_var: str):

    # Read the CSV file
    df = pd.read_csv(csv_file)

    markers = ['D', '>', 'o', 's', '*']
    palette = sns.color_palette("husl", n_colors=len(df['value'].unique()))

    # Plot the data
    plt.figure(figsize=(12, 6))
    for i, val in enumerate(df['value'].unique()):
        subset = df[df['value'] == val]
        plt.plot(subset['step'], 
                 subset['train_loss'], 
                 label=f'Train Loss ({exp_var}={val})',
                 marker=markers[i],
                 color=palette[i])
        plt.plot(subset['step'], 
                 subset['val_loss'], 
                 '--', 
                 label=f'Val Loss ({exp_var}={val})',
                 marker=markers[i],
                 color=palette[i])

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Train and Validation Loss for {exp_var}')
    plt.legend()
    plt.grid()
    plt.savefig(f"{exp_var}.png")
    plt.show()

if __name__ == "__main__":

    # for exp_var in ["n_layer", "n_head", "n_embed"]:


    exp_var = "n_layer"
    experiment(exp_var=exp_var,
               values=[4, 8, 16, 32],
               experiment_config={
                   "max_iters": 2000
               })
    
    graph_losses(csv_file=f"loss_values_{exp_var}.csv",
                 exp_var=exp_var)



## graph 

# run exp on the hidden dimension


# run on number of heads