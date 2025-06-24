import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_device_power(df):
    # Ensure 'time' column is in datetime format
    #df['time'] = pd.to_datetime(df['time'])

    plt.figure(figsize=(12, 7))  # Slightly larger figure for clarity

    # Loop through each column (except 'clientid' and 'time') and plot it
    for column in df.columns:
        if column not in ['clientid', 'time']:
            plt.plot(df.index, df[column], label=column)

    # Add labels, title, legend, and grid with larger fonts
    plt.xlabel('Time (sec)', fontsize=16)
    plt.ylabel('Power (VA-W)', fontsize=16)
    plt.title('All Columns vs Time', fontsize=18)
    plt.xticks(rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to make sure everything fits
    plt.show()


def compare_plot(aggregate_file, real_file, pred_file):
    """
    Plot and compare real and predicted values for each device in subplots, including aggregate signal.

    Args:
        aggregate_file (str): Path to the CSV file containing the aggregate signal.
        real_file (str): Path to the CSV file containing the real values.
        pred_file (str): Path to the CSV file containing the predicted values.
    """
    # Load the CSV files
    aggregate = pd.read_csv(aggregate_file)
    real_values = pd.read_csv(real_file)
    predictions = pd.read_csv(pred_file)

    # Check if the columns match
    if list(real_values.columns) != list(predictions.columns):
        raise ValueError("The columns in the real and predicted files do not match.")

    # Number of devices (columns) and create subplots (+1 for aggregate)
    num_devices = len(real_values.columns)
    fig, axes = plt.subplots(num_devices + 1, 1, figsize=(12, 4 * (num_devices + 1)), sharex=True)

    # Create subplots (+1 for aggregate)
    fig, axes = plt.subplots(num_devices + 1, 1, figsize=(12, 4 * (num_devices + 1)), sharex=True)

    # Make sure axes is always iterable
    axes = np.atleast_1d(axes)

    # Plot aggregate signal (assume single column)
    aggregate_column = aggregate.columns[0]
    axes[0].plot(aggregate[aggregate_column], label="Aggregate", color="green")
    axes[0].set_title(" Aggregate Signal ", fontsize=18, backgroundcolor='black', color='white')
    axes[0].grid(True)
    axes[0].tick_params(axis='both', labelsize=14)

    # Plot each device
    for i, column in enumerate(real_values.columns):
        axes[i + 1].plot(real_values[column], label="Real", color="blue", alpha=0.7)
        axes[i + 1].plot(predictions[column], label="Predicted", color="orange", alpha=0.7)
        axes[i + 1].grid(True)
        axes[i + 1].set_title(f" {column} ", fontsize=18, backgroundcolor='black', color='white')
        axes[i + 1].tick_params(axis='both', labelsize=14)

    # Add axis labels
    fig.supxlabel("Timestamp (sec)", fontsize=18)
    fig.supylabel("Apparent Power Consumption (VA)", fontsize=18, x=0.02)

    # Add a single legend for all plots
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    plt.show()



def plot_consumption_percentage(x, y_true, y_pred, devices):
    device_power_true_percentage = []
    device_power_pred_percentage = []

    # Aggregate total power
    agg_power = np.sum(x[:, :1], axis=0)

    # Calculate percentages
    for i in range(y_true.shape[1]):
        device_power_true_percentage.append((np.sum(y_true[:, i:i + 1], axis=0) / agg_power)[0])
        device_power_pred_percentage.append((np.sum(y_pred[:, i:i + 1], axis=0) / agg_power)[0])

    # Calculate the sum of percentages
    total_true_percentage = sum(device_power_true_percentage)
    total_pred_percentage = sum(device_power_pred_percentage)

    # Add "Others" category if necessary
    if total_true_percentage < 1.0:
        device_power_true_percentage.append(1.0 - total_true_percentage)
    if total_pred_percentage < 1.0:
        device_power_pred_percentage.append(1.0 - total_pred_percentage)

    if len(device_power_true_percentage) > len(devices):
        devices.append("Others")

    # Define colors
    colors = plt.cm.Paired.colors[:len(devices)]

    # Plot True Consumption
    plt.figure(figsize=(14, 6))  # Increased figure size for larger plot

    # Plot True Power Consumption
    plt.subplot(1, 2, 1)
    wedges, _ = plt.pie(device_power_true_percentage, startangle=140, colors=colors, radius=1)
    plt.title("True Power Consumption", fontsize=18)

    # Add legend with percentages
    legend_labels = [f"{device}: {percentage * 100:.1f}%" for device, percentage in
                     zip(devices, device_power_true_percentage)]
    plt.legend(wedges, legend_labels, loc="upper right", fontsize=16)

    # Plot Predicted Power Consumption
    plt.subplot(1, 2, 2)
    wedges, _ = plt.pie(np.abs(device_power_pred_percentage), startangle=140, colors=colors, radius=1)
    plt.title("Predicted Power Consumption", fontsize=18)

    # Add legend with percentages
    legend_labels = [f"{device}: {percentage * 100:.1f}%" for device, percentage in
                     zip(devices, device_power_pred_percentage)]
    plt.legend(wedges, legend_labels, loc="upper right", fontsize=16)

    # Show the plot
    plt.show()















