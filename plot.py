import pandas as pd

from  plot_utils import compare_plot, plot_consumption_percentage

devices = ['st', 'wh', 'dw', 'kettler', 'wm', 'toaster', 'fridge']

compare_plot('csv_files/aggregate.csv', 'csv_files/real_values.csv', 'csv_files/predictions.csv')

x_df = pd.read_csv("csv_files/aggregate.csv")
x = x_df.to_numpy()

y_true_df = pd.read_csv("csv_files/real_values.csv")
y_true = y_true_df.to_numpy()

y_pred_df = pd.read_csv("csv_files/predictions.csv")
y_pred = y_pred_df.to_numpy()

plot_consumption_percentage(x, y_true, y_pred, devices)