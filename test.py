import pandas as pd
from postprocess import redd_accuracy
import numpy as np

x = np.array([[1 ,2], [3, 4]])
y = np.array([[0, 1], [1, 2]])

print(x)
print(y)
accu = redd_accuracy(x, y)
print(accu)

"""
# Replace this with the path to your CSV file
file_path = 'csv_files/house1_mod.csv'

# Load the data from the CSV file
df = pd.read_csv(file_path)

# Calculate the total sum of the values in each column (including the new devices)
total_agg = df['agg'].sum()
total_st = df['st'].sum()
total_wh = df['wh'].sum()
total_wm = df['wm'].sum()
total_fridge = df['fridge'].sum()
total_dw = df['dw'].sum()
total_ac = df['ac'].sum()

# Assuming the new devices (tv1, PC, CofM, mWave) are in the CSV
# Calculate the total sum for the new devices
total_tv1 = df['tv1'].sum() if 'tv1' in df.columns else 0
total_PC = df['PC'].sum() if 'PC' in df.columns else 0
total_CofM = df['CofM'].sum() if 'CofM' in df.columns else 0
total_mWave = df['mWave'].sum() if 'mWave' in df.columns else 0

# Calculate the total percentage of each device with respect to the total 'agg'
st_percentage = (total_st / total_agg) * 100
wh_percentage = (total_wh / total_agg) * 100
wm_percentage = (total_wm / total_agg) * 100
fridge_percentage = (total_fridge / total_agg) * 100
dw_percentage = (total_dw / total_agg) * 100
ac_percentage = (total_ac / total_agg) * 100
tv1_percentage = (total_tv1 / total_agg) * 100
PC_percentage = (total_PC / total_agg) * 100
CofM_percentage = (total_CofM / total_agg) * 100
mWave_percentage = (total_mWave / total_agg) * 100

# Print the total percentages
print(f"Total Percentage of st: {st_percentage:.2f}%")
print(f"Total Percentage of wh: {wh_percentage:.2f}%")
print(f"Total Percentage of wm: {wm_percentage:.2f}%")
print(f"Total Percentage of fridge: {fridge_percentage:.2f}%")
print(f"Total Percentage of dw: {dw_percentage:.2f}%")
print(f"Total Percentage of ac: {ac_percentage:.2f}%")
print(f"Total Percentage of tv1: {tv1_percentage:.2f}%")
print(f"Total Percentage of PC: {PC_percentage:.2f}%")
print(f"Total Percentage of CofM: {CofM_percentage:.2f}%")
print(f"Total Percentage of mWave: {mWave_percentage:.2f}%")

# Optionally, save the total percentages to a new CSV file
percentages_df = pd.DataFrame({
    'Device': ['st', 'wh', 'wm', 'fridge', 'dw', 'ac', 'tv1', 'PC', 'CofM', 'mWave'],
    'Total Percentage': [st_percentage, wh_percentage, wm_percentage, fridge_percentage, dw_percentage,
                         ac_percentage, tv1_percentage, PC_percentage, CofM_percentage, mWave_percentage]
})

percentages_df.to_csv('total_percentages_with_new_devices.csv', index=False)
"""

