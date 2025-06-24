import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from datetime import datetime
from load_data import fetch_agg_data, fetch_3phase_data, fetch_plug_data, merge_dataframes, save_dataset
from postprocess import save_to_csv, revert_sequences,redd_accuracy
from preprocess import create_dataframe_from_csv, fill_empty_with_zero, rename_columns, merge_ac_columns, create_sequences, drop_columns
from plot_utils import plot_device_power, compare_plot
from mdl import create_optimized_lstm_model
import tensorflow as tf
import visualkeras

existing_dataset = 1
existing_model = 0

# Database configuration
db_config = {
    'dbname': 'smartmeterlogs',
    'user': 'readsmartmetermech',
    'password': 'zocsyx-qogcEf-saxbe8',
    'host': 'smartmetermech.chu4s6qua02r.eu-central-1.rds.amazonaws.com',
    'port': '9001'
}

# Define parameters of data to be loaded
#data_params ={'client_id': 'house16', 'shelly_agg_id': 'aggregate', 'shelly_3ph_id': 'st_wh_wm', 'plug1': 'ac1', 'plug2':'fridge'}

data_params ={'client_id': 'house21', 'shelly_3ph_id': 'aggregate_st_wh', 'plug1': 'dw', 'plug2':'kettler', 'plug3': 'wm', 'plug4': 'toaster',
              'plug5': 'fridge', 'plug6': 'ac1', 'plug7': 'tv'}

#data_params ={'client_id': 'farmakeio', 'shelly_agg_id': 'aggregate', 'shelly_3ph_id': 'acfarm_acgraf_acpat'}

devices = ['agg', 'st', 'wh', 'dw', 'kettler', 'wm', 'toaster', 'fridge', 'ac', 'tv']
#devices = ['agg', 'agg_act', 'ac1', 'ac2', 'ac3']
#devices = ['agg', 'agg_act', 'st', 'wh', 'wm', 'ac', 'fridge']

if existing_dataset == 0:

    start_date = datetime(2025, 5, 23)
    end_date = datetime(2025, 6, 23)

    # fetch aggregate data and store them to dataframe
    #agg_data = fetch_agg_data(db_config, data_params['client_id'], data_params['shelly_agg_id'], start_date, end_date)

    # fetch 3-phase device data and store them to dataframe
    three_phase_data = fetch_3phase_data(db_config, data_params['client_id'], data_params['shelly_3ph_id'], start_date,
                                         end_date)
    # fetch data from plugs and store them to dataframes
    plug1_data = fetch_plug_data(db_config, data_params['client_id'], data_params['plug1'], start_date, end_date)
    print("Data from plug1 fetched")
    plug2_data = fetch_plug_data(db_config, data_params['client_id'], data_params['plug2'], start_date, end_date)
    print("Data from plug2 fetched")
    plug3_data = fetch_plug_data(db_config, data_params['client_id'], data_params['plug3'], start_date, end_date)
    print("Data from plug3 fetched")
    plug4_data = fetch_plug_data(db_config, data_params['client_id'], data_params['plug4'], start_date, end_date)
    print("Data from plug4 fetched")
    plug5_data = fetch_plug_data(db_config, data_params['client_id'], data_params['plug5'], start_date, end_date)
    print("Data from plug5 fetched")
    plug6_data = fetch_plug_data(db_config, data_params['client_id'], data_params['plug6'], start_date, end_date)
    print("Data from plug6 fetched")
    plug7_data = fetch_plug_data(db_config, data_params['client_id'], data_params['plug7'], start_date, end_date)
    print("Data from plug7 fetched")
    """
    plug8_data = fetch_plug_data(db_config, data_params['client_id'], data_params['plug8'], start_date, end_date)
    print("Data from plug8 fetched")
    plug9_data = fetch_plug_data(db_config, data_params['client_id'], data_params['plug9'], start_date, end_date)
    print("Data from plug9 fetched")
    plug10_data = fetch_plug_data(db_config, data_params['client_id'], data_params['plug10'], start_date, end_date)
    print("Data from plug10 fetched")
    plug11_data = fetch_plug_data(db_config, data_params['client_id'], data_params['plug11'], start_date, end_date)
    print("Data from plug11 fetched")
    """

    # Collect all dataframes into a list
    dataframes_list = [
        #agg_data,
        three_phase_data,
        plug1_data,
        plug2_data,
        plug3_data,
        plug4_data,
        plug5_data,
        plug6_data,
        plug7_data,
        #plug8_data,
        #plug9_data,
        #plug10_data,
        #plug11_data,
    ]

    # Merge all the DataFrames and save to csv
    merged_df = merge_dataframes(dataframes_list)
    save_dataset(merged_df, 'house21')

else:
    # preprocess dataset
    loaded_data = create_dataframe_from_csv('csv_files/house21.csv')
    loaded_data.drop_duplicates(subset=['time'], keep='first', inplace=True)
    renamed_df = rename_columns(loaded_data, devices)
    #df_dropped_col = drop_columns(renamed_df, ['toaster', 'ac'])
    #df_merged_ac = merge_ac_columns(renamed_df)
    dropped_df = renamed_df.dropna()
    save_dataset(dropped_df, 'house21_mod')

    # Load the dataset
    data = pd.read_csv("csv_files/house21_mod.csv")

    # Preprocessing
    data['time'] = pd.to_datetime(data['time'])  # Convert timestamp to datetime
    data.set_index('time', inplace=True)  # Set time as the index
    #data = data[~((data.index >= '2025-04-16') & (data.index <= '2025-04-22'))]
    plot_device_power(data)

    # Selecting features
    aggregated_power = data[['agg']].values
    device_power = data[['st', 'wh', 'dw', 'kettler', 'wm', 'toaster', 'fridge']].values  # Device values

    # Normalize data
    scaler_agg = MinMaxScaler()
    aggregated_power_scaled = scaler_agg.fit_transform(aggregated_power)
    joblib.dump(scaler_agg, "models/scaler_agg_house21_all_devices_wp.pkl")

    scaler_appliances = MinMaxScaler()
    appliances_scaled = scaler_appliances.fit_transform(device_power)
    joblib.dump(scaler_appliances, "models/scaler_appliances_house21_all_devices_wp.pkl")

    X= np.array(aggregated_power_scaled)#[:28*86400, :]
    y= np.array(appliances_scaled)#[:28*86400, :]
    print(X.shape)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=None, shuffle=False)

    # Create sequence
    SEQ_LENGTH = 128  # Define the sequence length (e.g., 60 seconds)
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, SEQ_LENGTH)

if existing_model == 0:
    # Create the model
    model = create_optimized_lstm_model((SEQ_LENGTH, X_train_seq.shape[2]), y_train_seq.shape[2])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    model.fit(X_train_seq, y_train_seq, validation_split=0.07, epochs=20, batch_size=128, verbose=1)

    # Save the trained model
    model.save("models/house21_all_devices_wp.keras")
    model.save("models/house21_all_devices_wp.h5")

    # Evaluate the model
    loss, mae = model.evaluate(X_test_seq, y_test_seq, verbose=1)
    print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

else:
    # Load existing model
    model = tf.keras.models.load_model("models/farmakeio_all_devices.keras")
    image = visualkeras.layered_view(
        model,
        legend=True,
        spacing=30,
        draw_volume=True,
        max_z=100,
        scale_xy=1.5,
        scale_z=1,
    )
    image.save("model_architecture.png")

    # Make predictions
y_pred = model.predict(X_test_seq)

# Inverse scale the predictions
X_test_rescaled = scaler_agg.inverse_transform(X_test)
y_test_rescaled = scaler_appliances.inverse_transform(y_test_seq.reshape(-1, y_test_seq.shape[2])).reshape(y_test_seq.shape)
y_pred_rescaled = scaler_appliances.inverse_transform(y_pred.reshape(-1, y_test_seq.shape[2])).reshape(y_pred.shape)

# Reconstruct the original signals from the sequences
y_test_reconstructed = revert_sequences(y_test_rescaled, SEQ_LENGTH)
y_pred_reconstructed = revert_sequences(y_pred_rescaled, SEQ_LENGTH)

print('Model r2_accuracy after reconstruction is:', r2_score(y_test_reconstructed, y_pred_reconstructed))
print("redd accuracy of reconstructed signals is:", redd_accuracy(y_test_reconstructed, y_pred_reconstructed))

# save real values and predictions to csv
save_to_csv(y_test_reconstructed, y_pred_reconstructed)
X_test_df = pd.DataFrame(X_test_rescaled)
X_test_df.to_csv("csv_files/aggregate.csv", index=False)
compare_plot("csv_files/aggregate.csv",'csv_files/real_values.csv', 'csv_files/predictions.csv')
