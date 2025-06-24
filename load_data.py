import pandas as pd
import psycopg2
import os


def get_column_names(db_config, clientid_value, shellyid_value):
    """
    Fetch column names for a specific clientid and shellyid combination from the 'shellypro3em' table.

    Parameters:
    - db_config: Dictionary containing the database connection details.
    - client_id: The client 'id' e.g. 'house1'.
    - shelly_id: The shelly 'id' e.g. 'aggregate'.

    Returns:
    - A list of column names for the specified clientid and shellyid.
    """

    # Establish connection to the PostgreSQL database using the config dictionary
    try:
        connection = psycopg2.connect(
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        )

        cursor = connection.cursor()

        # Query to get the column names by selecting a single row (LIMIT 1) for the given clientid and shellyid
        # The table names can be shellypro3em or shellyplugS

        query = f"""
            SELECT * FROM public.shellyplugS 
            WHERE clientid = %s AND shellyid = %s LIMIT 1
        """
        cursor.execute(query, (clientid_value, shellyid_value))

        # Get the column names from cursor.description
        column_names = [desc[0] for desc in cursor.description]

        # Return the column names
        return column_names

    except Exception as e:
        print(f"Error fetching column names from the database: {e}")
        return []

    finally:
        # Ensure the database connection is closed
        if connection:
            cursor.close()
            connection.close()

def fetch_agg_data (db_config, client_id, shelly_id,  start_date, end_date):
    """
    Fetch aggregate data for a given clientid and shellyid from the 'shellypro3em' table.

    Parameters:
    - db_config: Dictionary containing the database connection details.
    - client_id: The client 'id' e.g. 'house1'.
    - shelly_id: The shelly 'id' e.g. 'aggregate'.
    - start_date: The start date of data.
    - end_date: The end date of data.

    Returns:
    - DataFrame containing the data fetched from the database.
    """

    columns = ['clientid', 'time', 'total_aprt_power', 'total_act_power']

    # Build the SQL query string dynamically for the specified columns
    query = f"""
        SELECT {', '.join(columns)}
        FROM public.shellypro3em
        WHERE clientid = %s AND shellyid = %s
        AND time >= %s AND time < %s
        ORDER BY time DESC
    """

    print("Connecting to the database to fetch Aggregate data ...")  # Prompt before connecting to the database
    # Establish connection to the PostgreSQL database using the config dictionary
    try:
        connection = psycopg2.connect(
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        )

        print("Successfully connected to the database")  # Inform user that connection was successful

        # Prepare the query parameters
        params = (client_id, shelly_id, start_date, end_date)

        # Execute the query
        cursor = connection.cursor()
        cursor.execute(query, params)

        # Fetch the results into a pandas DataFrame
        df = pd.DataFrame(cursor.fetchall(), columns= columns)

        print ("Successfully fetched Aggregate data")

    except Exception as e:
        print(f"Error fetching data from the database: {e}")
        df = pd.DataFrame()  # Return an empty DataFrame if there is an error

    finally:
        # Ensure the database connection is closed
        if connection:
            cursor.close()
            connection.close()
            print("Disconnected from the database.")
    return df


def fetch_3phase_data(db_config, client_id, shelly_id, start_date, end_date):
    """
    Fetch device data from 3-phase meter for a given clientid and shellyid from the 'shellypro3em' table.
    Calculates the apparent power for each phase (A, B, C) and adds them to the DataFrame.

    Parameters:
    - db_config: Dictionary containing the database connection details.
    - client_id: The client 'id' e.g. 'house1'.
    - shelly_id: The shelly 'id' e.g. 'st_wh_wm'.
    - start_date: The start date of data.
    - end_date: The end date of data.

    Returns:
    - DataFrame containing the data fetched from the database along with the apparent power for each phase.
    """

    # Define columns to fetch from the database
    columns = ['clientid', 'time', 'a_voltage', 'b_voltage', 'c_voltage', 'a_current', 'b_current', 'c_current']

    # Build the SQL query string dynamically for the specified columns
    query = f"""
        SELECT {', '.join(columns)}
        FROM public.shellypro3em
        WHERE clientid = %s AND shellyid = %s
        AND time >= %s AND time < %s
        ORDER BY time DESC
    """

    print(
        "Connecting to the database to fetch 3-phase meter device data ...")  # Prompt before connecting to the database
    # Establish connection to the PostgreSQL database using the config dictionary
    try:
        connection = psycopg2.connect(
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        )

        print("Successfully connected to the database.")  # Inform user that connection was successful

        # Prepare the query parameters
        params = (client_id, shelly_id, start_date, end_date)

        # Execute the query
        cursor = connection.cursor()
        cursor.execute(query, params)

        # Fetch the results into a pandas DataFrame
        df = pd.DataFrame(cursor.fetchall(), columns=columns)

        # Calculate the apparent power for each phase (S = V * I)
        df['a_aprt_power'] = df['a_voltage'] * df['a_current']
        df['b_aprt_power'] = df['b_voltage'] * df['b_current']
        df['c_aprt_power'] = df['c_voltage'] * df['c_current']

        print("Successfully fetched 3-phase meter device data and calculated apparent power")

        # Drop the voltage and current columns
        df.drop(columns=['a_voltage', 'b_voltage', 'c_voltage', 'a_current', 'b_current', 'c_current'], inplace=True)

    except Exception as e:
        print(f"Error fetching data from the database: {e}")
        df = pd.DataFrame()  # Return an empty DataFrame if there is an error

    finally:
        # Ensure the database connection is closed
        if connection:
            cursor.close()
            connection.close()
            print("Disconnected from the database.")

    return df

def fetch_plug_data(db_config, client_id, shelly_id, start_date, end_date):
    """
    Fetch data from different plugs.

    Parameters:
    - db_config: Dictionary containing the database connection details.
    - client_id: The client 'id' e.g. 'house1'.
    - shelly_id: The shelly 'id' e.g. 'wm'.
    - start_date: The start date of data.
    - end_date: The end date of data.

    Returns:
    - DataFrame containing the data fetched from the database for the specified plug.
    """

    # Define columns to fetch from the database
    columns = ['clientid', 'time', 'apower']

    # Build the SQL query string dynamically for the specified columns
    query = f"""
        SELECT {', '.join(columns)}
        FROM public.shellyplugS
        WHERE clientid = %s AND shellyid = %s
        AND time >= %s AND time < %s
        ORDER BY time DESC
    """

    print(
        "Connecting to the database to fetch plug data ...")  # Prompt before connecting to the database
    # Establish connection to the PostgreSQL database using the config dictionary
    try:
        connection = psycopg2.connect(
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        )

        print("Successfully connected to the database.")  # Inform user that connection was successful

        # Prepare the query parameters
        params = (client_id, shelly_id, start_date, end_date)

        # Execute the query
        cursor = connection.cursor()
        cursor.execute(query, params)

        # Fetch the results into a pandas DataFrame
        df = pd.DataFrame(cursor.fetchall(), columns=columns)

        print("Successfully fetched plug data")

    except Exception as e:
        print(f"Error fetching data from the database: {e}")
        df = pd.DataFrame()  # Return an empty DataFrame if there is an error

    finally:
        # Ensure the database connection is closed
        if connection:
            cursor.close()
            connection.close()
            print("Disconnected from the database.")

    return df


def merge_dataframes(dataframes_list, plug_prefix='plug'):
    """
    Merge multiple dataframes on the 'time' column.

    Parameters:
    - dataframes_list: A list of DataFrames to merge. All DataFrames should have a 'time' column to merge on.

    Returns:
    - Merged DataFrame with 'time' as the index and all the data from the given DataFrames.
    """
    if not dataframes_list:
        raise ValueError("The list of dataframes is empty.")

    # Start with the first DataFrame in the list
    merged_df = dataframes_list[0]

    # Sequentially merge the rest of the DataFrames on 'time' column
    for i, df in enumerate(dataframes_list[1:], start=1):
        # Drop 'clientid' from the other DataFrame if it already exists in the merged DataFrame
        if 'clientid' in df.columns and 'clientid' in merged_df.columns:
            df = df.drop(columns=['clientid'])

        df_renamed = df.rename(columns={'apower': f'{plug_prefix}{i}_apower'})
        merged_df = pd.merge(merged_df, df_renamed, on='time', how='outer')  # Merge using outer join to preserve all timestamps

    return merged_df

def save_dataset(df, filename, folder='csv_files'):
    """
    Save the DataFrame to a CSV file in the specified folder.

    Parameters:
    - df: The pandas DataFrame to save.
    - filename: The name of the CSV file (without the '.csv' extension).
    - folder: The folder where the CSV will be stored (default is 'csv_files').

    Returns:
    - None
    """
    # Ensure the folder exists, if not, create it
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Folder '{folder}' created.")

    # Create the full path for the CSV file
    file_path = os.path.join(folder, f"{filename}.csv")

    # Save the DataFrame to the CSV file
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved to {file_path}")
