import tensorflow as tf

def create_simple_lstm_model (input_shape, device_number):
    # Define optimized AEQ-to-SEQ LSTM model with fewer layers
    model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, activation='relu', return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(device_number, activation='linear'))
        ])
    return model

def create_lstm_model (input_shape, device_number):
    # Define optimized AEQ-to-SEQ LSTM model with fewer layers
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, activation='tanh', input_shape=input_shape, return_sequences=True),
        tf.keras.layers.LSTM(32, activation='tanh', return_sequences=False),
        tf.keras.layers.RepeatVector(input_shape[0]),
        tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True),
        tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(device_number, activation='linear'))
    ])
    return model


def create_optimized_lstm_model(input_shape, device_number):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(device_number, activation='relu'))
    ])
    model.compile(optimizer='adam', loss='msle', metrics=['mae'])
    return model


# Transformer Encoder Block
def transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1):
    x = tf.keras.layers.LayerNormalization()(inputs)
    x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(x, x)
    x = tf.keras.layers.Dropout(dropout)(x)
    res = x + inputs  # Residual connection

    x = tf.keras.layers.LayerNormalization()(res)
    x = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(inputs.shape[-1])(x)
    return x + res  # Another residual connection


# Model Definition
def create_transformer_model(input_shape, device_number):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Feature extraction (CNN layer before Transformer)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(inputs)

    # Transformer Layers
    for _ in range(3):  # Stack multiple Transformer layers
        x = transformer_encoder(x)

    # Final dense output layers
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(device_number, activation="linear")(x)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model