import tensorflow as tf
from keras_tuner.tuners import RandomSearch

def nn_tuning(X_train, y_train, input_shape, target_shape, project_name="my_model_tuning", max_trials=10):
    def build_model(hp):
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(
                units=hp.Choice('units_1', [64, 128, 256]),
                activation=hp.Choice('activation_1', ['relu', 'tanh']),
                input_shape=input_shape,
                return_sequences=True
            ),
            tf.keras.layers.LSTM(
                units=hp.Choice('units_2', [32, 64, 128]),
                activation=hp.Choice('activation_2', ['relu', 'tanh']),
                return_sequences=False
            ),
            tf.keras.layers.RepeatVector(input_shape[0]),
            tf.keras.layers.LSTM(
                units=hp.Choice('units_3', [64, 128, 256]),
                activation=hp.Choice('activation_3', ['relu', 'tanh']),
                return_sequences=True
            ),
            tf.keras.layers.LSTM(
                units=hp.Choice('units_4', [32, 64, 128]),
                activation=hp.Choice('activation_4', ['relu', 'tanh']),
                return_sequences=True
            ),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(target_shape, activation='linear')
            )
        ])

        lr = hp.Choice('learning_rate', [1e-5, 1e-4, 1e-3, 1e-2])
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name=project_name
    )

    for batch_size in [16, 32, 64, 128, 256, 512, 1024]:
        print(f"Searching with batch size: {batch_size}")
        tuner.search(
            X_train, y_train,
            epochs=3,
            validation_split=0.1,
            batch_size=batch_size,
            verbose=1
        )

    # Get the best model and hyperparameters
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    return best_model, best_hps
