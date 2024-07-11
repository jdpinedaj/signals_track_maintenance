import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
import matplotlib.pyplot as plt
from typing import Tuple


def preprocess_data(
    signal: pd.Series, window_size: int, step_size: int
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Preprocesses the given signal by converting it from a pandas Series to a numpy array, normalizing it using the StandardScaler, and creating sequences with sliding windows.
    Args:
        signal (pd.Series): The signal data to preprocess.
        window_size (int): The size of the sliding window.
        step_size (int): The step size between windows.
    Returns:
        sequences (np.ndarray): The preprocessed sequences.
        scaler (StandardScaler): The scaler used to preprocess the signal.
    """
    # Convert pandas Series to numpy array
    signal = signal.to_numpy()

    # Normalize the signal
    scaler = StandardScaler()
    signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()

    # Create sequences with sliding windows
    sequences = [
        signal[start : start + window_size]
        for start in range(0, len(signal) - window_size, step_size)
    ]
    sequences = np.array(sequences)

    return sequences, scaler


def create_lstm_autoencoder(sequence_length: int) -> Sequential:
    """
    Create a Long Short-Term Memory (LSTM) autoencoder model.
    Args:
        sequence_length (int): The length of each input sequence.
    Returns:
        model (Sequential): The compiled LSTM autoencoder model.
    """

    # Define the model architecture
    model = Sequential(
        [
            # Encoder layers
            LSTM(
                64,
                activation="relu",
                input_shape=(sequence_length, 1),
                return_sequences=True,  # Return sequences for the next LSTM layer
            ),
            LSTM(32, activation="relu", return_sequences=False),
            # Decoder layers
            RepeatVector(
                sequence_length
            ),  # Repeat the input sequence to match the shape
            LSTM(32, activation="relu", return_sequences=True),
            LSTM(64, activation="relu", return_sequences=True),
            # Output layer
            TimeDistributed(Dense(1)),  # Output a single value for each time step
        ]
    )

    # Compile the model
    model.compile(optimizer="adam", loss="mse")

    return model


def create_smaller_lstm_autoencoder(sequence_length):
    """
    Create a smaller Long Short-Term Memory (LSTM) autoencoder model.
    Args:
        sequence_length (int): The length of each input sequence.
    Returns:
        model (Sequential): The compiled LSTM autoencoder model.
    """

    if not isinstance(sequence_length, int) or sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer")

    # Define the model architecture
    model = Sequential(
        [
            # Encoder layers
            LSTM(
                16,
                activation="relu",
                input_shape=(sequence_length, 1),
                return_sequences=True,  # Return sequences for the next LSTM layer
            ),
            LSTM(8, activation="relu", return_sequences=False),
            # Decoder layers
            RepeatVector(
                sequence_length
            ),  # Repeat the input sequence to match the shape
            LSTM(8, activation="relu", return_sequences=True),
            LSTM(16, activation="relu", return_sequences=True),
            # Output layer
            TimeDistributed(Dense(1)),  # Output a single value for each time step
        ]
    )

    # Compile the model
    model.compile(optimizer="adam", loss="mse")

    return model


def train_lstm_autoencoder(
    model: Sequential, data: np.ndarray, epochs: int = 50, batch_size: int = 32
) -> None:
    """
    Trains the LSTM autoencoder model.
    Args:
        model (Sequential): The LSTM autoencoder model to train.
        data (numpy.ndarray): The input data to train the model on.
        epochs (int, optional): The number of epochs to train the model. Defaults to 50.
        batch_size (int, optional): The batch size for training. Defaults to 32.
    """
    # Train the LSTM autoencoder model
    # The model is trained on the input data, where the input and output are the same
    # The validation split is set to 0.1, meaning 10% of the data is used for validation
    model.fit(data, data, epochs=epochs, batch_size=batch_size, validation_split=0.1)


def detect_anomalies(
    model: Sequential, data: np.ndarray, threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect anomalies in the given data using the LSTM autoencoder model.

    Args:
        model (Sequential): The trained LSTM autoencoder model.
        data (numpy.ndarray): The input data.
        threshold (float): The MSE threshold to consider a data point as an anomaly.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the anomaly indices and
        the MSE values for each data point.
    """
    # Predict the output of the LSTM autoencoder model for the given input data
    predictions = model.predict(data)

    # Calculate the Mean Squared Error (MSE) between the predicted and actual values
    mse = np.mean(np.power(data - predictions, 2), axis=1)

    if threshold is None:
        return None, mse

    anomalies = mse > threshold
    return anomalies, mse


def determine_threshold(mse: np.ndarray, contamination: float = 0.01) -> float:
    """
    Determine the threshold for anomaly detection based on the given MSE values.
    Args:
        mse (numpy.ndarray): The Mean Squared Error values.
        contamination (float, optional): The proportion of outliers. Defaults to 0.01.
    Returns:
        float: The threshold value.
    """
    # Sort the MSE values in ascending order
    sorted_mse = np.sort(mse)

    # Calculate the index of the threshold value
    # The threshold is typically the value at the (1 - contamination) quantile
    threshold_index = int((1 - contamination) * len(sorted_mse))

    # Extract the threshold value from the sorted MSE values
    threshold = sorted_mse[threshold_index]

    return threshold


def plot_anomalies(
    signal: np.ndarray, anomaly_indices: np.ndarray, step_size: int
) -> None:
    """
    Plot the signal and highlight the detected anomalies.
    Args:
        signal (np.ndarray): The original signal.
        anomaly_indices (np.ndarray): Indices of the detected anomalies.
        step_size (int): The step size used to extract sequences from the signal.
    Returns:
        None
    """
    # Create a figure with a specified size
    plt.figure(figsize=(10, 6))

    # Plot the original signal
    plt.plot(signal, label="Signal")

    # Highlight the detected anomalies
    plt.scatter(
        anomaly_indices * step_size,  # Multiply the anomaly indices by the step size
        signal[anomaly_indices * step_size],  # Extract the corresponding signal values
        color="r",
        label="Anomalies",
    )

    # Add a legend to the plot
    plt.legend()

    # Display the plot
    plt.show()
