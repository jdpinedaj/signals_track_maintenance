#!##########################################
#!############# IMPORTS ####################
#!##########################################

# Standard Library Imports
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.signal import stft
from typing import Tuple

# Local Imports
from src.logs import logger


def extract_features(signal: np.ndarray, sampling_frequency: float) -> pd.DataFrame:
    """
    Extract relevant features from the magnitude spectrogram of the given signal.
    Args:
        signal (numpy.ndarray): Input signal.
        sampling_frequency (float): Sampling frequency in Hz.
    Returns:
        pandas.DataFrame: Features extracted from the magnitude spectrogram.
    """
    try:
        # Compute Short-Time Fourier Transform (STFT)
        frequencies, times, Zxx = stft(signal, fs=sampling_frequency)

        # Compute magnitude spectrogram
        magnitude_spectrogram = np.abs(Zxx)

        # Extract relevant features from the magnitude spectrogram
        # Features extracted: mean, standard deviation, maximum, and minimum
        features = {
            "mean": np.mean(magnitude_spectrogram, axis=1),  # Mean of each column
            "std": np.std(
                magnitude_spectrogram, axis=1
            ),  # Standard deviation of each column
            "max": np.max(
                magnitude_spectrogram, axis=1
            ),  # Maximum value of each column
            "min": np.min(
                magnitude_spectrogram, axis=1
            ),  # Minimum value of each column
        }

        # Convert to DataFrame for ease of use
        features_df = pd.DataFrame(features)

        return features_df, magnitude_spectrogram
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise e


def train_anomaly_detection_model_isolation_forest(
    features: pd.DataFrame, contamination: float
) -> Tuple[IsolationForest, StandardScaler]:
    """
    Train an Isolation Forest model for anomaly detection.
    Args:
        features (pd.DataFrame): DataFrame containing the extracted features.
        contamination (float): The proportion of outliers in the data set.
    Returns:
        model (IsolationForest): Trained Isolation Forest model.
        scaler (StandardScaler): Scaler used to normalize the features.
    """
    try:
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)

        # Train Isolation Forest
        model = IsolationForest(contamination=contamination)
        model.fit(features_normalized)

        return model, scaler
    except Exception as e:
        logger.error(f"Error training Isolation Forest model: {e}")
        raise e


def detect_anomalies_isolation_forest(
    model: IsolationForest, scaler: StandardScaler, features: pd.DataFrame
) -> np.ndarray:
    """
    Detect anomalies using the trained Isolation Forest model.
    Args:
        model (IsolationForest): Trained Isolation Forest model.
        scaler (StandardScaler): Scaler used to normalize the features.
        features (pd.DataFrame): DataFrame containing the extracted features.
    Returns:
        anomaly_indices (np.ndarray): Indices of the detected anomalies.
    """
    try:
        features_normalized = scaler.transform(features)
        anomalies = model.predict(features_normalized)
        anomaly_indices = np.where(anomalies == -1)[0]

        return anomaly_indices
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        raise e


def plot_anomalies_isolation_forest(
    signal: np.ndarray, anomaly_indices: np.ndarray
) -> None:
    """
    Plot the signal and highlight the detected anomalies.
    Args:
        signal (np.ndarray): The original signal.
        anomaly_indices (np.ndarray): Indices of the detected anomalies.
    Returns:
        None
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(signal, label="Signal")
        plt.scatter(
            anomaly_indices, signal[anomaly_indices], color="r", label="Anomalies"
        )
        plt.legend()
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting anomalies: {e}")
        raise e


# TODO: Pass the functions of unsupervised.
