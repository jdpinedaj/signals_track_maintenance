#!##########################################
#!############# IMPORTS ####################
#!##########################################

# Standard Library Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, stft
from scipy.signal.windows import hamming
from typing import Tuple, Optional, NoReturn
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import plotly.express as px
import plotly.graph_objects as go

# New imports (test supervised contrastive learning)
# import torch
# from torch.utils.data import DataLoader

# Additional imports for supervised contrastive learning
# from src.anomaly_detection.REFACTORED_supervised_contrastive_learning import (
#     RailwayDataset,
#     ResNetEncoder,
#     PrototypicalNetwork,
#     train_prototypical_network,
#     evaluate_model,
# )


# Local Imports
from src.logs import logger

#!##########################################
#!############ SUB FUNCTIONS ###############
#!##########################################


def _load_dat1_data(file_path: str) -> np.ndarray:
    """
    Load .dat1 file, assuming it's a text file with numerical data.
    """
    signal = np.loadtxt(file_path)
    return signal


def _load_mat_data(file_path: str, key: str = "datos_acel") -> np.ndarray:
    """
    Load .mat file. Default key is 'datos_acel'.
    """
    data = loadmat(file_path)
    signal = data[key].flatten()
    return signal


#!##########################################
#!############# FUNCTIONS ##################
#!##########################################


def preprocess_dat1_data(
    data_path: str,
    acel_to_process: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load and preprocess the data.
    Args:
        data_path (str): Path to the data file.
        acel_to_process (str): Acceleration data to process.
    Returns:
        data (numpy.ndarray): Preprocessed data.
        signal (numpy.ndarray): Signal to process.
        df (pd.DataFrame): Dataframe with the data.
    """
    try:
        # Load the data
        data = _load_dat1_data(data_path)
        # Add column names
        column_names = [
            "acc_long_veh_ms2",
            "acc_lat_veh_ms2",
            "acc_vert_veh_ms2",
            "speed_raw_kmh",
            "acc_vert_left_axle_box_ms2",
            "acc_vert_right_axle_box_ms2",
            "acc_lat_axle_box_ms2",
            "acc_lat_bogie_ms2",
        ]

        # Create a DataFrame
        df = pd.DataFrame(data, columns=column_names)

        signal = df[acel_to_process]

        return data, signal, df

    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return None


def preprocess_mat_data(
    data_path: str,
    acel_to_process: str,
    time_col_name: str,
    km_ref_col_name: str,
    key_f20_10: Optional[str] = "f20_10",
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, pd.DataFrame]:
    """
    Load and preprocess the data.
    Args:
        data_path (str): Path to the data file.
        acel_to_process (str): Acceleration data to process.
        time_col_name (str): Time column name.
        km_ref_col_name (str): Kilometer reference column name.
        key_f20_10 (str): Key for the .mat file, to identify the f20_10 data.
    Returns:
        data (numpy.ndarray): Preprocessed data.
        signal (numpy.ndarray): Signal to process.
        time_column (pd.Series): Time column.
        kilometer_ref (pd.Series): Kilometer reference column.
        df (pd.DataFrame): Dataframe with the data.
    """
    try:
        # Load the data
        data = _load_mat_data(data_path)
        # Add column names
        column_names = [
            "acc_long_veh_ms2",
            "acc_lat_veh_ms2",
            "acc_vert_veh_ms2",
            "speed_raw_kmh",
            "acc_vert_left_axle_box_ms2",  #! IMPORTANT
            "acc_vert_right_axle_box_ms2",  #! IMPORTANT
            "acc_lat_axle_box_ms2",  #! IMPORTANT
            "acc_lat_bogie_ms2",
            "timestamp_s",
            "distance_travelled_m",
            "kilometer_ref_no_fixed_km",
            "kilometer_ref_fixed_km",
        ]

        # Create a DataFrame
        df = pd.DataFrame(data[key_f20_10][0], columns=column_names)

        # Extract the signal and time_column
        signal = df[acel_to_process]
        time_column = df[time_col_name]
        kilometer_ref = df[km_ref_col_name]

        return data, signal, time_column, kilometer_ref, df

    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return None


def generate_time_vector_for_dat1_file(
    data: np.ndarray,
    sampling_frequency_tvp: float,
    cutoff_frequency: float,
    new_sampling_frequency: float,
    position_speed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate time vectors for the original and downsampled data.
    Args:
        data (numpy.ndarray): Data array with shape (n_samples, n_features).
        sampling_frequency_tvp (float): Sampling frequency in Hz.
        cutoff_frequency (float): Cutoff frequency in Hz.
        new_sampling_frequency (float): New sampling frequency in Hz.
        position_speed (int): Position of the speed data in the data array.
    Returns:
        original_time_vector (numpy.ndarray): Time vector for the original data.
        downsampled_time_vector (numpy.ndarray): Time vector for the downsampled data.
        velocity_data (numpy.ndarray): Velocity data.
        downsampled_velocity (numpy.ndarray): Downsampled velocity data.
    """

    # Initial setup
    num_samples = data.shape[0]  # Number of data points

    # Extract velocity data from the data array
    velocity_data = data[:, position_speed]

    # Filter parameters
    # Design a 4th order Butterworth filter
    b, a = butter(4, 2 * cutoff_frequency / sampling_frequency_tvp)
    # Apply the filter
    filtered_velocity = filtfilt(b, a, velocity_data)

    # Downsample the filtered velocity data
    # Downsampling factor
    downsampling_factor = int(sampling_frequency_tvp / new_sampling_frequency)
    # Downsample the data
    downsampled_velocity = filtered_velocity[::downsampling_factor]
    # Length of downsampled data
    num_downsampled_samples = len(downsampled_velocity)

    # Generate time vectors
    # Time vector for original data
    original_time_vector = np.arange(1, num_samples + 1) / sampling_frequency_tvp
    # Time vector for downsampled data
    downsampled_time_vector = (
        np.arange(1, num_downsampled_samples + 1) / new_sampling_frequency
    )

    # # Generate downsampled data
    # downsampled_data = np.zeros((num_downsampled_samples, 2))
    # downsampled_data[:, 0] = downsampled_time_vector
    # downsampled_data[:, 1] = downsampled_velocity

    return (
        original_time_vector,
        downsampled_time_vector,
        velocity_data,
        downsampled_velocity,
    )  # downsampled_data


def plot_time_vectors_for_dat1_file(
    original_time_vector: np.ndarray,
    downsampled_time_vector: np.ndarray,
    velocity_data: np.ndarray,
    downsampled_velocity: np.ndarray,
) -> NoReturn:
    """
    Plot the original and downsampled velocity data.
    Args:
        original_time_vector (numpy.ndarray): Time vector for the original data.
        downsampled_time_vector (numpy.ndarray): Time vector for the downsampled data.
        velocity_data (numpy.ndarray): Velocity data
        downsampled_velocity (numpy.ndarray): Downsampled velocity data
    Returns:
        None
    """

    # Plot the original velocity data
    plt.figure(1)
    plt.plot(original_time_vector, velocity_data, label="Original Velocity")
    plt.plot(
        downsampled_time_vector,
        downsampled_velocity,
        label="Filtered and Downsampled Velocity",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot the downsampled velocity data
    plt.figure(2)
    plt.plot(downsampled_time_vector, downsampled_velocity)
    plt.xlabel("Time (s)")
    plt.ylabel("Filtered and Downsampled Velocity")
    plt.grid(True)
    plt.show()


def short_term_fourier_transform_stft(
    signal: np.ndarray,
    sampling_frequency_stft: float,
    window_length: float,
    overlap: float,
    gamma: float,
    time_column: pd.Series,
    kilometer_ref: pd.Series,  # Add kilometer reference as input
    nfft: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Short-Term Fourier Transform (STFT) of a signal and align kilometer distances.
    Args:
        signal (numpy.ndarray): Input signal.
        sampling_frequency_stft (float): Sampling frequency in Hz.
        window_length (float): Window length in seconds.
        overlap (float): Overlap fraction.
        gamma (float): Dynamic margin.
        time_column (pandas.Series): Time column.
        kilometer_ref (pandas.Series): Kilometer reference column to align with STFT results.
        nfft (float): Length of the Fast Fourier Transform (FFT) used.
    Returns:
        frequencies (numpy.ndarray): Frequency vector.
        times (numpy.ndarray): Time vector.
        magnitude_spectrogram (numpy.ndarray): Magnitude spectrogram.
        X_prime (numpy.ndarray): Normalized spectrogram.
        total_time (numpy.ndarray): Total time vector.
        kilometer_ref_resampled (numpy.ndarray): Kilometer reference aligned to STFT time points.
    """

    # Initial setup
    start_time = time_column.iloc[0]  # Start time of the signal in seconds
    num_samples = len(signal)  # Number of data points
    logger.info(f"Number of samples: {num_samples}")
    total_time = (
        np.arange(num_samples) / sampling_frequency_stft + start_time
    )  # Adjusted time vector (s)
    logger.info(f"Total time: {total_time}")

    # Compute STFT
    window_samples = int(
        window_length * sampling_frequency_stft
    )  # Window length in samples
    logger.info(f"Window samples: {window_samples}")

    if window_samples < 1:
        raise ValueError(
            "window_length too small resulting in non-positive integer window_samples"
        )

    noverlap = int(overlap * window_samples)  # Number of overlapping samples
    logger.info(f"Overlap: {noverlap}")

    if noverlap >= window_samples:
        raise ValueError("overlap is too high, resulting in noverlap >= window_samples")
    window = hamming(window_samples)  # Hamming window

    # STFT calculation
    frequencies, times, Zxx = stft(
        signal,
        fs=sampling_frequency_stft,
        window=window,
        nperseg=window_samples,
        noverlap=noverlap,
        nfft=nfft,
    )

    # Adjust the times based on start_time
    times += start_time

    # Dynamic-margin normalization
    magnitude_spectrogram = np.abs(Zxx)
    epsilon = 10 ** (-gamma / 20)  # Dynamic margin. Default is 20 dB
    X_prime = (
        20
        * np.log10(
            magnitude_spectrogram / (np.max(magnitude_spectrogram) / 2) + epsilon
        )
        + gamma
    )
    X_prime = np.clip(X_prime / gamma, 0, 1)

    # Resample kilometer reference to match STFT times
    original_time = np.linspace(0, len(kilometer_ref) - 1, num=len(kilometer_ref))
    reduced_time = np.linspace(0, len(kilometer_ref) - 1, num=len(times))
    interp_func = interp1d(original_time, kilometer_ref, kind="linear")
    kilometer_ref_resampled = interp_func(reduced_time)

    return (
        frequencies,
        times,
        magnitude_spectrogram,
        total_time,
        kilometer_ref_resampled,
    )


def plot_stft_results(
    frequencies: np.ndarray,
    times: np.ndarray,
    magnitude_spectrogram: np.ndarray,
    X_prime: np.ndarray,
    total_time: np.ndarray,
    signal: np.ndarray,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot the results of the STFT analysis.
    Args:
        frequencies (numpy.ndarray): Frequency vector.
        times (numpy.ndarray): Time vector.
        magnitude_spectrogram (numpy.ndarray): Magnitude spectrogram.
        X_prime (numpy.ndarray): Normalized spectrogram.
        total_time (numpy.ndarray): Total time vector.
        signal (numpy.ndarray): Input signal.
        save_path (str): Path to save the plot.
    Returns:
        plt.Figure: The matplotlib figure object containing the plot.
    """

    # Create the figure and define a GridSpec layout
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 2, width_ratios=[20, 1], height_ratios=[1, 1, 1.2])

    # Plot 1: Original Spectrogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Original Spectrogram")
    img1 = ax1.pcolormesh(
        times, frequencies, magnitude_spectrogram, cmap="jet", shading="gouraud"
    )
    ax1.set_ylabel("Frequency [Hz]")
    ax1.grid(True)

    # Add colorbar for Plot 1
    cbar1 = fig.add_subplot(gs[0, 1])
    fig.colorbar(img1, cax=cbar1, label="Intensity [dB]")

    # Plot 2: Normalized Spectrogram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("Normalized Spectrogram")
    img2 = ax2.pcolormesh(times, frequencies, X_prime, cmap="jet", shading="gouraud")
    ax2.set_ylabel("Frequency [Hz]")
    ax2.grid(True)

    # Add colorbar for Plot 2
    cbar2 = fig.add_subplot(gs[1, 1])
    fig.colorbar(img2, cax=cbar2, label="Intensity [dB]")

    # Plot 3: Acceleration over Time
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_title("Acceleration over Time")
    ax3.plot(total_time, signal, "g")
    ax3.set_xlabel("Time [sec]")
    ax3.set_ylabel("Acceleration [m/s^2]")
    ax3.grid(True)

    # Adjust layout to ensure x-axes align
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim([times.min(), times.max()])

    # Tight layout without overlapping
    plt.tight_layout()

    # Save to file if save_path is provided
    if save_path:
        plt.savefig(save_path, format="png")
        plt.close()
    else:
        plt.show()

    return fig


def plot_stft_results_with_zero_mean(
    frequencies: np.ndarray,
    times: np.ndarray,
    magnitude_spectrogram: np.ndarray,
    total_time: np.ndarray,
    signal: np.ndarray,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot the results of the STFT analysis with zero-mean adjustment for the signal and aligned x-axes.
    Args:
        frequencies (numpy.ndarray): Frequency vector.
        times (numpy.ndarray): Time vector.
        magnitude_spectrogram (numpy.ndarray): Magnitude spectrogram.
        total_time (numpy.ndarray): Total time vector.
        signal (numpy.ndarray): Input signal.
        save_path (str): Path to save the plot.
    Returns:
        plt.Figure: The matplotlib figure object containing the plot.
    """
    # Remove the offset to make the signal zero-mean
    zero_mean_signal = signal - np.mean(signal)

    # Update the normalized spectrogram based on the zero-mean signal
    epsilon = 10 ** (-20 / 20)  # Adjusted small positive constant for stability
    max_magnitude = np.max(magnitude_spectrogram)
    normalized_spectrogram = (
        20 * np.log10(magnitude_spectrogram / (max_magnitude / 2) + epsilon) + 20
    ) / 20
    normalized_spectrogram = np.clip(normalized_spectrogram, 0, 1)

    # Create the figure and define a GridSpec layout
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 2, width_ratios=[20, 1], height_ratios=[1, 1, 1.2])

    # Plot 1: Original Spectrogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Original Spectrogram")
    img1 = ax1.pcolormesh(
        times, frequencies, magnitude_spectrogram, cmap="jet", shading="gouraud"
    )
    ax1.set_ylabel("Frequency [Hz]")
    ax1.grid(True)

    # Add colorbar for Plot 1
    cbar1 = fig.add_subplot(gs[0, 1])
    fig.colorbar(img1, cax=cbar1, label="Intensity [dB]")

    # Plot 2: Normalized Spectrogram (using zero-mean normalization)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("Normalized Spectrogram (Zero-Mean Adjusted)")
    img2 = ax2.pcolormesh(
        times, frequencies, normalized_spectrogram, cmap="jet", shading="gouraud"
    )
    ax2.set_ylabel("Frequency [Hz]")
    ax2.grid(True)

    # Add colorbar for Plot 2
    cbar2 = fig.add_subplot(gs[1, 1])
    fig.colorbar(img2, cax=cbar2, label="Intensity [dB]")

    # Plot 3: Acceleration over Time (Zero-Mean Adjusted)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_title("Acceleration over Time (Zero-Mean)")
    ax3.plot(total_time, zero_mean_signal, "g")
    ax3.set_xlabel("Time [sec]")
    ax3.set_ylabel("Acceleration [m/s^2]")
    ax3.grid(True)

    # Align x-axes across all subplots
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim([times.min(), times.max()])

    # Tight layout without overlapping
    plt.tight_layout()

    # Save to file if save_path is provided
    if save_path:
        plt.savefig(save_path, format="png")
        plt.close()
    else:
        plt.show()

    return fig


def preprocess_and_reduce(
    magnitude_spectrogram: np.ndarray,
    n_components: int = 10,
) -> np.ndarray:
    """
    Preprocess the magnitude spectrogram and reduce its dimensionality using PCA.
    This function takes a magnitude spectrogram as input, transposes it, and applies
    Principal Component Analysis (PCA) to reduce its dimensionality. The reduced features
    are then scaled using StandardScaler.
    Args:
        magnitude_spectrogram (numpy.ndarray): The input magnitude spectrogram.
        n_components (int, optional): The number of principal components to keep. Default is 10.

    Returns:
        numpy.ndarray: The scaled and reduced features.
    """
    spectrogram_2d = magnitude_spectrogram.T
    pca = PCA(n_components=n_components, random_state=42)
    reduced_features = pca.fit_transform(spectrogram_2d)
    scaler = StandardScaler()
    reduced_features_scaled = scaler.fit_transform(reduced_features)
    return reduced_features_scaled


def find_optimal_clusters(
    reduced_features_scaled: np.ndarray,
    max_k: int,
    save_path: str = None,
) -> int:
    """
    Find the optimal number of clusters using the elbow method.
    This function calculates the inertia scores for a range of cluster numbers
    and identifies the optimal number of clusters by finding the point where
    the second derivative of the inertia scores is maximized (elbow point).
    Args:
        reduced_features_scaled (numpy.ndarray): Scaled and reduced feature data.
        max_k (int): Maximum number of clusters to consider.
        save_path (str, optional): Path to save the plot.

    Returns:
        int: Optimal number of clusters.
    """

    num_clusters = list(range(2, max_k + 1))
    inertia_scores = []

    for n_clusters in num_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(reduced_features_scaled)
        inertia_scores.append(kmeans.inertia_)

    # Calculate the first and second derivatives
    first_derivative = np.diff(inertia_scores)
    second_derivative = np.diff(first_derivative)

    # The elbow point is where the second derivative is maximized
    best_num_clusters = num_clusters[np.argmax(second_derivative) + 2]

    # Plot the inertia scores
    plt.figure(figsize=(10, 6))
    plt.plot(num_clusters, inertia_scores)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia Score")
    plt.title("Inertia Score vs. Number of Clusters")
    plt.vlines(
        best_num_clusters,
        plt.ylim()[0],
        plt.ylim()[1],
        linestyles="dashed",
        colors="r",
        label="Optimal k",
    )
    plt.legend()

    # If save_path is provided, save the plot to the specified location
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

    logger.info(f"Optimal number of clusters: {best_num_clusters}")

    return best_num_clusters


def apply_MiniBatchKMeans(
    data: np.ndarray,
    n_clusters: int,
) -> Tuple[MiniBatchKMeans, np.ndarray]:
    """
    Apply MiniBatchKMeans clustering to the data.
    This function applies the MiniBatchKMeans clustering algorithm to the input data
    and returns the trained MiniBatchKMeans model along with the cluster labels for
    each data point.
    Args:
        data (numpy.ndarray): Input data to be clustered.
        n_clusters (int): Number of clusters to form.

    Returns:
        MiniBatchKMeans: Trained MiniBatchKMeans model.
        numpy.ndarray: Cluster labels for each data point.
    """
    mbkmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    labels = mbkmeans.fit_predict(data)
    return mbkmeans, labels


def identify_anomalies_kmeans(
    data: np.ndarray,
    mbkmeans: MiniBatchKMeans,
    percentile: float = 95,
) -> np.ndarray:
    """
    Identify anomalies using K-means clustering.
    This function identifies anomalies in the data by applying the MiniBatchKMeans
    clustering algorithm and calculating the distance of each data point to the nearest
    cluster center. Data points with distances greater than a specified threshold are
    considered anomalies.
    Args:
        data (numpy.ndarray): Input data to be clustered.
        mbkmeans (MiniBatchKMeans): Trained MiniBatchKMeans model.
        percentile (float, optional): The percentile of the distance distribution
            to use as the threshold. Default is 95.
    Returns:
        numpy.ndarray: Boolean array indicating anomalies.
    """
    distances = mbkmeans.transform(data).min(axis=1)
    threshold = np.percentile(distances, percentile)
    anomalies = distances > threshold
    return anomalies


def plot_clusters_and_anomalies_kmeans(
    x_axis: np.ndarray,
    data: np.ndarray,
    labels: np.ndarray,
    anomalies: np.ndarray,
    x_axis_label: str = "Time",
    save_path: str = None,
) -> plt.Figure:
    """
    Plot clustering results and anomalies, supporting time or kilometer as x-axis.
    Args:
        x_axis (numpy.ndarray): X-axis values (time or kilometer reference).
        data (numpy.ndarray): Input data (e.g., PCA reduced features).
        labels (numpy.ndarray): Cluster labels for each data point.
        anomalies (numpy.ndarray): Boolean array indicating anomalies.
        x_axis_label (str): Label for the x-axis (default: "Time").
        save_path (str, optional): Path to save the plot image.
    Returns:
        plt.Figure: The matplotlib figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = plt.scatter(x_axis, data[:, 0], c=labels, cmap="viridis", alpha=0.5)
    ax.scatter(
        x_axis[anomalies],
        data[anomalies, 0],
        color="red",
        marker="x",
        label="Anomalies",
    )
    fig.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel("Distance to Nearest Cluster Center")
    ax.set_title("K-means Clustering and Anomaly Detection")
    ax.legend()

    # Save to file if save_path is provided
    if save_path:
        plt.savefig(save_path, format="png")
        plt.close()
    else:
        plt.show()

    return fig


def identify_anomalies_distance(
    data: np.ndarray,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Identify anomalies based on distance from the mean spectrum.
    This function calculates the mean spectrum of the input data and then
    computes the Euclidean distance of each data point from this mean.
    Anomalies are identified as data points whose distance from the mean
    exceeds a threshold, defined as the mean distance plus two standard
    deviations.
    Args:
        data (numpy.ndarray): Input data, where each row represents a spectrum.

    Returns:
        tuple: A tuple containing three numpy.ndarray:
            - distances_from_mean: Array of distances from the mean spectrum.
            - threshold: The threshold value used for anomaly detection.
            - anomalies: Boolean array indicating which data points are anomalies.
    """
    mean_spectrum = np.mean(data, axis=0)
    distances_from_mean = np.linalg.norm(data - mean_spectrum, axis=0)
    threshold = np.mean(distances_from_mean) + 2 * np.std(distances_from_mean)
    anomalies = distances_from_mean > threshold

    return distances_from_mean, threshold, anomalies


def plot_clusters_and_anomalies_distance(
    x_axis: np.ndarray,
    anomalies: np.ndarray,
    distances_from_mean: np.ndarray,
    threshold: float,
    x_axis_label: str = "Time",
    save_path: str = None,
) -> plt.Figure:
    """
    Plot clustering results and anomalies based on distance from mean, with a flexible x-axis.

    Args:
        x_axis (numpy.ndarray): Array of values for the x-axis (e.g., Time or Kilometer Reference)
        anomalies (numpy.ndarray): Boolean array indicating anomalies
        distances_from_mean (numpy.ndarray): Array of distances from mean
        threshold (float): Threshold for anomaly detection. Calculated as mean + 2*std deviation
        x_axis_label (str, optional): Label for the x-axis (default is "Time")
        save_path (str, optional): Path to save the plot image. If None, the plot will not be saved.

    Returns:
        plt.Figure: The matplotlib figure object containing the plot.
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot distances and anomalies
    ax.plot(x_axis, distances_from_mean, label="Distance from Mean")
    ax.scatter(
        x_axis[anomalies],
        distances_from_mean[anomalies],
        c="red",
        marker="x",
        label="Anomalies",
    )

    # Plot threshold line
    ax.axhline(threshold, color="r", linestyle="--", label="Threshold")

    # Add labels and title
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel("Distance from Mean")
    ax.set_title("Distance from Mean and Anomalies")
    ax.legend()

    # Save to file if save_path is provided
    if save_path:
        plt.savefig(save_path, format="png")
        plt.close()
    else:
        plt.show()

    return fig


def save_anomalies_to_csv(
    anomalies: np.ndarray,
    times: np.ndarray,
    frequencies: np.ndarray,
    kilometer_ref_resampled: np.ndarray,
    filename: str,
) -> None:
    """
    Save identified anomalies to a CSV file.
    Args:
        anomalies (numpy.ndarray): Boolean array indicating anomalies
        times (numpy.ndarray): Array of time values
        frequencies (numpy.ndarray): Array of frequency values
        kilometer_ref_resampled (numpy.ndarray): Resampled kilometer reference values
        filename (str): Name of the CSV file to save the anomalies
    Returns:
        None
    """
    anomaly_indices = np.where(anomalies)[0]
    anomaly_times = times[anomaly_indices]
    anomaly_frequencies = frequencies[anomaly_indices % len(frequencies)]
    anomaly_kilometers = kilometer_ref_resampled[anomaly_indices]

    df = pd.DataFrame(
        {
            "Anomaly_Index": anomaly_indices,
            "Anomaly_Time": anomaly_times,
            "Kilometer_Ref_Aligned": anomaly_kilometers,
            "Anomaly_Frequency": anomaly_frequencies,
        }
    )
    df.to_csv(filename, index=False)
    logger.info(f"Anomalies saved to {filename}")


def load_anomalies(folder_path: str) -> dict:
    """
    Load all anomaly CSV files from the specified folder.
    Args:
        folder_path (str): Path to the folder containing anomaly CSV files.
    Returns:
        dict: Dictionary with dataframes for each CSV.
    """
    anomaly_data = {}

    # Check if folder exists to prevent FileNotFoundError
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder path {folder_path} does not exist.")

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            key = file.replace(".csv", "")  # Use the file name (minus extension) as key
            anomaly_data[key] = pd.read_csv(file_path)
    return anomaly_data


def plot_anomalies(
    anomaly_data: dict,
    x_axis: str = "Anomaly_Time",
    route_key: str = None,
) -> go.Figure:
    """
    Plot anomalies using Plotly as histograms and scatter plots with interactive controls for x-axis selection.
    Args:
        anomaly_data (dict): Dictionary with anomaly dataframes.
        x_axis (str): Column name for x-axis, either "Anomaly_Time" or "Kilometer_Ref_Aligned".
        route_key (str): Current route key to be used in the plot labels.
    Returns:
        go.Figure: Plotly figure with overlaid anomaly histogram and scatter plots.
    """
    fig = go.Figure()

    # Define colors for each trace
    colors = px.colors.qualitative.Plotly
    color_index = 0

    # Map acceleration codes to descriptive names
    acceleration_labels = {
        "acc_vert_left_axle_box_ms2": "Vertical Left Axle",
        "acc_vert_right_axle_box_ms2": "Vertical Right Axle",
        "acc_lat_axle_box_ms2": "Lateral Axle",
    }

    # Set bin size to 0.1 seconds for time or (1/10) km for kilometer
    bin_size = 0.1 if x_axis == "Anomaly_Time" else 1 / 10

    for key, data in anomaly_data.items():
        # Use route_key as the route name directly, if provided
        route = route_key if route_key else "_".join(key.split("_")[:2])

        # Extract acceleration and anomaly type from the filename
        parts = key.split("_")
        acceleration_code = "_".join(parts[2:-2])  # Acceleration code
        anomaly_type = parts[-1]  # Anomaly type

        # Map the acceleration code to its descriptive label
        acceleration = acceleration_labels.get(acceleration_code, acceleration_code)

        # Create a scatter plot trace for each (acceleration, anomaly type) combination
        fig.add_trace(
            go.Scatter(
                x=data[x_axis],
                y=data["Anomaly_Frequency"],
                mode="markers",  # "lines+markers" or "markers"
                # mode="markers",
                name=f"{route} - {acceleration} - {anomaly_type}",
                marker=dict(color=colors[color_index % len(colors)]),
            )
        )

        # Create histogram for each (acceleration, anomaly type) combination
        fig.add_trace(
            go.Histogram(
                x=data[x_axis],
                y=data["Anomaly_Frequency"],
                name=f"{route} - {acceleration} - {anomaly_type}",
                marker=dict(color=colors[color_index % len(colors)]),
                opacity=0.6,
                xbins=dict(
                    start=data[x_axis].min(),
                    end=data[x_axis].max(),
                    size=bin_size,
                ),
            )
        )
        color_index += 1

    # Set up layout
    fig.update_layout(
        title="Anomaly Frequency for Different Accelerations and Anomaly Types",
        xaxis_title=x_axis,
        yaxis_title="Anomaly Frequency",
        barmode="overlay",  # Overlay histograms for comparison
        legend_title="Route - Acceleration - Anomaly Type",
        # template="plotly_dark",
    )

    return fig


def plot_anomalies_streamlit(
    anomaly_data_kmeans: pd.DataFrame,
    anomaly_data_distance: pd.DataFrame,
    x_axis: str = "Anomaly_Time",
) -> go.Figure:
    """
    Plot anomalies using Plotly as histograms and scatter plots with interactive controls for x-axis selection.
    """
    fig = go.Figure()

    # Define colors for differentiation
    colors = px.colors.qualitative.Plotly
    color_index = 0

    # Labels for different accelerations
    acceleration_labels = {
        "acc_vert_left_axle_box_ms2": "Vertical Left Axle",
        "acc_vert_right_axle_box_ms2": "Vertical Right Axle",
        "acc_lat_axle_box_ms2": "Lateral Axle",
    }

    # Plot data for KMeans and Distance-Based detection
    for method, data in [
        ("KMeans", anomaly_data_kmeans),
        ("Distance-Based", anomaly_data_distance),
    ]:
        for accel_col, label in acceleration_labels.items():
            if accel_col in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data[x_axis],
                        y=data[accel_col],
                        mode="markers",
                        name=f"{method} - {label}",
                        marker=dict(color=colors[color_index % len(colors)]),
                    )
                )
                fig.add_trace(
                    go.Histogram(
                        x=data[x_axis],
                        y=data[accel_col],
                        name=f"{method} - {label} (Histogram)",
                        marker=dict(color=colors[color_index % len(colors)]),
                        opacity=0.6,
                    )
                )
                color_index += 1

    # Update layout
    fig.update_layout(
        title="Anomaly Visualization for KMeans and Distance-Based Methods",
        xaxis_title=x_axis,
        yaxis_title="Anomaly Frequency",
        barmode="overlay",
        legend_title="Method - Acceleration",
    )

    return fig


# #! ---- NEW FUNCTIONS - TBD ----
# def prepare_dataset_for_training(
#     data_path: str, items: List[str], input_shape: Tuple[int, int, int]
# ) -> RailwayDataset:
#     dataset = RailwayDataset(
#         dir_dataset=data_path, items=items, input_shape=input_shape
#     )
#     dataset.label_cruces_adif()
#     return dataset


# def create_dataloader(
#     dataset: RailwayDataset, batch_size: int, shuffle: bool = True
# ) -> DataLoader:
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# def initialize_model(
#     input_channels: int,
#     n_blocks: int,
#     projection_dim: int,
#     n_classes: int,
#     pretrained: bool = False,
# ) -> PrototypicalNetwork:
#     encoder = ResNetEncoder(
#         in_channels=input_channels, n_blocks=n_blocks, pretrained=pretrained
#     )
#     model = PrototypicalNetwork(
#         encoder=encoder,
#         projection_dim=projection_dim,
#         n_classes=n_classes,
#         contrastive_loss=True,
#     )
#     return model


# def run_training(
#     model: PrototypicalNetwork,
#     train_loader: DataLoader,
#     epochs: int,
#     learning_rate: float,
#     device: torch.device,
# ) -> Tuple[List[float], List[float], List[float], List[float]]:
#     return train_prototypical_network(
#         model, train_loader, epochs, learning_rate, device
#     )


# def run_evaluation(
#     model: PrototypicalNetwork, test_loader: DataLoader, device: torch.device
# ) -> Tuple[float, float, float, float, np.ndarray]:
#     return evaluate_model(model, test_loader, device)
