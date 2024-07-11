#!##########################################
#!############# IMPORTS ####################
#!##########################################

# Standard Library Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, stft
from scipy.signal.windows import hamming
from typing import Tuple, Optional, NoReturn, List

# New imports (test supervised contrastive learning)
import torch
from torch.utils.data import DataLoader

# Additional imports for supervised contrastive learning
from .anomaly_detection.REFACTORED_supervised_contrastive_learning import (
    RailwayDataset,
    ResNetEncoder,
    PrototypicalNetwork,
    train_prototypical_network,
    evaluate_model,
)


# Local Imports
from .logs import logger

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
    key_f20_10: Optional[str] = "f20_10",
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.DataFrame]:
    """
    Load and preprocess the data.
    Args:
        data_path (str): Path to the data file.
        acel_to_process (str): Acceleration data to process.
        key_f20_10 (str): Key for the .mat file, to identify the f20_10 data.
    Returns:
        data (numpy.ndarray): Preprocessed data.
        signal (numpy.ndarray): Signal to process.
        time_column (pd.Series): Time column.
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

        signal = df[acel_to_process]
        time_column = df[time_col_name]

        return data, signal, time_column, df

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Short-Term Fourier Transform (STFT) of a signal.
    Args:
        signal (numpy.ndarray): Input signal.
        sampling_frequency_stft (float): Sampling frequency in Hz.
        window_length (float): Window length in seconds.
        overlap (float): Overlap fraction.
        gamma (float): Dynamic margin.
        time_column (pandas.Series): Time column.
    Returns:
        frequencies (numpy.ndarray): Frequency vector.
        times (numpy.ndarray): Time vector.
        magnitude_spectrogram (numpy.ndarray): Magnitude spectrogram.
        X_prime (numpy.ndarray): Normalized spectrogram.
        total_time (numpy.ndarray): Total time vector.
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
    frequencies, times, Zxx = stft(
        signal,
        fs=sampling_frequency_stft,
        window=window,
        nperseg=window_samples,
        noverlap=noverlap,
    )
    times += start_time  # Adjust the time vector of the STFT output

    # Convert to magnitude spectrogram
    magnitude_spectrogram = np.abs(Zxx)

    # Dynamic-margin normalization
    epsilon = 10 ** (-gamma / 20)  # Dynamic margin. Default is 20 dB
    X_prime = (
        20
        * np.log10(
            magnitude_spectrogram / (np.max(magnitude_spectrogram) / 2) + epsilon
        )
        + gamma
    )
    X_prime = np.clip(X_prime / gamma, 0, 1)

    return frequencies, times, magnitude_spectrogram, X_prime, total_time


def plot_stft_results(
    frequencies: np.ndarray,
    times: np.ndarray,
    magnitude_spectrogram: np.ndarray,
    X_prime: np.ndarray,
    total_time: np.ndarray,
    signal: np.ndarray,
) -> NoReturn:
    """
    Plot the results of the STFT analysis.
    Args:
        frequencies (numpy.ndarray): Frequency vector.
        times (numpy.ndarray): Time vector.
        magnitude_spectrogram (numpy.ndarray): Magnitude spectrogram.
        X_prime (numpy.ndarray): Normalized spectrogram.
        total_time (numpy.ndarray): Total time vector.
        signal (numpy.ndarray): Input signal.
    Returns:
        None
    """

    # Plotting the results
    fig, axs = plt.subplots(
        3, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 1, 1.2]}
    )

    # Plot 1: Original Spectrogram
    axs[0].set_title("Original Spectrogram")
    img1 = axs[0].pcolormesh(
        times, frequencies, magnitude_spectrogram, cmap="jet", shading="gouraud"
    )
    axs[0].set_ylabel("Frequency [Hz]")
    axs[0].set_xlabel("Time [sec]")
    fig.colorbar(img1, ax=axs[0], label="Intensity [dB]")
    # axs[0].set_ylim([0, 1600])
    axs[0].grid(True)

    # Plot 2: Normalized Spectrogram
    axs[1].set_title("Normalized Spectrogram")
    img2 = axs[1].pcolormesh(times, frequencies, X_prime, cmap="jet", shading="gouraud")
    axs[1].set_ylabel("Frequency [Hz]")
    axs[1].set_xlabel("Time [sec]")
    fig.colorbar(img2, ax=axs[1], label="Intensity [dB]")
    # axs[1].set_ylim([0, 1600])
    axs[1].grid(True)

    # Plot 3: Acceleration over Time
    axs[2].set_title("Acceleration over Time")
    axs[2].plot(total_time, signal, "g")
    axs[2].set_xlabel("Time [sec]")
    axs[2].set_ylabel("Acceleration [m/s^2]")
    axs[2].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()


#! ---- NEW FUNCTIONS - TBD ----
def prepare_dataset_for_training(
    data_path: str, items: List[str], input_shape: Tuple[int, int, int]
) -> RailwayDataset:
    dataset = RailwayDataset(
        dir_dataset=data_path, items=items, input_shape=input_shape
    )
    dataset.label_cruces_adif()
    return dataset


def create_dataloader(
    dataset: RailwayDataset, batch_size: int, shuffle: bool = True
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def initialize_model(
    input_channels: int,
    n_blocks: int,
    projection_dim: int,
    n_classes: int,
    pretrained: bool = False,
) -> PrototypicalNetwork:
    encoder = ResNetEncoder(
        in_channels=input_channels, n_blocks=n_blocks, pretrained=pretrained
    )
    model = PrototypicalNetwork(
        encoder=encoder,
        projection_dim=projection_dim,
        n_classes=n_classes,
        contrastive_loss=True,
    )
    return model


def run_training(
    model: PrototypicalNetwork,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    return train_prototypical_network(
        model, train_loader, epochs, learning_rate, device
    )


def run_evaluation(
    model: PrototypicalNetwork, test_loader: DataLoader, device: torch.device
) -> Tuple[float, float, float, float, np.ndarray]:
    return evaluate_model(model, test_loader, device)
