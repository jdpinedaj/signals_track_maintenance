#  poetry run python src/main.py --file_key RA_AP --accel_key lateral_axle


#!##########################################
#!############# IMPORTS ####################
#!##########################################

import argparse
import os
import sys
import numpy as np

# Add project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from src.utils import (
    preprocess_mat_data,
    short_term_fourier_transform_stft,
    plot_stft_results,
    preprocess_and_reduce,
    find_optimal_clusters,
    apply_MiniBatchKMeans,
    identify_anomalies_kmeans,
    plot_clusters_and_anomalies_kmeans,
    identify_anomalies_distance,
    plot_clusters_and_anomalies_distance,
    save_anomalies_to_csv,
)
from src.logs import logger
from src.load_config import LoadConfig


#!##########################################
#!############# FUNCTIONS ##################
#!##########################################


def process_data_and_analyze(file_key: str, accel_key: str) -> None:
    # Load the configuration
    APPCFG = LoadConfig()

    # Set the file and acceleration keys dynamically
    APPCFG.mat_data = os.path.join(
        APPCFG.data_path, APPCFG.data_to_analyze["data_files"][file_key]
    )
    APPCFG.acceleration_to_analyze = APPCFG.data_to_analyze["accelerations"][accel_key]

    # Update anomalies path to include the specific file and acceleration key
    APPCFG.anomalies_path = os.path.join(
        APPCFG.anomalies_base_path, file_key  # , accel_key
    )
    os.makedirs(APPCFG.anomalies_path, exist_ok=True)

    logger.info(f"Processing file: {APPCFG.mat_data}")
    logger.info(f"Using acceleration: {APPCFG.acceleration_to_analyze}")

    # Step 1: Load and preprocess data
    (
        data_f20_10,
        signal_acc_mat,
        time_column_mat,
        kilometer_ref,
        df_mat,
    ) = preprocess_mat_data(
        data_path=APPCFG.mat_data,
        acel_to_process=APPCFG.acceleration_to_analyze,
        time_col_name="timestamp_s",
        km_ref_col_name="kilometer_ref_fixed_km",
    )

    # Step 2: Apply STFT
    (
        frequencies_acc,
        times_acc,
        magnitude_spectrogram_acc,
        X_prime_acc,
        total_time_acc,
    ) = short_term_fourier_transform_stft(
        signal=signal_acc_mat,
        sampling_frequency_stft=APPCFG.sampling_frequency_stft_prepared,
        window_length=APPCFG.window_length,
        overlap=APPCFG.overlap,
        gamma=APPCFG.gamma,
        time_column=time_column_mat,
        nfft=APPCFG.nfft_prepared,
    )
    plot_stft_results(
        frequencies_acc,
        times_acc,
        magnitude_spectrogram_acc,
        X_prime_acc,
        total_time_acc,
        signal_acc_mat,
        save_path=APPCFG.get_anomalies_filename("stft", file_extension="png"),
    )

    # Step 3: Dimensionality reduction
    reduced_features_scaled = preprocess_and_reduce(
        magnitude_spectrogram_acc, n_components=10
    )

    # Step 4: Find optimal clusters and apply MiniBatchKMeans
    max_k = 10
    optimal_k = find_optimal_clusters(
        reduced_features_scaled,
        max_k,
        save_path=APPCFG.get_anomalies_filename(
            "optimal_clusters", file_extension="png"
        ),
    )
    mbkmeans, labels = apply_MiniBatchKMeans(reduced_features_scaled, optimal_k)

    # Step 5: Detect anomalies using K-means
    anomalies_kmeans = identify_anomalies_kmeans(
        reduced_features_scaled, mbkmeans, percentile=APPCFG.percentile_kmeans
    )
    plot_clusters_and_anomalies_kmeans(
        times_acc,
        reduced_features_scaled,
        labels,
        anomalies_kmeans,
        save_path=APPCFG.get_anomalies_filename("kmeans", file_extension="png"),
    )
    save_anomalies_to_csv(
        anomalies_kmeans,
        times_acc,
        frequencies_acc,
        kilometer_ref,
        APPCFG.get_anomalies_filename("kmeans"),
    )

    # Step 6: Detect anomalies using distance from mean
    distances_from_mean, threshold, anomalies_distance = identify_anomalies_distance(
        magnitude_spectrogram_acc
    )
    plot_clusters_and_anomalies_distance(
        times_acc,
        anomalies_distance,
        distances_from_mean,
        threshold,
        save_path=APPCFG.get_anomalies_filename("distance", file_extension="png"),
    )
    save_anomalies_to_csv(
        anomalies_distance,
        times_acc,
        frequencies_acc,
        kilometer_ref,
        APPCFG.get_anomalies_filename("distance"),
    )

    # Step 7: Print a summary of the results
    logger.info(f"Total data points: {len(reduced_features_scaled)}")
    logger.info(
        f"Anomalies by K-means: {np.sum(anomalies_kmeans)}, {np.round(np.sum(anomalies_kmeans)/len(times_acc)*100, 2)}%"
    )
    logger.info(
        f"K-means time range: {times_acc[np.where(anomalies_kmeans)[0]].min()} to {times_acc[np.where(anomalies_kmeans)[0]].max()}"
    )
    logger.info(
        f"Anomalies by distance: {np.sum(anomalies_distance)}, {np.round(np.sum(anomalies_distance)/len(times_acc)*100, 2)}%"
    )
    logger.info(
        f"Distance time range: {times_acc[np.where(anomalies_distance)[0]].min()} to {times_acc[np.where(anomalies_distance)[0]].max()}"
    )


#!##########################################
#!################ MAIN ####################
#!##########################################


def main():
    """
    Main function to run the data processing and analysis.
    It can be executed like this:
        poetry run python src/main.py --file_key RA_AP --accel_key lateral_axle
    """
    parser = argparse.ArgumentParser(description="Run data processing and analysis.")
    parser.add_argument(
        "--file_key", type=str, help="Key for the data file to process (e.g., 'RA_AP')."
    )
    parser.add_argument(
        "--accel_key",
        type=str,
        help="Key for the acceleration type to analyze (e.g., 'lateral_axle').",
    )
    args = parser.parse_args()

    if args.file_key and args.accel_key:
        # Run with specified arguments
        process_data_and_analyze(args.file_key, args.accel_key)
    else:
        # Loop through all combinations
        data_files = ["RA_AP", "PX_RA", "RA_AD"]  # Update this list as needed
        accelerations = ["vertical_left_axle", "vertical_right_axle", "lateral_axle"]

        for file_key in data_files:
            for accel_key in accelerations:
                process_data_and_analyze(file_key, accel_key)


if __name__ == "__main__":
    main()
