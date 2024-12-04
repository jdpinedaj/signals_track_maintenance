#  poetry run streamlit run tpp-text2sql_app.py

#! Imports
import streamlit as st
import os
import numpy as np
import pandas as pd
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
)
from src.logs import logger


#! Main function
def main():

    # App Title
    st.title("Anomaly Detection for Railway Track Data")
    st.image(image="images/upv-logo.png")
    st.divider()

    # Sidebar: File Upload
    uploaded_file = st.file_uploader("Upload .mat file", type=["mat"])
    accel_options = {
        "Vertical Left Axle": "acc_vert_left_axle_box_ms2",
        "Vertical Right Axle": "acc_vert_right_axle_box_ms2",
        "Lateral Axle": "acc_lat_axle_box_ms2",
    }
    selected_accel = st.selectbox(
        "Select Acceleration to Process", list(accel_options.keys())
    )

    # Process Button
    if uploaded_file and st.button("Process and Analyze"):
        try:
            # Set acceleration key dynamically
            accel_key = accel_options[selected_accel]

            # Step 1: Preprocess Data
            st.subheader("Preprocessing Data")
            data, signal, time_column, kilometer_ref, df = preprocess_mat_data(
                data_path=uploaded_file,
                acel_to_process=accel_key,
                time_col_name="timestamp_s",
                km_ref_col_name="kilometer_ref_fixed_km",
            )
            st.write("Data Loaded and Preprocessed")
            st.dataframe(df.head())

            # Step 2: Compute STFT
            st.subheader("Short-Term Fourier Transform (STFT)")
            (
                frequencies,
                times,
                magnitude_spectrogram,
                X_prime,
                total_time,
                kilometer_ref_resampled,
            ) = short_term_fourier_transform_stft(
                signal=signal,
                sampling_frequency_stft=100,  # Use pre-defined config values
                window_length=0.25,
                overlap=0.95,
                gamma=20,
                time_column=time_column,
                kilometer_ref=kilometer_ref,
                nfft=128,
            )
            st.write("STFT Computed")
            fig_stft = plot_stft_results(
                frequencies,
                times,
                magnitude_spectrogram,
                X_prime,
                total_time,
                signal,
            )
            st.pyplot(fig_stft)
            logger.info("STFT results plotted")

            # Step 3: Dimensionality Reduction
            reduced_features_scaled = preprocess_and_reduce(
                magnitude_spectrogram, n_components=10
            )

            # Step 4: KMeans Clustering
            st.subheader("KMeans Clustering and Anomaly Detection")
            optimal_k = find_optimal_clusters(reduced_features_scaled, max_k=10)
            mbkmeans, labels = apply_MiniBatchKMeans(reduced_features_scaled, optimal_k)
            anomalies_kmeans = identify_anomalies_kmeans(
                reduced_features_scaled, mbkmeans, percentile=98
            )
            fig_kmeans = plot_clusters_and_anomalies_kmeans(
                x_axis=kilometer_ref_resampled,
                data=reduced_features_scaled,
                labels=labels,
                anomalies=anomalies_kmeans,
                x_axis_label="Kilometers",
            )
            st.pyplot(fig_kmeans)
            logger.info("KMeans results plotted")

            # Display anomalies table for KMeans
            st.subheader("KMeans Anomalies Table")
            anomaly_indices_kmeans = np.where(anomalies_kmeans)[0]
            kmeans_anomalies_df = pd.DataFrame(
                {
                    "Anomaly_Index": anomaly_indices_kmeans,
                    "Anomaly_Kilometer": kilometer_ref_resampled[
                        anomaly_indices_kmeans
                    ],
                }
            )
            st.dataframe(kmeans_anomalies_df)

            # Step 5: Distance-Based Anomaly Detection
            st.subheader("Distance-Based Anomaly Detection")
            distances_from_mean, threshold, anomalies_distance = (
                identify_anomalies_distance(magnitude_spectrogram)
            )
            fig_distance = plot_clusters_and_anomalies_distance(
                x_axis=kilometer_ref_resampled,
                anomalies=anomalies_distance,
                distances_from_mean=distances_from_mean,
                threshold=threshold,
                x_axis_label="Kilometers",
            )
            st.pyplot(fig_distance)
            logger.info("Distance-Based results plotted")

            # Display anomalies table for Distance-Based Method
            st.subheader("Distance-Based Anomalies Table")
            anomaly_indices_distance = np.where(anomalies_distance)[0]
            distance_anomalies_df = pd.DataFrame(
                {
                    "Anomaly_Index": anomaly_indices_distance,
                    "Anomaly_Kilometer": kilometer_ref_resampled[
                        anomaly_indices_distance
                    ],
                }
            )
            st.dataframe(distance_anomalies_df)

            # st.success("Processing Completed Successfully!")

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
