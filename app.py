import streamlit as st
import os
import numpy as np
import pandas as pd
from src.utils import (
    preprocess_mat_data,
    short_term_fourier_transform_stft,
    plot_stft_results_with_zero_mean,
    preprocess_and_reduce,
    find_optimal_clusters,
    apply_MiniBatchKMeans,
    identify_anomalies_kmeans,
    plot_clusters_and_anomalies_kmeans,
    identify_anomalies_distance,
    plot_clusters_and_anomalies_distance,
    plot_anomalies_streamlit,
)
from src.logs import logger

from src.load_config import LoadConfig

# Load configuration
APPCFG = LoadConfig()


def main():
    st.title("Anomaly Detection for Railway Track Data")
    st.image("images/upv-logo.png")
    st.divider()

    # Sidebar for File Upload and Acceleration Selection
    uploaded_file = st.sidebar.file_uploader("Upload .mat file", type=["mat"])
    accel_options = {
        "Vertical Left Axle": "acc_vert_left_axle_box_ms2",
        "Vertical Right Axle": "acc_vert_right_axle_box_ms2",
        "Lateral Axle": "acc_lat_axle_box_ms2",
    }
    selected_accel = st.sidebar.selectbox(
        "Select Acceleration to Process", list(accel_options.keys())
    )
    accel_key = accel_options[selected_accel]

    # File upload validation
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.success("File uploaded successfully!")
    elif "uploaded_file" in st.session_state:
        uploaded_file = st.session_state.uploaded_file
    else:
        st.warning("Please upload a .mat file to proceed.")
        return

    # Data Preprocessing
    try:
        (
            data,
            signal_acc_mat,
            time_column_mat,
            kilometer_ref,
            df_mat,
        ) = preprocess_mat_data(
            data_path=uploaded_file,
            acel_to_process=accel_key,
            time_col_name="timestamp_s",
            km_ref_col_name="kilometer_ref_fixed_km",
        )
        st.session_state.data = {
            "signal": signal_acc_mat,
            "time_column": time_column_mat,
            "kilometer_ref": kilometer_ref,
            "df": df_mat,
        }
        st.session_state.signal = signal_acc_mat
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return

    # Tabs for Visualizations
    (
        tab1,
        tab2,
        tab3,
        tab4,
        # tab5,
    ) = st.tabs(
        [
            "Raw Data",
            "STFT Analysis",
            "KMeans Clustering",
            "Distance-Based Detection",
            # "Anomaly Visualization",
        ]
    )

    # Tab 1: Raw Data
    with tab1:
        st.subheader("Raw Data Visualization")
        if "data" in st.session_state:
            st.dataframe(st.session_state.data["df"].head())
        else:
            st.warning("Data not available. Please process the uploaded file.")

    # Tab 2: STFT Analysis
    with tab2:
        st.subheader("Short-Term Fourier Transform (STFT)")
        if "signal" in st.session_state:
            try:
                (
                    frequencies_acc,
                    times_acc,
                    magnitude_spectrogram_acc,
                    X_prime_acc,
                    total_time_acc,
                    kilometer_ref_resampled,
                ) = short_term_fourier_transform_stft(
                    signal=st.session_state.signal,
                    sampling_frequency_stft=APPCFG.sampling_frequency_stft_prepared,
                    window_length=APPCFG.window_length,
                    overlap=APPCFG.overlap,
                    gamma=APPCFG.gamma,
                    time_column=st.session_state.data["time_column"],
                    kilometer_ref=st.session_state.data["kilometer_ref"],
                    nfft=128,
                )
                fig_stft = plot_stft_results_with_zero_mean(
                    frequencies_acc,
                    times_acc,
                    magnitude_spectrogram_acc,
                    X_prime_acc,
                    total_time_acc,
                    st.session_state.signal,
                )
                st.pyplot(fig_stft)
                st.session_state.stft_data = {
                    "magnitude_spectrogram": magnitude_spectrogram_acc,
                    "kilometer_ref_resampled": kilometer_ref_resampled,
                }
            except Exception as e:
                st.error(f"Error during STFT analysis: {e}")
        else:
            st.warning("Signal data not available.")

    # Tab 3: KMeans Clustering
    with tab3:
        st.subheader("KMeans Clustering and Anomaly Detection")
        if "stft_data" in st.session_state:
            try:
                stft_data = st.session_state.stft_data
                reduced_features_scaled = preprocess_and_reduce(
                    stft_data["magnitude_spectrogram"], n_components=10
                )
                optimal_k = find_optimal_clusters(reduced_features_scaled, max_k=10)
                mbkmeans, labels = apply_MiniBatchKMeans(
                    reduced_features_scaled, optimal_k
                )
                anomalies_kmeans = identify_anomalies_kmeans(
                    reduced_features_scaled, mbkmeans, percentile=98
                )
                anomaly_indices = np.where(anomalies_kmeans)[
                    0
                ]  # Get indices of anomalies

                fig_kmeans = plot_clusters_and_anomalies_kmeans(
                    x_axis=stft_data["kilometer_ref_resampled"],
                    data=reduced_features_scaled,
                    labels=labels,
                    anomalies=anomalies_kmeans,
                    x_axis_label="Kilometers",
                )
                st.pyplot(fig_kmeans)

                # Save KMeans anomalies for visualization
                st.session_state.kmeans_anomalies_df = pd.DataFrame(
                    {
                        "Anomaly_Time": st.session_state.data["time_column"]
                        .iloc[anomaly_indices]
                        .values,
                        "Kilometer_Ref_Aligned": stft_data["kilometer_ref_resampled"][
                            anomaly_indices
                        ],
                        "acc_vert_left_axle_box_ms2": reduced_features_scaled[
                            anomaly_indices, 0
                        ],
                        "acc_vert_right_axle_box_ms2": reduced_features_scaled[
                            anomaly_indices, 1
                        ],
                        "acc_lat_axle_box_ms2": reduced_features_scaled[
                            anomaly_indices, 2
                        ],
                    }
                )
            except Exception as e:
                st.error(f"Error during KMeans clustering: {e}")
        else:
            st.warning("STFT data not available.")

    # Tab 4: Distance-Based Detection
    with tab4:
        st.subheader("Distance-Based Anomaly Detection")
        if "stft_data" in st.session_state:
            try:
                stft_data = st.session_state.stft_data
                distances_from_mean, threshold, anomalies_distance = (
                    identify_anomalies_distance(stft_data["magnitude_spectrogram"])
                )
                anomaly_indices = np.where(anomalies_distance)[
                    0
                ]  # Get indices of anomalies

                fig_distance = plot_clusters_and_anomalies_distance(
                    x_axis=stft_data["kilometer_ref_resampled"],
                    anomalies=anomalies_distance,
                    distances_from_mean=distances_from_mean,
                    threshold=threshold,
                    x_axis_label="Kilometers",
                )
                st.pyplot(fig_distance)

                # Save Distance-Based anomalies for visualization
                st.session_state.distance_anomalies_df = pd.DataFrame(
                    {
                        "Anomaly_Time": st.session_state.data["time_column"]
                        .iloc[anomaly_indices]
                        .values,
                        "Kilometer_Ref_Aligned": stft_data["kilometer_ref_resampled"][
                            anomaly_indices
                        ],
                        "acc_vert_left_axle_box_ms2": distances_from_mean[
                            anomaly_indices
                        ],
                        "acc_vert_right_axle_box_ms2": distances_from_mean[
                            anomaly_indices
                        ],
                        "acc_lat_axle_box_ms2": distances_from_mean[anomaly_indices],
                    }
                )

            except Exception as e:
                st.error(f"Error during Distance-Based detection: {e}")
        else:
            st.warning("STFT data not available.")

    # TODO: Add anomaly visualization
    # # Tab 5: Anomaly Visualization
    # with tab5:
    #     st.subheader("Anomaly Visualization")
    #     x_axis_option = st.radio(
    #         "Select X-Axis", ["Anomaly_Time", "Kilometer_Ref_Aligned"]
    #     )

    #     # Check if KMeans and Distance-Based results are available
    #     if (
    #         "kmeans_anomalies_df" in st.session_state
    #         and "distance_anomalies_df" in st.session_state
    #     ):
    #         anomaly_data_kmeans = st.session_state.kmeans_anomalies_df
    #         anomaly_data_distance = st.session_state.distance_anomalies_df

    #         # Generate plot
    #         fig = plot_anomalies_streamlit(
    #             anomaly_data_kmeans=anomaly_data_kmeans,
    #             anomaly_data_distance=anomaly_data_distance,
    #             x_axis=x_axis_option,
    #         )
    #         st.plotly_chart(fig, use_container_width=True)
    #     else:
    #         st.warning(
    #             "Anomaly data not available. Please process KMeans and Distance-Based Detection first."
    #         )


if __name__ == "__main__":
    main()
