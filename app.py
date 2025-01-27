# poetry run streamlit run app.py


import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from src.utils import (
    _downsample_signal,
    cached_preprocess_mat_data,
    cached_stft_analysis_optimized,
    plot_stft_results_with_zero_mean,
    cached_preprocess_and_reduce,
    find_optimal_clusters,
    apply_MiniBatchKMeans,
    identify_anomalies_kmeans,
    plot_clusters_and_anomalies_kmeans,
    identify_anomalies_distance,
    plot_clusters_and_anomalies_distance,
    plot_anomalies_streamlit,
)
from src.load_config import LoadConfig

# Load configuration
APPCFG = LoadConfig()


def main():
    # Load the image and resize it with a fixed height
    original_image = Image.open("images/upv-logo.png")
    fixed_height = 100
    width_ratio = fixed_height / original_image.height
    new_width = int(original_image.width * width_ratio)
    resized_image = original_image.resize((new_width, fixed_height))
    st.image(resized_image, use_container_width=False)
    st.title("Anomaly Detection for Railway Track Data")
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

    # Automatically clear cache when new file or acceleration is selected
    if (
        "uploaded_file" in st.session_state
        and uploaded_file != st.session_state.get("uploaded_file")
    ) or (
        "selected_accel" in st.session_state
        and accel_key != st.session_state.get("selected_accel")
    ):
        st.cache_data.clear()
        st.session_state.uploaded_file = uploaded_file
        st.session_state.selected_accel = accel_key
        st.success("Cache cleared due to new selection!")

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
            signal_acc_mat,
            time_column_mat,
            kilometer_ref,
            df_mat,
        ) = cached_preprocess_mat_data(
            file=uploaded_file,
            accel_key=accel_key,
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
    tabs = st.tabs(
        [
            "Raw Data",
            "STFT Analysis",
            "KMeans Clustering",
            "Distance-Based Detection",
            # "Anomaly Visualization",
        ]
    )

    # Tab 1: Raw Data
    with tabs[0]:
        st.subheader("Raw Data Visualization")
        st.dataframe(st.session_state.data["df"])

    # Tab 2: STFT Analysis
    with tabs[1]:
        st.subheader(
            "Short-Term Fourier Transform (STFT) - DOWNSAMPLED BY a factor of 10"
        )
        if "signal" in st.session_state:
            try:
                # Compute STFT
                (
                    frequencies_acc,
                    times_acc,
                    magnitude_spectrogram_acc,
                    total_time_acc,
                    kilometer_ref_resampled,
                ) = cached_stft_analysis_optimized(
                    signal=st.session_state.signal,
                    time_column=st.session_state.data["time_column"],
                    kilometer_ref=st.session_state.data["kilometer_ref"],
                )

                st.session_state.stft_data = {
                    "frequencies": frequencies_acc,
                    "magnitude_spectrogram": magnitude_spectrogram_acc,
                    "kilometer_ref_resampled": kilometer_ref_resampled,
                }

                # Plot STFT using aligned time and downsampled signal
                downsampled_signal, _, _ = _downsample_signal(
                    st.session_state.signal,
                    st.session_state.data["time_column"],
                    st.session_state.data["kilometer_ref"],
                )

                # st.write(f"Downsampled signal shape: {downsampled_signal.shape}")
                # st.write(f"Frequencies shape: {frequencies_acc.shape}")
                # st.write(f"Times shape: {times_acc.shape}")
                # st.write(f"Spectrogram shape: {magnitude_spectrogram_acc.shape}")

                # Plot STFT
                fig_stft = plot_stft_results_with_zero_mean(
                    frequencies=frequencies_acc,
                    times=times_acc,
                    magnitude_spectrogram=magnitude_spectrogram_acc,
                    total_time=total_time_acc,
                    signal=downsampled_signal,
                )
                st.pyplot(fig_stft)

            except Exception as e:
                st.error(f"Error during STFT analysis: {e}")
        else:
            st.warning("Signal data not available.")

    # Tab 3: KMeans Clustering
    with tabs[2]:
        st.subheader("KMeans Clustering and Anomaly Detection")
        if "stft_data" in st.session_state:
            try:
                stft_data = st.session_state.stft_data

                # Perform dimensionality reduction
                reduced_features_scaled = cached_preprocess_and_reduce(
                    stft_data["magnitude_spectrogram"]
                )

                # Find optimal clusters
                optimal_k = find_optimal_clusters(reduced_features_scaled, max_k=7)

                # Apply MiniBatchKMeans
                mbkmeans, labels = apply_MiniBatchKMeans(
                    reduced_features_scaled, optimal_k
                )

                # Identify anomalies using KMeans
                anomalies_kmeans = identify_anomalies_kmeans(
                    reduced_features_scaled,
                    mbkmeans,
                    percentile=APPCFG.percentile_kmeans,
                )

                # Plot KMeans clusters and anomalies
                fig_kmeans = plot_clusters_and_anomalies_kmeans(
                    x_axis=stft_data["kilometer_ref_resampled"],
                    data=reduced_features_scaled,
                    labels=labels,
                    anomalies=anomalies_kmeans,
                    x_axis_label="Mileage traveled [km]",
                )
                st.pyplot(fig_kmeans)

                # Create DataFrame for KMeans anomalies
                anomaly_indices_kmeans = np.where(anomalies_kmeans)[0]
                anomaly_kilometers = stft_data["kilometer_ref_resampled"][
                    anomaly_indices_kmeans
                ]
                anomaly_frequencies = stft_data["frequencies"][
                    anomaly_indices_kmeans % len(stft_data["frequencies"])
                ]
                anomaly_times = times_acc[anomaly_indices_kmeans]

                # Create DataFrame for KMeans anomalies
                kmeans_anomalies_df = pd.DataFrame(
                    {
                        "Anomaly_Index": anomaly_indices_kmeans,
                        "Anomaly_Time": anomaly_times,
                        "Kilometer_Ref_Aligned": anomaly_kilometers,
                        "Anomaly_Frequency": anomaly_frequencies,
                    }
                )

                # Display the DataFrame and provide a download button
                st.dataframe(kmeans_anomalies_df)
                st.download_button(
                    label="Download KMeans Anomalies as CSV",
                    data=kmeans_anomalies_df.to_csv(index=False),
                    file_name="kmeans_anomalies.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Error during KMeans clustering: {e}")
        else:
            st.warning("STFT data not available.")

    # Tab 4: Distance-Based Detection
    with tabs[3]:
        st.subheader("Distance-Based Anomaly Detection")
        if "stft_data" in st.session_state:
            try:
                stft_data = st.session_state.stft_data

                # Identify anomalies based on distance
                distances_from_mean, threshold, anomalies_distance = (
                    identify_anomalies_distance(stft_data["magnitude_spectrogram"])
                )

                # Plot Distance-Based anomalies
                fig_distance = plot_clusters_and_anomalies_distance(
                    x_axis=stft_data["kilometer_ref_resampled"],
                    anomalies=anomalies_distance,
                    distances_from_mean=distances_from_mean,
                    threshold=threshold,
                    x_axis_label="Mileage traveled [km]",
                )
                st.pyplot(fig_distance)

                # Create a DataFrame for Distance-Based anomalies
                anomaly_indices_distance = np.where(anomalies_distance)[0]
                anomaly_kilometers = stft_data["kilometer_ref_resampled"][
                    anomaly_indices_distance
                ]
                anomaly_frequencies = stft_data["frequencies"][
                    anomaly_indices_distance % len(stft_data["frequencies"])
                ]
                anomaly_times = times_acc[anomaly_indices_distance]

                # Create DataFrame for Distance-Based anomalies
                distance_anomalies_df = pd.DataFrame(
                    {
                        "Anomaly_Index": anomaly_indices_distance,
                        "Anomaly_Time": anomaly_times,
                        "Kilometer_Ref_Aligned": anomaly_kilometers,
                        "Anomaly_Frequency": anomaly_frequencies,
                        "Distance_From_Mean": distances_from_mean[
                            anomaly_indices_distance
                        ],
                    }
                )

                # Display the DataFrame and provide a download button
                st.dataframe(distance_anomalies_df)
                st.download_button(
                    label="Download Distance-Based Anomalies as CSV",
                    data=distance_anomalies_df.to_csv(index=False),
                    file_name="distance_anomalies.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Error during Distance-Based detection: {e}")
        else:
            st.warning("STFT data not available.")

    # TODO: Add anomaly visualization
    # # Tab 5: Anomaly Visualization
    # with tabs[4]:
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
