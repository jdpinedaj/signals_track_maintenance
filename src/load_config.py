# Imports
import yaml
from typing import Dict, Any
from pyprojroot import here
import os


class LoadConfig:
    """
    LoadConfig loads the configuration from `app_config.yml` and stores parameters as class attributes.
    This class is responsible for reading configuration sections related to data analysis, paths, and STFT feature extraction.
    The parameters are organized into attributes accessible throughout the app using `LoadConfig().attribute_name`.

    Attributes:
        Path Parameters:
            data_path (str): Path to the directory containing data files.
            anomalies_path (str): Path to the directory where anomaly files will be saved, resolved from the project root.

        Data Parameters:
            mat_data (str): Path to the .mat file containing the data. It selects the file based on the selected key.
            acceleration_to_analyze (str): Acceleration data to analyze. It selects the acceleration based on the selected key.

        Time Vector Parameters:
            sampling_frequency_tvp (int): Sampling frequency in Hz.
            cutoff_frequency (int): Cutoff frequency in Hz.
            new_sampling_frequency (int): New sampling frequency in Hz.

        STFT Features:
            window_length (float): Window length in seconds for STFT.
            overlap (float): Overlap percentage for STFT.
            gamma (int): Dynamic margin in decibels for STFT.
            sampling_frequency_stft_raw (int): Raw data sampling frequency in Hz.
            nfft_raw (int): Number of FFT points for raw data.
            sampling_frequency_stft_prepared (int): Prepared data sampling frequency in Hz.
            nfft_prepared (int): Number of FFT points for prepared data.

        Anomalies:
            percentile_kmeans (int): Percentile used for anomaly detection.

        Developer Comments:
            save_logs (bool): If True, logs are saved.

    Methods:
        __init__: Initializes the configuration by loading parameters from the configuration file.
        _load_data_to_analyze: Loads the data-related parameters.
        _load_paths: Loads the directory paths for data and anomaly saving.
        _load_time_vector_params: Loads time vector related parameters.
        _load_features_stft: Loads the STFT feature extraction parameters.
        _load_anomalies_params: Loads the anomaly detection parameters.
        _load_dev_comments: Loads the developer comments, such as whether to save logs.
        get_anomalies_filename: Constructs the full path for saving the anomalies, based on the anomaly type.
    """

    def __init__(self) -> None:
        # Load configuration from YAML file
        with open(here("configs/app_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # Load each section of the configuration
        self._load_paths(app_config)
        self._load_data_to_analyze(app_config)
        self._load_time_vector_params(app_config)
        self._load_features_stft(app_config)
        self._load_anomalies_params(app_config)
        self._load_dev_comments(app_config)

    def _load_paths(self, config: Dict[str, Any]) -> None:
        paths = config["paths"]

        # Resolve the root data and anomalies paths
        self.data_path = here(paths["data_path"])
        self.anomalies_base_path = here(paths["anomalies_path"])  # Base anomalies path

    def _load_data_to_analyze(self, config: Dict[str, Any]) -> None:
        self.data_to_analyze = config["data_to_analyze"]

        # Retrieve the selected file and acceleration keys
        self.selected_file_key = self.data_to_analyze["selected_file_key"]
        self.mat_data = os.path.join(
            self.data_path, self.data_to_analyze["data_files"][self.selected_file_key]
        )
        selected_acceleration_key = self.data_to_analyze["selected_acceleration_key"]
        self.acceleration_to_analyze = self.data_to_analyze["accelerations"][
            selected_acceleration_key
        ]

        # Set up the anomalies path with selected file key
        self.anomalies_path = os.path.join(
            self.anomalies_base_path, self.selected_file_key
        )
        os.makedirs(self.anomalies_path, exist_ok=True)  # Ensure the directory exists

    def _load_time_vector_params(self, config: Dict[str, Any]) -> None:
        time_vector_params = config["time_vector_params"]

        self.sampling_frequency_tvp = time_vector_params["sampling_frequency_tvp"]
        self.cutoff_frequency = time_vector_params["cutoff_frequency"]
        self.new_sampling_frequency = time_vector_params["new_sampling_frequency"]

    def _load_features_stft(self, config: Dict[str, Any]) -> None:
        features_stft = config["features_stft"]

        self.window_length = features_stft["window_length"]
        self.overlap = features_stft["overlap"]
        self.gamma = features_stft["gamma"]
        self.sampling_frequency_stft_raw = features_stft["sampling_frequency_stft_raw"]
        self.nfft_raw = features_stft["nfft_raw"]
        self.sampling_frequency_stft_prepared = features_stft[
            "sampling_frequency_stft_prepared"
        ]
        self.nfft_prepared = features_stft["nfft_prepared"]

    def _load_dev_comments(self, config: Dict[str, Any]) -> None:
        self.save_logs = config["dev_comments"]["save_logs"]

    def _load_anomalies_params(self, config: Dict[str, Any]) -> None:
        anomalies = config["anomalies"]

        self.percentile_kmeans = anomalies["percentile_kmeans"]

    def get_anomalies_filename(
        self,
        anomaly_type: str,
        file_key: str = None,
        acceleration_key: str = None,
        file_extension: str = "csv",
    ) -> str:
        """
        Constructs the full path for saving the anomalies file, based on the anomaly type and file extension.

        Args:
            anomaly_type (str): Type of anomaly (e.g., 'stft', 'distance', 'kmeans').
            file_key (str): Key of the data file being processed. Defaults to `self.selected_file_key`.
            acceleration_key (str): Key of the acceleration being processed. Defaults to `self.acceleration_to_analyze`.
            file_extension (str): File extension (e.g., 'csv', 'png'). Defaults to 'csv'.

        Returns:
            str: Full path for the anomalies file.
        """
        # Use provided keys if available; fallback to defaults
        file_key = file_key if file_key else self.selected_file_key
        acceleration_key = (
            acceleration_key if acceleration_key else self.acceleration_to_analyze
        )

        # Construct the filename
        filename = (
            f"{file_key}_{acceleration_key}_anomalies_{anomaly_type}.{file_extension}"
        )
        return os.path.join(self.anomalies_path, filename)
