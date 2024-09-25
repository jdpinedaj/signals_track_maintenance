# Imports
import yaml
from typing import Dict, Any
from pyprojroot import here


class LoadConfig:
    """
    LoadConfig loads the configuration from `app_config.yml` and stores parameters as class attributes.
    This class is responsible for reading configuration sections related to data analysis and STFT feature extraction.
    The parameters are organized into attributes accessible throughout the app using `LoadConfig().attribute_name`.

    Attributes:
        data_to_analyze (str): Type of acceleration data to analyze.

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

        Developer Comments:
            save_logs (bool): If True, logs are saved.

    Methods:
        __init__: Initializes the configuration by loading parameters from the configuration file.
    """

    def __init__(self) -> None:
        # Load configuration from YAML file
        with open(here("configs/app_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # Load each section of the configuration
        self._load_data_to_analyze(app_config)
        self._load_time_vector_params(app_config)
        self._load_features_stft(app_config)
        self._load_dev_comments(app_config)

    def _load_data_to_analyze(self, config: Dict[str, Any]) -> None:
        self.acceleration_to_analyze = config["data_to_analyze"][
            "acceleration_to_analyze"
        ]

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
