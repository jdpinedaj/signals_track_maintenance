<!-- pandoc README.md -s -o README.docx -->

# Signal processing for railway track maintenance prediction.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

<hr>

The aim of this project is to predict the maintenance of railway tracks using signal processing techniques. The data used in this project is from Metro Valencia, a Spanish railway company. The data consists of the vibration signals of the railway tracks. The data is collected from the sensors placed on the locomotives to monitor the condition of the railway tracks.

# Files

The main files of the project are:

1. `src/utils.py`: Utility functions.
2. `src/logs.py`: Logging functions.
3. `src/load_config.py` and `configs/app_config.yml`: Configuration files and parameters.

# Parameters

- window_length: This is 0.25 seconds. With a sampling_frequency_stft_prepared of 100 Hz, this means each window will contain 0.25×100=25 samples.
- overlap: 95% overlap means there is substantial overlap between windows. Only 5% of the data is unique per window.
- nfft: This determines the number of points used for the FFT. The nfft_prepared value of 128 determines the frequency resolution of the resulting spectrogram.
- sampling_frequency_stft_prepared: 100 Hz means you're working with data points spaced at 0.01-second intervals.

# TODOs

<!-- - Revisar modelo kmeans con 5%. -->
<!-- - Pasar threshold a parametros -->
<!-- - Agregar distancia en csvs (kilometer_ref_fixed_km). -->
<!-- - plotly para superponer graficos. -->

- Revisar nuevos datos.
- Documentar.
- tablero
