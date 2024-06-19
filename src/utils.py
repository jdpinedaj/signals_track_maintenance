#!##########################################
#!############# IMPORTS ####################
#!##########################################


import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, stft, ShortTimeFFT
from scipy.signal.windows import hamming, gaussian


# Local Imports
from .logs import logger

#!##########################################
#!############# FUNCTIONS ##################
#!##########################################


def generate_time_vector(
    data,
    sampling_frequency_tvp,
    cutoff_frequency,
    new_sampling_frequency,
    position_speed,
):
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
        velocity_data (numpy.ndarray): Velocity data
    """

    # Initial setup
    num_samples = data.shape[0]  # Number of data points
    total_time = num_samples / sampling_frequency_tvp  # Total time in seconds

    # Extract velocity data
    velocity_data = data[
        :, position_speed
    ]  # Extract the velocity data from the data array

    # Filter parameters
    b, a = butter(
        4, 2 * cutoff_frequency / sampling_frequency_tvp
    )  # Design a 4th order Butterworth filter
    filtered_velocity = filtfilt(b, a, velocity_data)  # Apply the filter

    # Downsample the filtered velocity data
    downsampling_factor = int(
        sampling_frequency_tvp / new_sampling_frequency
    )  # Downsampling factor
    downsampled_velocity = filtered_velocity[
        ::downsampling_factor
    ]  # Downsample the data
    num_downsampled_samples = len(downsampled_velocity)  # Length of downsampled data

    # Generate time vectors
    original_time_vector = (
        np.arange(1, num_samples + 1) / sampling_frequency_tvp
    )  # Time vector for original data
    downsampled_time_vector = (
        np.arange(1, num_downsampled_samples + 1) / new_sampling_frequency
    )  # Time vector for downsampled data

    # Generate downsampled data
    downsampled_data = np.zeros((num_downsampled_samples, 2))
    downsampled_data[:, 0] = downsampled_time_vector
    downsampled_data[:, 1] = downsampled_velocity

    # Plots
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

    return original_time_vector, downsampled_time_vector, velocity_data


#!############################
#! Short Term Fourier Transform
#!############################


def short_term_fourier_transform_stft(
    signal, sampling_frequency_stft, window_length, overlap, gamma, start_time=0
):
    """
    Compute the Short-Term Fourier Transform (STFT) of a signal.
    Args:
        signal (numpy.ndarray): Input signal.
        sampling_frequency_stft (float): Sampling frequency in Hz.
        window_length (float): Window length in seconds.
        overlap (float): Overlap fraction.
        gamma (float): Dynamic margin.
        start_time (float): Start time of the signal in seconds.
    Returns:
        frequencies (numpy.ndarray): Frequency vector.
        times (numpy.ndarray): Time vector.
        magnitude_spectrogram (numpy.ndarray): Magnitude spectrogram.
        X_prime (numpy.ndarray): Normalized spectrogram.
    """

    # Initial setup
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

    noverlap = int(overlap * sampling_frequency_stft)  # Number of overlapping samples
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

    # Plotting the results
    fig, axs = plt.subplots(
        3, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 1, 1.2]}
    )

    axs[0].set_title("Original Spectrogram")
    img1 = axs[0].pcolormesh(
        times, frequencies, magnitude_spectrogram, cmap="jet", shading="gouraud"
    )
    axs[0].set_ylabel("Frequency [Hz]")
    axs[0].set_xlabel("Time [sec]")
    fig.colorbar(img1, ax=axs[0], label="Intensity [dB]")
    # axs[0].set_ylim([0, 1600])
    axs[0].grid(True)

    axs[1].set_title("Normalized Spectrogram")
    img2 = axs[1].pcolormesh(times, frequencies, X_prime, cmap="jet", shading="gouraud")
    axs[1].set_ylabel("Frequency [Hz]")
    axs[1].set_xlabel("Time [sec]")
    fig.colorbar(img2, ax=axs[1], label="Intensity [dB]")
    # axs[1].set_ylim([0, 1600])
    axs[1].grid(True)

    axs[2].set_title("Acceleration over Time")
    axs[2].plot(total_time, signal, "g")
    axs[2].set_xlabel("Time [sec]")
    axs[2].set_ylabel("Acceleration [m/s^2]")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    return frequencies, times, magnitude_spectrogram, X_prime


def short_term_fourier_transform_ShortTimeFFT(
    signal, sampling_frequency_stft, num_samples, window_length, overlap, gamma, g_std
):
    """
    Compute the Short-Time Fourier Transform (STFT) of a signal using ShortTimeFFT.
    Args:
        signal (numpy.ndarray): Input signal.
        sampling_frequency_stft (float): Sampling frequency in Hz.
        num_samples (int): Number of samples.
        window_length (float): Window length in seconds.
        overlap (float): Overlap fraction.
        gamma (float): Dynamic margin.
        g_std (float): Standard deviation for Gaussian window in samples.
    Returns:
        frequencies (numpy.ndarray): Frequency vector.
        times (numpy.ndarray): Time vector.
        magnitude_spectrogram (numpy.ndarray): Magnitude spectrogram.
        X_prime (numpy.ndarray): Normalized spectrogram.
    """

    # Initial setup
    t = np.arange(num_samples) / sampling_frequency_stft  # Time vector (s)
    f_i = 1 * np.arctan((t - t[num_samples // 2]) / 2) + 5  # varying frequency

    # Compute STFT
    window = int(window_length * sampling_frequency_stft)

    # Calculate the hop
    noverlap = int(window * overlap)  # Number of overlapping samples
    hop = window - noverlap  # Hop size

    # Calculate the Gaussian window
    window_samples = gaussian(window, std=g_std, sym=True)  # Gaussian window

    mfft = (
        window * 4
    )  # The utilized Gaussian window is 50 samples or 2.5 s long. The parameter ``mfft=200`` in `ShortTimeFFT` causes the spectrum to be oversampled by a factor of 4:

    SFT = ShortTimeFFT(
        window_samples,
        hop=hop,
        fs=sampling_frequency_stft,
        mfft=mfft,
        scale_to="magnitude",
    )
    Sx = SFT.stft(signal)  # Perform the STFT

    # Convert to magnitude spectrogram
    magnitude_spectrogram = np.abs(Sx)

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

    # Generate time and frequency vectors for plotting
    num_time_bins = magnitude_spectrogram.shape[1]
    num_freq_bins = magnitude_spectrogram.shape[0]
    times = np.linspace(0, num_samples / sampling_frequency_stft, num_time_bins)
    frequencies = np.linspace(0, sampling_frequency_stft / 2, num_freq_bins)

    # Debug prints
    print("times shape:", times.shape)
    print("frequencies shape:", frequencies.shape)
    print("magnitude_spectrogram shape:", magnitude_spectrogram.shape)
    print("times:", times)
    print("frequencies:", frequencies)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.title("Original Spectrogram")
    plt.pcolormesh(
        times, frequencies, magnitude_spectrogram, shading="gouraud", cmap="jet"
    )
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.colorbar(label="Intensity [dB]")
    plt.ylim([0, 1600])

    plt.subplot(3, 1, 2)
    plt.title("Normalized Spectrogram")
    plt.pcolormesh(times, frequencies, X_prime, shading="gouraud", cmap="jet")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.colorbar(label="Intensity [dB]")
    plt.ylim([0, 1600])

    plt.subplot(3, 1, 3)
    plt.title("Acceleration over Time")
    plt.plot(t, signal, "g")
    plt.xlabel("Time [sec]")
    plt.ylabel("Acceleration [m/s^2]")
    plt.ylim([-6, 6])
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plotting the detailed plot from the documentation example
    fig1, ax1 = plt.subplots(figsize=(6.0, 4.0))  # enlarge plot a bit
    t_lo, t_hi = SFT.extent(num_samples)[:2]  # time range of plot
    ax1.set_title(
        rf"STFT ({SFT.m_num*SFT.T:g}$\,s$ Gaussian window, "
        + rf"$\sigma_t={g_std*SFT.T}\,$s)"
    )
    ax1.set(
        xlabel=f"Time $t$ in seconds ({SFT.p_num(num_samples)} slices, "
        + rf"$\Delta t = {SFT.delta_t:g}\,$s)",
        ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, "
        + rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
        xlim=(t_lo, t_hi),
    )

    im1 = ax1.imshow(
        abs(Sx),
        origin="lower",
        aspect="auto",
        extent=SFT.extent(num_samples),
        cmap="viridis",
    )
    ax1.plot(t, f_i, "r--", alpha=0.5, label="$f_i(t)$")
    fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")

    # Shade areas where window slices stick out to the side:
    for t0_, t1_ in [
        (t_lo, SFT.lower_border_end[0] * SFT.T),
        (SFT.upper_border_begin(num_samples)[0] * SFT.T, t_hi),
    ]:
        ax1.axvspan(t0_, t1_, color="w", linewidth=0, alpha=0.2)
    for t_ in [0, num_samples * SFT.T]:  # mark signal borders with vertical line:
        ax1.axvline(t_, color="y", linestyle="--", alpha=0.5)
    ax1.legend()
    fig1.tight_layout()
    plt.show()

    return frequencies, times, magnitude_spectrogram, X_prime


def short_term_fourier_transform_ShortTimeFFT_v2(
    signal, sampling_frequency_stft, num_samples, window_length, overlap, gamma
):
    """
    Compute the Short-Time Fourier Transform (STFT) of a signal using ShortTimeFFT.
    Args:
        signal (numpy.ndarray): Input signal.
        sampling_frequency_stft (float): Sampling frequency in Hz.
        num_samples (int): Number of samples.
        window_length (float): Window length in seconds.
        overlap (float): Overlap fraction.
        gamma (float): Dynamic margin.
    Returns:
        frequencies (numpy.ndarray): Frequency vector.
        times (numpy.ndarray): Time vector.
        magnitude_spectrogram (numpy.ndarray): Magnitude spectrogram.
        X_prime (numpy.ndarray): Normalized spectrogram.
    """

    # Compute STFT
    window_samples = int(
        window_length * sampling_frequency_stft
    )  # Window length in samples
    noverlap = int(window_samples * overlap)  # Number of overlapping samples
    hop = window_samples - noverlap  # Hop size
    # hop = 10

    #! ANOTHER WAY
    # g_std = window_samples / 8  # Standard deviation for Gaussian window in samples
    # window = gaussian(window_samples, std=g_std, sym=True)  # Gaussian window

    window = hamming(window_samples)  # Hamming window

    SFT = ShortTimeFFT(
        window,
        hop=hop,
        fs=sampling_frequency_stft,
        mfft=window_samples,
        scale_to="magnitude",
    )
    Sx = SFT.stft(signal)  # Perform the STFT

    # Convert to magnitude spectrogram
    magnitude_spectrogram = np.abs(Sx)

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

    # Generate time and frequency vectors for plotting
    num_time_bins = magnitude_spectrogram.shape[1]
    num_freq_bins = magnitude_spectrogram.shape[0]
    times = np.linspace(0, num_samples / sampling_frequency_stft, num_time_bins)
    frequencies = np.linspace(0, sampling_frequency_stft / 2, num_freq_bins)

    # Debug prints
    print("times shape:", times.shape)
    print("frequencies shape:", frequencies.shape)
    print("magnitude_spectrogram shape:", magnitude_spectrogram.shape)
    print("times:", times)
    print("frequencies:", frequencies)

    # # Plotting the results
    # plt.figure(figsize=(10, 6))
    # plt.subplot(3, 1, 1)
    # plt.title("Original Spectrogram")
    # plt.pcolormesh(
    #     times, frequencies, magnitude_spectrogram, shading="gouraud", cmap="jet"
    # )
    # plt.ylabel("Frequency [Hz]")
    # plt.xlabel("Time [sec]")
    # plt.colorbar(label="Intensity [dB]")
    # plt.ylim([0, 1600])

    # plt.subplot(3, 1, 2)
    # plt.title("Normalized Spectrogram")
    # plt.pcolormesh(times, frequencies, X_prime, shading="gouraud", cmap="jet")
    # plt.ylabel("Frequency [Hz]")
    # plt.xlabel("Time [sec]")
    # plt.colorbar(label="Intensity [dB]")
    # plt.ylim([0, 1600])

    # plt.subplot(3, 1, 3)
    # plt.title("Acceleration over Time")
    # plt.plot(t, signal, "g")
    # plt.xlabel("Time [sec]")
    # plt.ylabel("Acceleration [m/s^2]")
    # plt.ylim([-6, 6])
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()

    # Plotting the detailed plot from the documentation example
    fig1, ax1 = plt.subplots(figsize=(6.0, 4.0))  # enlarge plot a bit
    t_lo, t_hi = times[0], times[-1]  # time range of plot
    ax1.set_title(
        rf"STFT ({window_samples / sampling_frequency_stft:g}$\,s$ Hamming window, "
        + rf"$\sigma_t={window_samples / sampling_frequency_stft}\,$s)"
    )
    ax1.set(
        xlabel=f"Time $t$ in seconds ({num_time_bins} slices, "
        + rf"$\Delta t = {(times[1] - times[0]):g}\,$s)",
        ylabel=f"Freq. $f$ in Hz ({num_freq_bins} bins, "
        + rf"$\Delta f = {frequencies[1] - frequencies[0]:g}\,$Hz)",
        xlim=(t_lo, t_hi),
    )

    im1 = ax1.imshow(
        magnitude_spectrogram,
        origin="lower",
        aspect="auto",
        extent=[t_lo, t_hi, frequencies[0], frequencies[-1]],
        cmap="viridis",
    )
    dominant_frequencies = frequencies[np.argmax(magnitude_spectrogram, axis=0)]
    ax1.plot(times, dominant_frequencies, "r--", alpha=0.5, label="$f_i(t)$")
    fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")

    # Shade areas where window slices stick out to the side:
    for t0_, t1_ in [
        (t_lo, times[0]),
        (times[-1], t_hi),
    ]:
        ax1.axvspan(t0_, t1_, color="w", linewidth=0, alpha=0.2)
    for t_ in [
        0,
        num_samples / sampling_frequency_stft,
    ]:  # mark signal borders with vertical line:
        ax1.axvline(t_, color="y", linestyle="--", alpha=0.5)
    ax1.legend()
    fig1.tight_layout()
    plt.show()

    return frequencies, times, magnitude_spectrogram, X_prime
