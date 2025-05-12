import numpy as np
from scipy import signal, fftpack
import pywt  # PyWavelets


class SignalProcessing:
    """
    Performs signal processing on raw I/Q data obtained from the SDR.
    """

    def __init__(self, sample_rate):
        self.sample_rate = float(sample_rate)
        print(f"SignalProcessing initialized with sample rate: {self.sample_rate / 1e6} Msps")

    def process(self, raw_data):
        """
        Main processing method that handles raw I/Q data and returns processed results.

        Args:
            raw_data (np.ndarray): Raw I/Q data from SDR

        Returns:
            dict: Dictionary containing processed data and analysis results
        """
        try:
            if not isinstance(raw_data, np.ndarray) or raw_data.size == 0:
                print("Process Error: Invalid input data")
                return None

            # Basic processing pipeline
            results = {}

            # 1. FFT analysis
            fft_freq, fft_mag = self.perform_fft(raw_data)
            if fft_freq is not None:
                results['fft'] = {'frequencies': fft_freq, 'magnitude': fft_mag}

            # 2. Envelope analysis
            amp_env, phase = self.analyze_envelope(raw_data)
            if amp_env is not None:
                results['envelope'] = {'amplitude': amp_env, 'phase': phase}

            # 3. Wavelet analysis (on a subset for performance)
            coeffs, freqs = self.perform_wavelet_analysis(raw_data[:1024])
            if coeffs is not None:
                results['wavelet'] = {'coefficients': coeffs, 'frequencies': freqs}

            return results

        except Exception as e:
            print(f"Error in processing pipeline: {e}")
            return None

    def perform_stft(self, iq_data, nperseg=256, noverlap=None):
        """
        Performs Short-Time Fourier Transform (STFT) on the I/Q data.

        Args:
            iq_data (np.ndarray): Complex I/Q data.
            nperseg (int): Length of each segment for STFT.
            noverlap (int, optional): Number of points to overlap between segments. 
                                      If None, noverlap = nperseg // 2.

        Returns:
            tuple: (frequencies (np.ndarray), times (np.ndarray), Zxx (np.ndarray) - STFT result)
                   Returns (None, None, None) if an error occurs.
        """
        if not isinstance(iq_data, np.ndarray) or iq_data.ndim != 1:
            print("STFT Error: Input data must be a 1D numpy array.")
            return None, None, None
        if iq_data.size == 0:
            print("STFT Error: Input data is empty.")
            return None, None, None
        try:
            f, t, Zxx = signal.stft(iq_data, fs=self.sample_rate, nperseg=nperseg, noverlap=noverlap)
            return f, t, np.abs(Zxx)  # Return magnitude of STFT
        except Exception as e:
            print(f"Error during STFT: {e}")
            return None, None, None

    def create_bscan_matrix(self, traces):
        """
        Converts individual radar traces into a B-scan matrix.

        Args:
            traces (list of np.ndarray): List of complex or magnitude traces (1D arrays).

        Returns:
            np.ndarray: 2D B-scan matrix where each column is a trace.
        """
        if not traces:
            return np.array([])

        # Normalize all traces to the same length
        max_len = max(len(trace) for trace in traces)
        bscan = np.zeros((max_len, len(traces)), dtype=np.float32)

        # Fill the B-scan matrix with the magnitudes of each trace
        for i, trace in enumerate(traces):
            trace_len = len(trace)
            bscan[:trace_len, i] = np.abs(trace)  # Use magnitude of each trace

        return bscan

    def perform_wavelet_analysis(self, iq_data, wavelet_name='morl', scales=None):
        """
        Performs Continuous Wavelet Transform (CWT) on the I/Q data (magnitude).

        Args:
            iq_data (np.ndarray): Complex I/Q data.
            wavelet_name (str): Name of the wavelet to use (e.g., 'gaus1', 'morl', 'cmorB-C').
            scales (np.ndarray, optional): Scales to use for CWT. If None, a default range is used.

        Returns:
            tuple: (coefficients (np.ndarray), frequencies (np.ndarray))
                   Returns (None, None) if an error occurs.
        """
        if not isinstance(iq_data, np.ndarray) or iq_data.ndim != 1:
            print("Wavelet Error: Input data must be a 1D numpy array.")
            return None, None
        if iq_data.size == 0:
            print("Wavelet Error: Input data is empty.")
            return None, None

        try:
            # Using magnitude of I/Q data for CWT, or could process I and Q separately
            data_magnitude = np.abs(iq_data)

            if scales is None:
                # Define a range of scales, e.g., from 1 to a fraction of signal length
                max_scale = len(data_magnitude) // 4
                if max_scale < 2: max_scale = 2  # Ensure at least a minimal scale range
                scales = np.arange(1, max_scale)

            if len(scales) == 0 or scales[-1] == 0:
                print("Wavelet Error: Invalid scales provided or calculated.")
                return None, None

            coefficients, frequencies = pywt.cwt(data_magnitude, scales, wavelet_name,
                                                 sampling_period=(1.0 / self.sample_rate))
            return coefficients, frequencies
        except Exception as e:
            print(f"Error during Wavelet analysis: {e}")
            return None, None

    def analyze_envelope(self, iq_data):
        """
        Calculates the amplitude envelope (magnitude) and instantaneous phase.

        Args:
            iq_data (np.ndarray): Complex I/Q data.

        Returns:
            tuple: (amplitude_envelope (np.ndarray), instantaneous_phase (np.ndarray))
                   Returns (None, None) if an error occurs.
        """
        if not isinstance(iq_data, np.ndarray) or iq_data.ndim != 1:
            print("Envelope Analysis Error: Input data must be a 1D numpy array.")
            return None, None
        if iq_data.size == 0:
            print("Envelope Analysis Error: Input data is empty.")
            return None, None
        try:
            amplitude_envelope = np.abs(iq_data)
            instantaneous_phase = np.unwrap(np.angle(iq_data))  # Unwrap to avoid phase jumps
            return amplitude_envelope, instantaneous_phase
        except Exception as e:
            print(f"Error during envelope analysis: {e}")
            return None, None

    def calculate_instantaneous_frequency(self, iq_data):
        """
        Calculates the instantaneous frequency from the phase.

        Args:
            iq_data (np.ndarray): Complex I/Q data.

        Returns:
            np.ndarray: Instantaneous frequency, or None if an error occurs.
        """
        _, instantaneous_phase = self.analyze_envelope(iq_data)
        if instantaneous_phase is None:
            return None
        try:
            # Differentiate phase and divide by 2*pi, then scale by sample rate
            instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi)) * self.sample_rate
            # Pad to match original length (optional, or return one shorter)
            instantaneous_frequency = np.concatenate(([instantaneous_frequency[0]], instantaneous_frequency))
            return instantaneous_frequency
        except Exception as e:
            print(f"Error calculating instantaneous frequency: {e}")
            return None

    def multi_resolution_analysis_dwt(self, iq_data, wavelet_name='db4', level=None):
        """
        Performs Multi-Resolution Analysis using Discrete Wavelet Transform (DWT).
        This decomposes the signal into approximation and detail coefficients at multiple levels.

        Args:
            iq_data (np.ndarray): Complex I/Q data (will process magnitude).
            wavelet_name (str): Name of the wavelet to use (e.g., 'db4', 'sym8').
            level (int, optional): Decomposition level. If None, it will be automatically determined.

        Returns:
            list: A list of DWT coefficients [cA_n, cD_n, cD_n-1, ..., cD_1],
                  or None if an error occurs.
        """
        if not isinstance(iq_data, np.ndarray) or iq_data.ndim != 1:
            print("MRA Error: Input data must be a 1D numpy array.")
            return None
        if iq_data.size == 0:
            print("MRA Error: Input data is empty.")
            return None

        try:
            data_magnitude = np.abs(iq_data)
            if level is None:
                level = pywt.dwt_max_level(len(data_magnitude), pywt.Wavelet(wavelet_name))
                print(f"MRA: Auto-determined DWT level: {level}")

            coeffs = pywt.wavedec(data_magnitude, wavelet_name, level=level)
            return coeffs
        except Exception as e:
            print(f"Error during Multi-Resolution Analysis (DWT): {e}")
            return None

    def estimate_dielectric_properties(self, reflection_data, transmitted_signal_params):
        """
        Placeholder for estimating dielectric properties.
        This is a complex topic often requiring frequency domain analysis and modeling.

        Args:
            reflection_data (np.ndarray): Received reflection data (time-domain I/Q).
            transmitted_signal_params (dict): Parameters of the transmitted signal (e.g., center_freq, bandwidth).
        Returns:
            dict: Estimated properties (e.g., relative permittivity, conductivity).
        """
        print("Dielectric property estimation is a complex R&D task and not fully implemented here.")
        print("Conceptual: Would involve analyzing frequency-dependent response based on models like Debye, Cole-Cole.")
        return {"relative_permittivity_est": None, "conductivity_est": None}