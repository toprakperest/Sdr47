import adi
import numpy as np
import time

class SDRController:
    def _calculate_fft(self, iq_data):
        # Ensure sample_rate is available if this method is used
        if not hasattr(self, 'sample_rate') or self.sample_rate == 0:
            print("Warning: Sample rate not set for FFT calculation.")
            return {'frequencies': np.array([]), 'magnitude': np.array([])}
        N = len(iq_data)
        if N == 0:
            return {'frequencies': np.array([]), 'magnitude': np.array([])}
        yf = np.fft.fft(iq_data)
        xf = np.fft.fftfreq(N, 1 / self.sample_rate)[:N // 2]
        return {'frequencies': xf, 'magnitude': 2.0 / N * np.abs(yf[:N // 2])}

    def _calculate_envelope(self, iq_data):
        if len(iq_data) == 0:
            return {'amplitude': np.array([]), 'phase': np.array([])}
        return {
            'amplitude': np.abs(iq_data),
            'phase': np.unwrap(np.angle(iq_data))
        }

    def __init__(self, signal_processor=None, ai_model_handler=None,
                 sdr_ip="ip:192.168.2.1", sample_rate=2e6, rx_buffer_size=2 ** 12,
                 center_freq=100e6, rx_rf_bandwidth=2e6,
                 gain_control_mode_chan0="manual", rx_hardware_gain_chan0=30.0,
                 gain_control_mode_chan1="manual", rx_hardware_gain_chan1=30.0):
        """
        Initializes the SDRController.
        Parameters are typically loaded from config but can be overridden.
        """
        self.signal_processor = signal_processor
        self.ai_model_handler = ai_model_handler

        self.sdr_ip = sdr_ip
        self.sdr = None # SDR object, initialized in connect()
        self.sample_rate = int(sample_rate)
        self.rx_buffer_size = int(rx_buffer_size) # Default buffer size
        self.center_freq = int(center_freq)
        self.rx_rf_bandwidth = int(rx_rf_bandwidth)
        
        # Channel specific gain settings
        self.gain_control_mode_chan0 = gain_control_mode_chan0
        self.rx_hardware_gain_chan0 = float(rx_hardware_gain_chan0)
        self.gain_control_mode_chan1 = gain_control_mode_chan1
        self.rx_hardware_gain_chan1 = float(rx_hardware_gain_chan1)
        
        # Store default TX attenuations, can be changed by AntennaManager or set_tx_attenuation
        self.tx_hardware_gain_chan0_default = -10.0 # dB, for PlutoSDR this is attenuation
        self.tx_hardware_gain_chan1_default = -89.75 # dB, max attenuation

        print(f"SDRController initialized with target IP: {self.sdr_ip}")

    def connect(self):
        """
        Connects to the SDR hardware and configures it.
        Returns:
            tuple: (bool, str) indicating success and a message.
        """
        try:
            print(f"Attempting to connect to SDR at {self.sdr_ip}...")
            self.sdr = adi.Pluto(self.sdr_ip) # Connect to PlutoSDR
            print(f"Successfully connected to SDR: {self.sdr.uri}")
            self._configure_sdr()
            return True, "SDR bağlantısı başarılı." # "SDR connection successful."
        except Exception as e:
            print(f"Error connecting to or configuring SDR: {e}")
            print("Please ensure the SDR (e.g., PlutoSDR) is connected, powered on,")
            print("and necessary drivers (like libiio and pyadi-iio) are correctly installed.")
            self.sdr = None
            return False, f"SDR bağlantı hatası: {e}" # "SDR connection error"

    def _configure_sdr(self):
        """
        Internal method to configure the connected SDR with parameters from __init__.
        This is called by connect().
        """
        if not self.sdr:
            print("SDR not connected. Cannot configure.")
            return

        try:
            print(f"Configuring SDR: Sample Rate={self.sample_rate/1e6:.2f} Msps, Center Freq={self.center_freq/1e6:.2f} MHz, RX RF Bandwidth={self.rx_rf_bandwidth/1e6:.2f} MHz")
            
            self.sdr.sample_rate = self.sample_rate
            self.sdr.rx_lo = self.center_freq
            self.sdr.tx_lo = self.center_freq 
            self.sdr.rx_rf_bandwidth = self.rx_rf_bandwidth
            # For Pluto, tx_rf_bandwidth is usually coupled with rx_rf_bandwidth or set similarly.
            # If your device allows separate setting and it's needed:
            # self.sdr.tx_rf_bandwidth = self.rx_rf_bandwidth # Or another configured value

            self.sdr.rx_buffer_size = self.rx_buffer_size

            # Configure RX Channel 0 Gain
            self.sdr.gain_control_mode_chan0 = self.gain_control_mode_chan0
            if self.gain_control_mode_chan0 == "manual":
                self.sdr.rx_hardware_gain_chan0 = self.rx_hardware_gain_chan0
            print(f"SDR RX0: Gain Mode='{self.sdr.gain_control_mode_chan0}', Hardware Gain={self.sdr.rx_hardware_gain_chan0} dB")

            # Configure RX Channel 1 Gain (if available)
            if hasattr(self.sdr, 'gain_control_mode_chan1') and hasattr(self.sdr, 'rx_hardware_gain_chan1'):
                self.sdr.gain_control_mode_chan1 = self.gain_control_mode_chan1
                if self.gain_control_mode_chan1 == "manual":
                    self.sdr.rx_hardware_gain_chan1 = self.rx_hardware_gain_chan1
                print(f"SDR RX1: Gain Mode='{self.sdr.gain_control_mode_chan1}', Hardware Gain={self.sdr.rx_hardware_gain_chan1} dB")
            else:
                print("SDR RX1: Gain control not available or attributes not found.")

            # Configure TX Channel 0 Attenuation (tx_hardware_gain for Pluto is attenuation)
            if hasattr(self.sdr, 'tx_hardware_gain_chan0'):
                self.sdr.tx_hardware_gain_chan0 = self.tx_hardware_gain_chan0_default 
                print(f"SDR TX0: Attenuation set to {self.sdr.tx_hardware_gain_chan0} dB")
            
            # Configure TX Channel 1 Attenuation (if available)
            if hasattr(self.sdr, 'tx_hardware_gain_chan1'):
                 self.sdr.tx_hardware_gain_chan1 = self.tx_hardware_gain_chan1_default
                 print(f"SDR TX1: Attenuation set to {self.sdr.tx_hardware_gain_chan1} dB (max attenuation)")

            # Initialize RX/TX channels to disabled; AntennaManager or specific functions will enable them.
            self.sdr.rx_enabled_channels = []
            self.sdr.tx_enabled_channels = []
            print("SDR base configuration complete. RX/TX channels are initially disabled.")
            print("Use AntennaManager or set_active_rx/tx_channels to enable specific channels.")

        except Exception as e:
            print(f"Error during SDR configuration: {e}")
            # Consider how to handle partial configuration errors.
            # For now, connection might still be considered 'up' but config failed.

    def set_active_rx_channels(self, channels: list[int]):
        if not self.sdr:
            print("SDR not connected. Cannot set active RX channels.")
            return False
        try:
            self.sdr.rx_enabled_channels = channels
            print(f"SDR Active RX channels set to: {self.sdr.rx_enabled_channels}")
            return True
        except Exception as e:
            print(f"Error setting active RX channels to {channels}: {e}")
            return False

    def set_active_tx_channels(self, channels: list[int]):
        if not self.sdr:
            print("SDR not connected. Cannot set active TX channels.")
            return False
        try:
            self.sdr.tx_enabled_channels = channels
            print(f"SDR Active TX channels set to: {self.sdr.tx_enabled_channels}")
            return True
        except Exception as e:
            print(f"Error setting active TX channels to {channels}: {e}")
            return False

    def set_center_frequency(self, freq_hz: int):
        if not self.sdr:
            print("SDR not connected. Cannot set frequency.")
            return False
        try:
            self.center_freq = int(freq_hz)
            self.sdr.rx_lo = self.center_freq
            self.sdr.tx_lo = self.center_freq
            print(f"SDR RX/TX LO frequency set to: {self.sdr.rx_lo / 1e6:.2f} MHz")
            return True
        except Exception as e:
            print(f"Error setting center frequency to {freq_hz / 1e6:.2f} MHz: {e}")
            return False

    def set_sample_rate(self, sample_rate_sps: int):
        if not self.sdr:
            print("SDR not connected. Cannot set sample rate.")
            return False
        try:
            self.sample_rate = int(sample_rate_sps)
            self.sdr.sample_rate = self.sample_rate
            print(f"SDR sample rate set to: {self.sdr.sample_rate / 1e6:.2f} Msps")
            return True
        except Exception as e:
            print(f"Error setting sample rate to {sample_rate_sps / 1e6:.2f} Msps: {e}")
            return False

    def set_rx_rf_bandwidth(self, bandwidth_hz: int):
        if not self.sdr:
            print("SDR not connected. Cannot set RX RF bandwidth.")
            return False
        try:
            self.rx_rf_bandwidth = int(bandwidth_hz)
            self.sdr.rx_rf_bandwidth = self.rx_rf_bandwidth
            print(f"SDR RX RF bandwidth set to: {self.sdr.rx_rf_bandwidth / 1e6:.2f} MHz")
            return True
        except Exception as e:
            print(f"Error setting RX RF bandwidth to {bandwidth_hz / 1e6:.2f} MHz: {e}")
            return False

    def set_rx_gain(self, gain_db: float, channel: int = 0, mode: str = "manual"):
        if not self.sdr:
            print("SDR not connected. Cannot set RX gain.")
            return False
        try:
            target_gain_attr = f"rx_hardware_gain_chan{channel}"
            target_mode_attr = f"gain_control_mode_chan{channel}"
            
            if hasattr(self.sdr, target_gain_attr) and hasattr(self.sdr, target_mode_attr):
                setattr(self.sdr, target_mode_attr, mode)
                if mode == "manual":
                    setattr(self.sdr, target_gain_attr, float(gain_db))
                # Read back the actual gain set
                current_mode = getattr(self.sdr, target_mode_attr)
                current_gain = getattr(self.sdr, target_gain_attr)
                print(f"SDR RX CH{channel}: Gain mode set to '{current_mode}', Hardware Gain is {current_gain} dB")
                return True
            else:
                print(f"SDR RX CH{channel}: Gain control attributes not available.")
                return False
        except Exception as e:
            print(f"Error setting RX gain for CH{channel}: {e}")
            return False

    def set_tx_attenuation(self, attenuation_db: float, channel: int = 0):
        """ Sets TX attenuation. For PlutoSDR, tx_hardware_gain is attenuation (0 max power, -89.75 min power). """
        if not self.sdr:
            print("SDR not connected. Cannot set TX attenuation.")
            return False
        try:
            target_atten_attr = f"tx_hardware_gain_chan{channel}"
            if hasattr(self.sdr, target_atten_attr):
                # Clamp attenuation_db to PlutoSDR's valid range for safety
                # Valid range for tx_hardware_gain is typically 0.0 to -89.75 dB
                clamped_attenuation = np.clip(float(attenuation_db), -89.75, 0.0)
                setattr(self.sdr, target_atten_attr, clamped_attenuation)
                current_atten = getattr(self.sdr, target_atten_attr)
                print(f"SDR TX CH{channel}: Attenuation set to {current_atten} dB")
                return True
            else:
                print(f"SDR TX CH{channel}: Attenuation control attribute not available.")
                return False
        except Exception as e:
            print(f"Error setting TX attenuation for CH{channel}: {e}")
            return False

    def capture_data(self, num_samples: int = None):
        """
        Captures data from the configured RX channels.
        Args:
            num_samples (int, optional): Number of samples to capture. 
                                         If None, uses current self.rx_buffer_size.
                                         Changing rx_buffer_size on-the-fly can be device-specific.
        Returns:
            numpy.ndarray or list[numpy.ndarray] or None:
            - A single NumPy array if one RX channel is active.
            - A list of NumPy arrays if multiple RX channels are active.
            - None if SDR is not connected or an error occurs.
        """
        if not self.sdr:
            print("SDR not connected. Cannot capture data.")
            return None # Changed from np.array([]) to None for clearer error indication

        actual_num_samples = self.rx_buffer_size
        if num_samples is not None:
            num_samples = int(num_samples)
            if self.sdr.rx_buffer_size != num_samples:
                try:
                    # Attempt to set buffer size if different. This might be restricted.
                    print(f"Attempting to set SDR RX buffer size to: {num_samples}")
                    self.sdr.rx_buffer_size = num_samples
                    actual_num_samples = num_samples
                    print(f"SDR RX buffer size updated to: {self.sdr.rx_buffer_size}")
                except Exception as e:
                    print(f"Warning: Could not update SDR RX buffer size to {num_samples}: {e}. Using current: {self.sdr.rx_buffer_size}")
                    actual_num_samples = self.sdr.rx_buffer_size # Use existing buffer size
        
        try:
            if not self.sdr.rx_enabled_channels:
                print("Warning: No RX channels enabled. Capturing may fail or use SDR default.")
                # Consider enabling a default channel if this is a common issue, e.g.,
                # if not self.set_active_rx_channels([0]): return None 
                # However, AntennaManager should ideally handle this.
                return None # No channels enabled, so no data

            print(f"Capturing {actual_num_samples} samples from RX channels: {self.sdr.rx_enabled_channels}...")
            data = self.sdr.rx()  # This is a blocking call

            if data is None: # Should ideally not happen if rx() itself doesn't raise error
                print("SDR rx() returned None unexpectedly.")
                return None
            
            # sdr.rx() returns:
            # - single numpy.ndarray if 1 channel enabled
            # - list of numpy.ndarray if >1 channel enabled
            # This behavior is fine, consumer needs to be aware.
            if isinstance(data, list):
                print(f"Captured {len(data[0])} samples each from {len(data)} channels.")
            elif isinstance(data, np.ndarray):
                print(f"Captured {len(data)} samples from a single channel.")
            
            return data
        except Exception as e:
            print(f"Error during data capture: {e}")
            return None

    def transmit_data(self, data_iq: np.ndarray, cyclic: bool = False):
        """
        Transmits IQ data using the configured TX channels.
        Args:
            data_iq (numpy.ndarray): Complex IQ data to transmit.
            cyclic (bool): Whether to transmit cyclically.
        """
        if not self.sdr:
            print("SDR not connected. Cannot transmit data.")
            return False
        
        if not isinstance(data_iq, np.ndarray) or data_iq.dtype not in [np.complex64, np.complex128]:
            print("Error: transmit_data expects a complex NumPy array.")
            return False

        try:
            if not self.sdr.tx_enabled_channels:
                 print("Warning: No TX channels enabled. Transmission may fail or use SDR default.")
                 # Consider enabling a default TX channel if appropriate, e.g.,
                 # if not self.set_active_tx_channels([0]): return False
                 # AntennaManager should handle this.
                 return False # No channels enabled for TX

            self.sdr.tx_cyclic_buffer = cyclic
            self.sdr.tx(data_iq) # data_iq must be complex64 for Pluto
            print(f"Transmitted {len(data_iq)} samples. Cyclic: {cyclic}. TX Channels: {self.sdr.tx_enabled_channels}")
            return True
        except Exception as e:
            print(f"Error transmitting data: {e}")
            return False

    def stop_tx(self):
        """Stops any ongoing transmission and disables TX channels."""
        if not self.sdr:
            print("SDR not connected. Cannot stop TX.")
            return False
        try:
            if hasattr(self.sdr, 'tx_destroy_buffer'): # Check if method exists
                self.sdr.tx_destroy_buffer()
            # Disabling TX channels is a more robust way to ensure transmission stops
            self.sdr.tx_enabled_channels = []
            print("SDR TX buffer destroyed and all TX channels disabled.")
            return True
        except Exception as e:
            print(f"Error stopping TX: {e}")
            return False

    def get_sdr_properties(self):
        """Returns a dictionary of current SDR properties."""
        if not self.sdr:
            print("SDR not connected. Cannot get properties.")
            return {}
        try:
            props = {
                "uri": self.sdr.uri,
                "sdr_ip_configured": self.sdr_ip,
                "sample_rate_sps": self.sdr.sample_rate,
                "rx_lo_hz": self.sdr.rx_lo,
                "tx_lo_hz": self.sdr.tx_lo,
                "rx_rf_bandwidth_hz": self.sdr.rx_rf_bandwidth,
                "tx_rf_bandwidth_hz": self.sdr.tx_rf_bandwidth if hasattr(self.sdr, "tx_rf_bandwidth") else "N/A",
                "rx_buffer_size": self.sdr.rx_buffer_size,
                "rx_enabled_channels": list(self.sdr.rx_enabled_channels),
                "tx_enabled_channels": list(self.sdr.tx_enabled_channels),
            }
            for i in range(2): # Assuming max 2 channels for PlutoSDR (0 and 1)
                if hasattr(self.sdr, f"gain_control_mode_chan{i}"):
                    props[f"rx_gain_control_mode_chan{i}"] = getattr(self.sdr, f"gain_control_mode_chan{i}")
                if hasattr(self.sdr, f"rx_hardware_gain_chan{i}"):
                    props[f"rx_hardware_gain_chan{i}_db"] = getattr(self.sdr, f"rx_hardware_gain_chan{i}")
                if hasattr(self.sdr, f"tx_hardware_gain_chan{i}"):
                    props[f"tx_attenuation_chan{i}_db"] = getattr(self.sdr, f"tx_hardware_gain_chan{i}")
            return props
        except Exception as e:
            print(f"Error getting SDR properties: {e}")
            return {}

    def close(self):
        """Closes the SDR connection and releases resources."""
        if self.sdr:
            try:
                print("Closing SDR connection...")
                self.stop_tx() # Ensure TX is stopped
                if hasattr(self.sdr, 'rx_destroy_buffer'): # Check if method exists
                    self.sdr.rx_destroy_buffer()
                del self.sdr # Delete the adi.Pluto object instance
                self.sdr = None
                print("SDR connection closed and resources released.")
            except Exception as e:
                print(f"Error closing SDR: {e}")
        else:
            print("SDR connection already closed or was never established.")

if __name__ == '__main__':
    print("SDRController Test Script")
    # Load settings to get sdr_ip, etc.
    # This requires config_loader.py to be accessible.
    # For simplicity in standalone test, we might hardcode or use defaults.
    # from ..config_loader import load_settings # Relative import if run as part of package
    # settings = load_settings()
    # sdr_params = settings.get('sdr_settings', {})
    
    # Using default parameters for the test
    sdr_ip_test = "ip:192.168.2.1" # Change if your PlutoSDR has a different IP
    sample_rate_test = 3e6
    center_freq_test = 2450e6 # 2.45 GHz
    rx_gain_test = 20.0 # dB
    
    print(f"Attempting to initialize SDRController with IP: {sdr_ip_test}")
    sdr_controller = SDRController(
        sdr_ip=sdr_ip_test,
        sample_rate=sample_rate_test,
        center_freq=center_freq_test,
        rx_hardware_gain_chan0=rx_gain_test
    )

    success, message = sdr_controller.connect()
    if success:
        print(f"Connection attempt: {message}")
        print("\nSDR Properties After Connection and Configuration:")
        props = sdr_controller.get_sdr_properties()
        for key, value in props.items(): print(f"  {key}: {value}")

        # Test RX channel activation and data capture
        print("\n--- Testing RX0 Data Capture ---")
        if sdr_controller.set_active_rx_channels([0]):
            # Optionally adjust gain again if needed
            # sdr_controller.set_rx_gain(rx_gain_test + 5, channel=0)
            
            print("Attempting to capture data from RX0...")
            # Capture a small number of samples for test
            captured_data = sdr_controller.capture_data(num_samples=4096) 
            
            if captured_data is not None:
                if isinstance(captured_data, np.ndarray) and captured_data.size > 0:
                    print(f"Captured {len(captured_data)} samples from RX0.")
                    print(f"  Data type: {captured_data.dtype}, Shape: {captured_data.shape}")
                    print(f"  First 5 samples: {captured_data[:5]}")
                    # Example: Calculate and print FFT of the first 1024 samples
                    # fft_result = sdr_controller._calculate_fft(captured_data[:1024])
                    # print(f"  FFT calculated (first few freqs): {fft_result['frequencies'][:5]}")
                else:
                    print("Captured data from RX0 is empty or not a NumPy array.")
            else:
                print("Failed to capture samples from RX0.")
        else:
            print("Failed to set active RX channel to [0].")

        # Test TX (optional, be careful with transmissions)
        # print("\n--- Testing TX0 Data Transmission (Example) ---")
        # if sdr_controller.set_active_tx_channels([0]):
        #     # Create a simple sine wave for transmission
        #     # For Pluto, data must be complex64 and scaled between -1 and 1 (approx)
        #     # Or, more precisely, within the DAC range, often represented by int16 values scaled.
        #     # pyadi-iio handles scaling if you provide float complex data.
        #     fs = sdr_controller.sample_rate
        #     N_tx = 2048
        #     fc_signal = 0.1e6 # 100 kHz offset from LO
        #     t = np.arange(N_tx) / fs
        #     tx_signal_i = 0.5 * np.cos(2 * np.pi * fc_signal * t)
        #     tx_signal_q = 0.5 * np.sin(2 * np.pi * fc_signal * t)
        #     tx_iq_data = (tx_signal_i + 1j * tx_signal_q).astype(np.complex64)
            
        #     print(f"Attempting to transmit {len(tx_iq_data)} samples on TX0...")
        #     if sdr_controller.transmit_data(tx_iq_data, cyclic=True):
        #         print("Transmission started. Sleeping for 2 seconds...")
        #         time.sleep(2)
        #         sdr_controller.stop_tx()
        #         print("Transmission stopped.")
        #     else:
        #         print("Failed to transmit data.")
        # else:
        #     print("Failed to set active TX channel to [0].")

        print("\nClosing SDR connection...")
        sdr_controller.close()
    else:
        print(f"Failed to connect to SDR: {message}")

    print("\nSDRController Test Script Finished.")

