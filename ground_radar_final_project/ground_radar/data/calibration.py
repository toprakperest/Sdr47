import numpy as np
import json
import os
import time

# Assuming SDRController might be needed to trigger calibration captures
from hardware.sdr_controller import SDRController

class CalibrationData:
    """
    Manages calibration data for the GPR system.
    This includes ground calibration using RX2, noise profiles, and potentially
    antenna calibration parameters if they are determined dynamically or stored.
    """
    def __init__(self, sdr_controller_ref=None, calibration_dir="./calibration_data"):
        """
        Initializes the CalibrationData module.

        Args:
            sdr_controller_ref (SDRController, optional): Reference to the SDRController 
                                                        for active calibration routines.
            calibration_dir (str): Directory to store calibration files.
        """
        self.sdr_controller = sdr_controller_ref
        self.calibration_dir = calibration_dir
        if not os.path.isabs(self.calibration_dir):
            # Make path relative to this file if it's not absolute
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.calibration_dir = os.path.join(base_dir, self.calibration_dir)
        
        os.makedirs(self.calibration_dir, exist_ok=True)
        print(f"CalibrationData initialized. Data will be stored in: {self.calibration_dir}")

        self.ground_calibration_profile = None # Stores data from RX2 ground listening
        self.noise_profile_rx1 = None # Ambient noise for RX1
        self.noise_profile_rx2 = None # Ambient noise for RX2
        self.antenna_patterns = {} # Placeholder for antenna calibration data

    def perform_rx2_ground_calibration(self, duration_sec=10, num_samples_per_capture=2048):
        """
        Performs ground calibration by listening with RX2 to characterize
        the soil/terrain properties (e.g., average dielectric constant, dominant reflections).
        This requires the SDRController to be active and configured for RX2.

        Args:
            duration_sec (int): How long to capture data for averaging.
            num_samples_per_capture (int): Samples per individual SDR read.

        Returns:
            dict: A profile characterizing the ground, or None if failed.
        """
        if not self.sdr_controller or not self.sdr_controller.sdr:
            print("RX2 Ground Calib Error: SDRController not available or not connected.")
            return None

        print(f"Starting RX2 ground calibration for {duration_sec} seconds...")
        # Ensure RX2 is the active channel (AntennaManager should handle this based on mode)
        # For this direct call, we might need to instruct SDRController or assume it's set.
        # Example: self.sdr_controller.set_active_rx_channels([1]) # Assuming RX2 is channel 1
        # This should ideally be coordinated by a higher-level process that sets scan_mode.

        all_captures = []
        start_time = time.time()
        try:
            original_rx_channels = list(self.sdr_controller.sdr.rx_enabled_channels)
            self.sdr_controller.set_active_rx_channels([1]) # Explicitly set RX2 (channel 1)

            while time.time() - start_time < duration_sec:
                # data_rx2 = self.sdr_controller.capture_data(num_samples=num_samples_per_capture)
                # For testing without live SDR, use dummy data
                data_rx2 = (np.random.randn(num_samples_per_capture) + 1j * np.random.randn(num_samples_per_capture)) * 0.1
                if isinstance(data_rx2, np.ndarray) and data_rx2.size > 0:
                    all_captures.append(np.abs(data_rx2)) # Store magnitude or I/Q as needed
                time.sleep(0.1) # Small delay between captures
            
            self.sdr_controller.set_active_rx_channels(original_rx_channels) # Restore original channels

        except Exception as e:
            print(f"Error during RX2 ground calibration capture: {e}")
            if hasattr(self.sdr_controller, 'sdr') and self.sdr_controller.sdr: # Check if sdr object exists before trying to access its attributes. 
                 self.sdr_controller.set_active_rx_channels(original_rx_channels) # Ensure restoration
            return None

        if not all_captures:
            print("RX2 Ground Calib Error: No data captured.")
            return None

        # Process the captured data (e.g., average, FFT, identify dominant features)
        avg_spectrum = np.mean([np.abs(np.fft.fft(cap)) for cap in all_captures], axis=0)
        avg_time_domain = np.mean(all_captures, axis=0)

        self.ground_calibration_profile = {
            "timestamp": time.time(),
            "duration_sec": duration_sec,
            "num_traces_averaged": len(all_captures),
            "average_time_domain_envelope": avg_time_domain.tolist(), # Store as list for JSON
            "average_frequency_spectrum": avg_spectrum.tolist(),
            "estimated_terrain_type": self._estimate_terrain_from_profile(avg_time_domain, avg_spectrum)
        }
        self.save_calibration_data("ground_profile_rx2.json", self.ground_calibration_profile)
        print("RX2 ground calibration complete. Profile saved.")
        return self.ground_calibration_profile

    def _estimate_terrain_from_profile(self, time_domain_env, freq_spectrum):
        """Placeholder: Estimates terrain type based on RX2 profile."""
        # This would involve more complex logic, potentially using AI or rule-based systems
        # based on known GPR responses of different materials (from GeologyDB).
        # For example, high attenuation + specific dielectric signature -> Clay
        # Low attenuation, clear reflections -> Limestone or dry sand
        if np.mean(time_domain_env) > 0.5: # Arbitrary threshold for strong reflections
            return "Kalker_Yogun_Tahmini" # Example
        else:
            return "Kil_Marn_Tahmini"

    def measure_noise_profile(self, channel=0, duration_sec=5, num_samples_per_capture=2048):
        """
        Measures ambient noise profile for a given RX channel.
        The antenna should ideally be terminated or pointed to a known quiet source.
        """
        if not self.sdr_controller or not self.sdr_controller.sdr:
            print(f"Noise Profile Error (CH{channel}): SDRController not available.")
            return None
        print(f"Measuring noise profile for RX Channel {channel} for {duration_sec}s...")
        # Ensure correct channel is active
        # self.sdr_controller.set_active_rx_channels([channel])

        all_noise_captures = []
        start_time = time.time()
        try:
            original_rx_channels = list(self.sdr_controller.sdr.rx_enabled_channels)
            self.sdr_controller.set_active_rx_channels([channel])

            while time.time() - start_time < duration_sec:
                # noise_data = self.sdr_controller.capture_data(num_samples=num_samples_per_capture)
                noise_data = (np.random.randn(num_samples_per_capture) + 1j*np.random.randn(num_samples_per_capture)) * 0.01 # Dummy noise
                if isinstance(noise_data, np.ndarray) and noise_data.size > 0:
                    all_noise_captures.append(noise_data)
                time.sleep(0.1)
            self.sdr_controller.set_active_rx_channels(original_rx_channels)
        except Exception as e:
            print(f"Error during noise capture (CH{channel}): {e}")
            if hasattr(self.sdr_controller, 'sdr') and self.sdr_controller.sdr:
                 self.sdr_controller.set_active_rx_channels(original_rx_channels)
            return None

        if not all_noise_captures:
            print(f"Noise Profile Error (CH{channel}): No data captured.")
            return None

        avg_noise_power_spectrum = np.mean([np.abs(np.fft.fft(cap))**2 for cap in all_noise_captures], axis=0)
        noise_profile = {
            "timestamp": time.time(),
            "channel": channel,
            "duration_sec": duration_sec,
            "average_power_spectrum_dB": (10 * np.log10(avg_noise_power_spectrum + 1e-12)).tolist() # Avoid log(0)
        }
        
        filename = f"noise_profile_rx{channel}.json"
        if channel == 0:
            self.noise_profile_rx1 = noise_profile
        elif channel == 1:
            self.noise_profile_rx2 = noise_profile
        self.save_calibration_data(filename, noise_profile)
        print(f"Noise profile for RX Channel {channel} measured and saved.")
        return noise_profile

    def save_calibration_data(self, filename, data):
        filepath = os.path.join(self.calibration_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Calibration data saved to {filepath}")
        except Exception as e:
            print(f"Error saving calibration data to {filepath}: {e}")

    def load_calibration_data(self, filename):
        filepath = os.path.join(self.calibration_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"Loaded calibration data from {filepath}")
                # Update internal state if it's a known profile type
                if filename == "ground_profile_rx2.json": self.ground_calibration_profile = data
                elif filename == "noise_profile_rx0.json": self.noise_profile_rx1 = data
                elif filename == "noise_profile_rx2.json": self.noise_profile_rx2 = data
                return data
            except Exception as e:
                print(f"Error loading calibration data from {filepath}: {e}")
        return None

    def get_ground_profile(self):
        if not self.ground_calibration_profile:
            return self.load_calibration_data("ground_profile_rx2.json")
        return self.ground_calibration_profile

    def get_noise_profile(self, channel=0):
        if channel == 0:
            if not self.noise_profile_rx1:
                return self.load_calibration_data("noise_profile_rx0.json")
            return self.noise_profile_rx1
        elif channel == 1:
            if not self.noise_profile_rx2:
                return self.load_calibration_data("noise_profile_rx2.json")
            return self.noise_profile_rx2
        return None

# Example Usage (for testing, comment out when integrating):
if __name__ == "__main__":
    # Mock SDRController for testing
    class MockSDRController:
        def __init__(self):
            self.sdr = True # Simulate connected SDR
            self.rx_enabled_channels = [0] # Default
            print("MockSDRController initialized for Calibration testing.")
        def set_active_rx_channels(self, channels):
            self.rx_enabled_channels = channels
            print(f"MockSDR: Active RX channels set to {channels}")
        def capture_data(self, num_samples):
            # Simulate some data with a slight DC offset and noise
            return (np.random.randn(num_samples) + 1j*np.random.randn(num_samples)) * 0.05 + 0.01 

    mock_sdr = MockSDRController()
    
    # Create a temporary directory for calibration files for this test
    test_cal_dir = "./temp_calibration_data_test"
    if os.path.exists(test_cal_dir):
        import shutil
        shutil.rmtree(test_cal_dir) # Clean up from previous test

    cal_manager = CalibrationData(sdr_controller_ref=mock_sdr, calibration_dir=test_cal_dir)

    print("\n--- Testing RX2 Ground Calibration ---")
    ground_prof = cal_manager.perform_rx2_ground_calibration(duration_sec=2) # Short duration for test
    if ground_prof:
        print(f"Ground Profile Estimated Terrain: {ground_prof.get('estimated_terrain_type')}")

    print("\n--- Testing Noise Profile Measurement (RX0) ---")
    noise_prof0 = cal_manager.measure_noise_profile(channel=0, duration_sec=1)
    if noise_prof0:
        print(f"Noise Profile RX0 (first 10 spectrum values): {noise_prof0['average_power_spectrum_dB'][:10]}")

    print("\n--- Testing Noise Profile Measurement (RX1) ---")
    noise_prof1 = cal_manager.measure_noise_profile(channel=1, duration_sec=1)
    if noise_prof1:
        print(f"Noise Profile RX1 (first 10 spectrum values): {noise_prof1['average_power_spectrum_dB'][:10]}")

    print("\n--- Testing Loading Calibration Data ---")
    loaded_ground_prof = cal_manager.load_calibration_data("ground_profile_rx2.json")
    if loaded_ground_prof:
        print(f"Successfully loaded ground profile. Timestamp: {loaded_ground_prof['timestamp']}")
    
    loaded_noise_prof0 = cal_manager.get_noise_profile(0)
    if loaded_noise_prof0:
         print(f"Successfully loaded noise profile RX0. Channel: {loaded_noise_prof0['channel']}")

    print("\n--- CalibrationData tests complete ---")
    # Clean up test directory
    # import shutil
    # if os.path.exists(test_cal_dir):
    #     shutil.rmtree(test_cal_dir)

