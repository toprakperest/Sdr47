import unittest
import time
import numpy as np
import os

# Attempt to import actual hardware control modules
# These tests are designed to be run on the actual hardware environment
# where adi (pyadi-iio) and other necessary drivers are installed and configured.

# Adjust import paths based on the project structure
# Assuming this test script is in ground_radar/tests/
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from hardware.sdr_controller import SDRController
    from hardware.thermal_monitor import ThermalMonitor
    # from hardware.antenna_manager import AntennaManager # If specific tests are needed
    HARDWARE_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Hardware modules not found, tests will be skipped or will fail: {e}")
    HARDWARE_MODULES_AVAILABLE = False
    # Define mock classes if real ones can't be imported, to allow script to parse
    # However, tests requiring these will be skipped.
    class SDRController:
        def __init__(self, ip_address=None, sdr_type='ad9363'): pass
        def connect(self): return False
        def configure_sdr(self, center_freq, sample_rate, rx_buffer_size, rf_bandwidth, gain_mode_ch0, gain_ch0, tx_gain_ch0): pass
        def capture_data(self, num_samples, channels=[0]): return np.array([])
        def get_sdr_temperature(self): return None # AD9361/3 has temp sensor
        def disconnect(self): pass
        @property
        def is_connected(self): return False

    class ThermalMonitor:
        def __init__(self, sdr_controller_ref): pass
        def get_temperature(self): return None
        def start_monitoring(self, interval=5): pass
        def stop_monitoring(self): pass

@unittest.skipIf(not HARDWARE_MODULES_AVAILABLE, "Hardware modules (e.g., SDRController, pyadi-iio) not available. Skipping hardware tests.")
class TestHardwareIntegration(unittest.TestCase):

    SDR_IP = "ip:192.168.2.1" # Default PlutoSDR IP, should be configured in settings.yaml

    @classmethod
    def setUpClass(cls):
        """Set up for all tests in this class."""
        cls.sdr_controller = SDRController(ip_address=cls.SDR_IP, sdr_type='ad9363')
        cls.thermal_monitor = ThermalMonitor(sdr_controller_ref=cls.sdr_controller)
        # cls.antenna_manager = AntennaManager(sdr_controller_ref=cls.sdr_controller) # If needed

        print(f"Attempting to connect to SDR at {cls.SDR_IP} for hardware tests...")
        if not cls.sdr_controller.connect():
            print(f"Failed to connect to SDR at {cls.SDR_IP}. Hardware tests will likely fail or be limited.")
            # We might choose to raise an exception here to stop tests if connection is critical
            # raise ConnectionError(f"Could not connect to SDR at {cls.SDR_IP} for testing.")
        else:
            print("SDR connected successfully.")

    @classmethod
    def tearDownClass(cls):
        """Tear down after all tests in this class."""
        if cls.sdr_controller and cls.sdr_controller.is_connected:
            print("Disconnecting SDR...")
            cls.sdr_controller.disconnect()

    def test_01_sdr_connection_status(self):
        """Test if the SDR controller reports a connected state."""
        self.assertTrue(self.sdr_controller.is_connected, f"SDR should be connected at {self.SDR_IP}")

    def test_02_sdr_configuration(self):
        """Test basic SDR configuration."""
        if not self.sdr_controller.is_connected:
            self.skipTest("SDR not connected, skipping configuration test.")
        
        center_freq = 200e6  # 200 MHz
        sample_rate = 2e6    # 2 MSps
        rx_buffer_size = 1024
        rf_bandwidth = 1.5e6 # 1.5 MHz
        gain_mode = "manual"
        gain_val = 30        # dB
        tx_gain = -10        # dB

        try:
            self.sdr_controller.configure_sdr(
                center_freq=center_freq,
                sample_rate=sample_rate,
                rx_buffer_size=rx_buffer_size,
                rf_bandwidth=rf_bandwidth,
                gain_mode_ch0=gain_mode,
                gain_ch0=gain_val,
                tx_gain_ch0=tx_gain
            )
            # Verify some readable parameters if possible (pyadi-iio specific)
            self.assertEqual(self.sdr_controller.sdr.rx_lo, int(center_freq))
            self.assertEqual(self.sdr_controller.sdr.sample_rate, int(sample_rate))
            self.assertEqual(self.sdr_controller.sdr.rx_rf_bandwidth, int(rf_bandwidth))
            self.assertEqual(self.sdr_controller.sdr.rx_hardwaregain_chan0, gain_val)
            self.assertEqual(self.sdr_controller.sdr.tx_hardwaregain_chan0, tx_gain)

        except Exception as e:
            self.fail(f"SDR configuration failed: {e}")

    def test_03_sdr_data_capture_rx1(self):
        """Test data capture from SDR RX1."""
        if not self.sdr_controller.is_connected:
            self.skipTest("SDR not connected, skipping data capture test.")
        
        num_samples = 2048
        try:
            # Ensure RX1 (channel 0) is active for capture
            self.sdr_controller.set_active_rx_channels([0])
            data = self.sdr_controller.capture_data(num_samples=num_samples, channels=[0])
            self.assertIsNotNone(data, "Captured data should not be None.")
            self.assertEqual(len(data), num_samples, f"Captured data length mismatch. Expected {num_samples}, got {len(data)}.")
            self.assertTrue(isinstance(data, np.ndarray), "Captured data should be a NumPy array.")
            # Basic check for non-zero data (might be all zeros in a quiet environment or if TX is off)
            # print(f"Sample of captured RX1 data: {data[:5]}")
        except Exception as e:
            self.fail(f"SDR RX1 data capture failed: {e}")

    def test_04_sdr_data_capture_rx2(self):
        """Test data capture from SDR RX2 (if AD9363/Pluto supports dual RX easily)."""
        if not self.sdr_controller.is_connected:
            self.skipTest("SDR not connected, skipping RX2 data capture test.")
        if self.sdr_controller.sdr_type != 'ad9363' and len(self.sdr_controller.sdr.rx_enabled_channels) < 2:
             self.skipTest("SDR does not support easily switchable dual RX or RX2 is not enabled by default for this test.")

        num_samples = 2048
        try:
            # Ensure RX2 (channel 1) is active for capture
            self.sdr_controller.set_active_rx_channels([1]) 
            data = self.sdr_controller.capture_data(num_samples=num_samples, channels=[1])
            self.assertIsNotNone(data, "Captured RX2 data should not be None.")
            self.assertEqual(len(data), num_samples, f"Captured RX2 data length mismatch. Expected {num_samples}, got {len(data)}.")
            self.assertTrue(isinstance(data, np.ndarray), "Captured RX2 data should be a NumPy array.")
            # print(f"Sample of captured RX2 data: {data[:5]}")
        except Exception as e:
            self.fail(f"SDR RX2 data capture failed: {e}")
        finally:
            # Revert to default (RX1) if needed for other tests
            self.sdr_controller.set_active_rx_channels([0])

    def test_05_thermal_monitor_temperature_reading(self):
        """Test reading temperature from the SDR's thermal sensor via ThermalMonitor."""
        if not self.sdr_controller.is_connected:
            self.skipTest("SDR not connected, skipping temperature reading test.")
        
        try:
            temperature = self.thermal_monitor.get_temperature()
            self.assertIsNotNone(temperature, "Temperature reading should not be None.")
            self.assertTrue(isinstance(temperature, (int, float)), "Temperature should be a number.")
            # Typical operating range, e.g., 0 to 100 Celsius. AD9361 sensor is often in C.
            self.assertTrue(0 < temperature < 100, f"Temperature reading {temperature}°C seems out of typical range.")
            print(f"SDR Temperature: {temperature}°C")
        except Exception as e:
            # Some PlutoSDRs might not expose temp sensor easily or it might not be implemented in sdr_controller
            self.fail(f"Temperature reading failed: {e}. This might be an expected failure if sensor is not available/implemented.")

    # Add more tests for AntennaManager if it has testable software-controlled functions
    # def test_06_antenna_switching(self):
    #     if not self.sdr_controller.is_connected:
    #         self.skipTest("SDR not connected, skipping antenna switching test.")
    #     try:
    #         # Assuming AntennaManager has methods like switch_to_tx1_rx1, switch_to_rx2
    #         # self.antenna_manager.set_mode("TX1_RX1")
    #         # Add assertions to check if switching was successful (e.g., by checking SDR registers if possible)
    #         pass
    #     except Exception as e:
    #         self.fail(f"Antenna switching test failed: {e}")

if __name__ == '__main__':
    print("Starting hardware integration tests...")
    print("IMPORTANT: These tests are designed to run on the actual hardware (e.g., Zynq7020 + AD9363). Ensure the SDR is connected and accessible.")
    print("If run in a simulated environment without actual hardware, they will likely be skipped or fail.")
    
    # Create the tests directory if it doesn't exist
    test_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        
    unittest.main(verbosity=2)

