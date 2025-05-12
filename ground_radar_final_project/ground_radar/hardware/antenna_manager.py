import numpy as np
# Ensure the SDRController is imported correctly based on its location
# If sdr_controller.py is in the same directory (hardware):
from .sdr_controller import SDRController
# If sdr_controller.py is in the parent directory (less common for modules):
# from ..sdr_controller import SDRController 

class AntennaManager:
    """
    Manages antenna configurations for the ground radar system.
    It controls which antennas are active for transmission and reception
    by interacting with the SDRController to enable/disable specific SDR channels
    mapped to these antennas.
    This class is designed for real hardware interaction.
    
    Antenna Mapping (Assumed Example - verify with actual hardware):
    - TX1 (e.g., UWB Directional for transmit): Mapped to SDR TX Channel 0
    - RX1 (e.g., UWB Directional for receive): Mapped to SDR RX Channel 0
    - RX2 (e.g., Telescopic for calibration/diversity): Mapped to SDR RX Channel 1 (listen-only or active)
    """
    def __init__(self, sdr_controller: SDRController):
        """
        Initializes the AntennaManager.

        Args:
            sdr_controller (SDRController): An instance of the SDRController.
                                            The SDRController should be connected before using most AntennaManager methods.
        """
        if not isinstance(sdr_controller, SDRController):
            raise ValueError("sdr_controller must be an instance of SDRController.")
        self.sdr_controller = sdr_controller
        self.current_tx_antenna_name = None # e.g., "TX1"
        self.current_rx_antenna_names = []  # e.g., ["RX1", "RX2"]
        
        # Define antenna to SDR channel mapping (example, adjust as per hardware)
        self.antenna_to_sdr_tx_channel = {
            "TX1": 0,
            # "TX2": 1 # if another TX antenna exists and is mapped to SDR TX1
        }
        self.antenna_to_sdr_rx_channel = {
            "RX1": 0,
            "RX2": 1,
        }
        print("AntennaManager initialized. Ensure SDR is connected before configuring antennas.")

    def _set_sdr_tx_channels(self, sdr_channel_indices: list[int]):
        """Helper to activate specific SDR TX channels via SDRController."""
        if not self.sdr_controller.sdr: # Check if SDR is connected
            print("AntennaManager: SDR not connected. Cannot set TX channels.")
            return False
        return self.sdr_controller.set_active_tx_channels(sdr_channel_indices)

    def _set_sdr_rx_channels(self, sdr_channel_indices: list[int]):
        """Helper to activate specific SDR RX channels via SDRController."""
        if not self.sdr_controller.sdr: # Check if SDR is connected
            print("AntennaManager: SDR not connected. Cannot set RX channels.")
            return False
        return self.sdr_controller.set_active_rx_channels(sdr_channel_indices)

    def configure_antennas(self, tx_antenna_name: str = None, rx_antenna_names: list[str] = None):
        """
        Configures the specified transmit and receive antennas.
        Deactivates all other antennas/channels first for a clean state.

        Args:
            tx_antenna_name (str, optional): Name of the TX antenna to activate (e.g., "TX1"). 
                                             If None, all TX channels are deactivated.
            rx_antenna_names (list[str], optional): List of RX antenna names to activate (e.g., ["RX1", "RX2"]).
                                                  If None or empty, all RX channels are deactivated.
        Returns:
            bool: True if configuration was attempted successfully (even if channels don't exist), 
                  False if SDR is not connected.
        """
        if not self.sdr_controller.sdr:
            print("AntennaManager: SDR not connected. Cannot configure antennas.")
            self.current_tx_antenna_name = None
            self.current_rx_antenna_names = []
            return False

        print(f"Configuring antennas: TX='{tx_antenna_name}', RX={rx_antenna_names}")

        # Deactivate all channels first
        self._set_sdr_tx_channels([])
        self._set_sdr_rx_channels([])

        active_sdr_tx_channels = []
        if tx_antenna_name:
            if tx_antenna_name in self.antenna_to_sdr_tx_channel:
                active_sdr_tx_channels.append(self.antenna_to_sdr_tx_channel[tx_antenna_name])
            else:
                print(f"Warning: TX Antenna '{tx_antenna_name}' not defined in mapping.")
        
        if active_sdr_tx_channels:
            self._set_sdr_tx_channels(active_sdr_tx_channels)
            self.current_tx_antenna_name = tx_antenna_name
        else:
            self.current_tx_antenna_name = None

        active_sdr_rx_channels = []
        if rx_antenna_names:
            for name in rx_antenna_names:
                if name in self.antenna_to_sdr_rx_channel:
                    active_sdr_rx_channels.append(self.antenna_to_sdr_rx_channel[name])
                else:
                    print(f"Warning: RX Antenna '{name}' not defined in mapping.")
        
        if active_sdr_rx_channels:
            # Remove duplicates if any, though mapping should be unique per antenna name
            self._set_sdr_rx_channels(list(set(active_sdr_rx_channels)))
            self.current_rx_antenna_names = rx_antenna_names # Store names, not channels
        else:
            self.current_rx_antenna_names = []
            
        print(f"Antenna configuration complete: Active TX SDR Channels: {self.sdr_controller.sdr.tx_enabled_channels}, Active RX SDR Channels: {self.sdr_controller.sdr.rx_enabled_channels}")
        return True

    def configure_for_main_scan(self):
        """
        Configures antennas for the main scanning operation (e.g., TX1 and RX1 active).
        Assumes TX1 -> SDR TX0, RX1 -> SDR RX0.
        """
        print("AntennaManager: Configuring for Main Scan (e.g., TX1, RX1)...")
        return self.configure_antennas(tx_antenna_name="TX1", rx_antenna_names=["RX1"])

    def configure_for_calibration_rx2_only(self):
        """
        Configures antennas for calibration using only RX2 (listen-only mode).
        Assumes RX2 -> SDR RX1. All TX channels are deactivated.
        """
        print("AntennaManager: Configuring for Calibration (RX2 listen-only)...")
        return self.configure_antennas(tx_antenna_name=None, rx_antenna_names=["RX2"])
        
    def configure_for_hybrid_scan_tx1_rx1_rx2(self):
        """
        Configures for a hybrid scan: TX1 transmits, RX1 and RX2 receive simultaneously.
        Assumes TX1->SDR TX0, RX1->SDR RX0, RX2->SDR RX1.
        """
        print("AntennaManager: Configuring for Hybrid Scan (TX1, RX1, RX2)...")
        return self.configure_antennas(tx_antenna_name="TX1", rx_antenna_names=["RX1", "RX2"])

    def set_tx_antenna_power(self, antenna_name: str, power_level_db: float):
        """
        Sets the transmission power/attenuation for a specific TX antenna.
        The `power_level_db` for PlutoSDR's `tx_hardware_gain_chanX` is attenuation.
        0 dB is maximum power, more negative values (e.g., -10 dB) mean less power (more attenuation).
        Valid range for PlutoSDR is typically 0.0 dB to -89.75 dB.

        Args:
            antenna_name (str): The name of the TX antenna (e.g., "TX1").
            power_level_db (float): Desired SDR attenuation value in dB. 
                                     Example: 0 for max power, -20 for 20dB attenuation.
        Returns:
            bool: True if setting was attempted, False if SDR not connected or antenna unknown.
        """
        if not self.sdr_controller.sdr:
            print(f"AntennaManager: SDR not connected. Cannot set TX power for {antenna_name}.")
            return False

        if antenna_name not in self.antenna_to_sdr_tx_channel:
            print(f"AntennaManager: Unknown TX antenna '{antenna_name}'. Cannot set power.")
            return False
        
        sdr_tx_channel = self.antenna_to_sdr_tx_channel[antenna_name]
        print(f"AntennaManager: Setting TX power for {antenna_name} (SDR CH{sdr_tx_channel}) to {power_level_db} dB attenuation.")
        # The SDRController's set_tx_attenuation handles clamping to valid range.
        return self.sdr_controller.set_tx_attenuation(attenuation_db=power_level_db, channel=sdr_tx_channel)

    def set_rx_antenna_gain(self, antenna_name: str, gain_db: float, mode: str = "manual"):
        """
        Sets the reception gain for a specific RX antenna.

        Args:
            antenna_name (str): The name of the RX antenna (e.g., "RX1", "RX2").
            gain_db (float): Desired gain in dB for manual mode.
            mode (str): SDR gain control mode (e.g., "manual", "slow_attack", "fast_attack").
        Returns:
            bool: True if setting was attempted, False if SDR not connected or antenna unknown.
        """
        if not self.sdr_controller.sdr:
            print(f"AntennaManager: SDR not connected. Cannot set RX gain for {antenna_name}.")
            return False

        if antenna_name not in self.antenna_to_sdr_rx_channel:
            print(f"AntennaManager: Unknown RX antenna '{antenna_name}'. Cannot set gain.")
            return False
            
        sdr_rx_channel = self.antenna_to_sdr_rx_channel[antenna_name]
        print(f"AntennaManager: Setting RX gain for {antenna_name} (SDR CH{sdr_rx_channel}) to {gain_db} dB, mode: {mode}")
        return self.sdr_controller.set_rx_gain(gain_db=gain_db, channel=sdr_rx_channel, mode=mode)

    def get_active_config_display(self):
        """Returns a human-readable string of the current active antenna configuration."""
        if not self.sdr_controller.sdr:
            return "SDR not connected. Antenna configuration unknown."
        
        tx_ant = self.current_tx_antenna_name if self.current_tx_antenna_name else "None"
        rx_ants = ", ".join(self.current_rx_antenna_names) if self.current_rx_antenna_names else "None"
        sdr_tx_ch = list(self.sdr_controller.sdr.tx_enabled_channels)
        sdr_rx_ch = list(self.sdr_controller.sdr.rx_enabled_channels)
        
        return (f"Active TX Antenna: {tx_ant} (SDR CH: {sdr_tx_ch})\n"
                f"Active RX Antennas: {rx_ants} (SDR CH: {sdr_rx_ch})")

    def get_active_config_dict(self):
        """Returns a dictionary of the current active antenna configuration."""
        if not self.sdr_controller.sdr:
            return {
                "tx_antenna_name": None,
                "rx_antenna_names": [],
                "sdr_tx_channels_active": [],
                "sdr_rx_channels_active": []
            }
        return {
            "tx_antenna_name": self.current_tx_antenna_name,
            "rx_antenna_names": list(self.current_rx_antenna_names),
            "sdr_tx_channels_active": list(self.sdr_controller.sdr.tx_enabled_channels),
            "sdr_rx_channels_active": list(self.sdr_controller.sdr.rx_enabled_channels)
        }

# Example Usage (for testing, comment out or guard when integrating):
if __name__ == '__main__':
    print("AntennaManager Test Script")
    
    # This test requires a running SDR or a MockSDRController that behaves like the real one.
    # For this example, we assume sdr_controller.py (with SDRController) is accessible.
    # And we will use the actual SDRController, so an SDR should be connected and configured (e.g. PlutoSDR at ip:192.168.2.1)
    
    # --- Option 1: Use actual SDRController (requires SDR hardware) ---
    sdr_ip_test = "ip:192.168.2.1" # Ensure this IP is correct for your SDR
    print(f"Initializing actual SDRController for test (IP: {sdr_ip_test})...")
    # Use default parameters from SDRController's __init__ for simplicity here
    # In a real app, these would come from a config file.
    test_sdr_controller = SDRController(sdr_ip=sdr_ip_test, sample_rate=2e6, center_freq=100e6)
    connected, msg = test_sdr_controller.connect()

    if not connected:
        print(f"SDR Connection Failed: {msg}")
        print("AntennaManager tests cannot proceed without a connected SDR. Exiting test.")
    else:
        print(f"SDR Connection Successful: {msg}")
        antenna_mgr = AntennaManager(sdr_controller=test_sdr_controller)

        print("\n--- Initial Antenna Configuration --- ")
        print(antenna_mgr.get_active_config_display())

        print("\n--- Testing Main Scan Configuration (TX1, RX1) ---")
        antenna_mgr.configure_for_main_scan()
        print(antenna_mgr.get_active_config_display())
        # Set power for TX1 (mapped to SDR TX0)
        antenna_mgr.set_tx_antenna_power(antenna_name="TX1", power_level_db=-10.0) # -10dB attenuation
        # Set gain for RX1 (mapped to SDR RX0)
        antenna_mgr.set_rx_antenna_gain(antenna_name="RX1", gain_db=30.0, mode="manual")
        # Try capturing some data
        data = test_sdr_controller.capture_data(1024)
        if data is not None:
            print(f"Main Scan: Captured data type: {type(data)}, length: {len(data) if isinstance(data, np.ndarray) else 'N/A for list'}")
        else:
            print("Main Scan: Failed to capture data.")

        print("\n--- Testing Calibration RX2 Listen-Only Configuration ---")
        antenna_mgr.configure_for_calibration_rx2_only()
        print(antenna_mgr.get_active_config_display())
        antenna_mgr.set_rx_antenna_gain(antenna_name="RX2", gain_db=35.0, mode="slow_attack")
        data = test_sdr_controller.capture_data(1024)
        if data is not None:
            print(f"Calibration Scan: Captured data type: {type(data)}, length: {len(data) if isinstance(data, np.ndarray) else 'N/A for list'}")
        else:
            print("Calibration Scan: Failed to capture data.")
        
        print("\n--- Testing Hybrid Scan Configuration (TX1, RX1, RX2) ---")
        antenna_mgr.configure_for_hybrid_scan_tx1_rx1_rx2()
        print(antenna_mgr.get_active_config_display())
        antenna_mgr.set_tx_antenna_power(antenna_name="TX1", power_level_db=0.0) # Max power
        antenna_mgr.set_rx_antenna_gain(antenna_name="RX1", gain_db=25.0)
        antenna_mgr.set_rx_antenna_gain(antenna_name="RX2", gain_db=28.0)
        # Capturing from multiple RX channels will return a list of arrays
        multi_data = test_sdr_controller.capture_data(1024)
        if multi_data is not None and isinstance(multi_data, list) and len(multi_data) == 2:
            print(f"Hybrid Scan: Captured data from {len(multi_data)} channels.")
            print(f"  RX0 (RX1) data length: {len(multi_data[0])}")
            print(f"  RX1 (RX2) data length: {len(multi_data[1])}")
        elif multi_data is not None: # If it's not a list of 2, something is unexpected
             print(f"Hybrid Scan: Captured data, but format is unexpected: {type(multi_data)}")
        else:
            print("Hybrid Scan: Failed to capture data.")

        print("\n--- Deactivating all antennas ---")
        antenna_mgr.configure_antennas(None, None)
        print(antenna_mgr.get_active_config_display())

        print("\nClosing SDR connection...")
        test_sdr_controller.close()

    print("\nAntennaManager Test Script Finished.")

