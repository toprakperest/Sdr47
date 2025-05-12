import time
from sdr_controller import SDRController

class ThermalMonitor:
    """
    Monitors the temperature of the AD9363 SDR chip.
    It accesses the temperature sensor data through the pyadi-iio library,
    via the SDRController instance.
    This class is designed for real hardware interaction.
    """
    def __init__(self, sdr_controller: SDRController, update_interval_sec=5):
        """
        Initializes the ThermalMonitor.

        Args:
            sdr_controller (SDRController): An instance of the SDRController.
            update_interval_sec (int): How often to attempt to read temperature, in seconds.
        """
        if not isinstance(sdr_controller, SDRController):
            raise ValueError("sdr_controller must be an instance of SDRController")
        self.sdr_controller = sdr_controller
        self.update_interval = update_interval_sec
        self.current_temperature_c = None
        self._phy_device = None # To store the IIO device for temperature reading
        print("ThermalMonitor initialized.")
        self._find_phy_device()

    def _find_phy_device(self):
        """Attempts to find the IIO device associated with the AD936x phy for temperature readings."""
        if not self.sdr_controller or not self.sdr_controller.sdr:
            print("ThermalMonitor: SDR Controller or SDR device not available.")
            return

        try:
            # The AD936x phy device is usually accessible via sdr.sdr.phy in pyadi-iio
            # For PlutoSDR, sdr.sdr is the adi.Pluto object, and sdr.sdr.phy is the ad9361 device context
            if hasattr(self.sdr_controller.sdr, 'phy'):
                self._phy_device = self.sdr_controller.sdr.phy
                print(f"ThermalMonitor: Found phy device: {self._phy_device.name}")
            else:
                # Fallback: Search for a device that might contain temperature attributes
                # Common names for AD936x phy are 'ad9361-phy' or similar.
                # The temperature sensor might also be a separate IIO device.
                target_device_name = "ad9361-phy" # Common name for AD936x phy
                if self.sdr_controller.sdr.ctx:
                    for dev in self.sdr_controller.sdr.ctx.devices:
                        if dev.name == target_device_name:
                            self._phy_device = dev
                            print(f"ThermalMonitor: Found phy device by name: {dev.name}")
                            break
                        # Some platforms might expose temp sensor under a different device
                        # Check for channels named 'temp' or similar
                        for ch in dev.channels:
                            if "temp" in ch.id.lower() or "temp" in (ch.name or "").lower():
                                self._phy_device = dev # Assume temp sensor is on this device
                                print(f"ThermalMonitor: Found device with temperature channel: {dev.name}")
                                break
                        if self._phy_device:
                            break 
                if not self._phy_device:
                    print(f"ThermalMonitor: Could not automatically find AD936x phy device named '{target_device_name}' or a temperature sensor device.")
                    print("ThermalMonitor: Temperature monitoring might not be available or requires specific device name.")

        except Exception as e:
            print(f"ThermalMonitor: Error finding phy device: {e}")
            self._phy_device = None

    def get_temperature(self):
        """
        Reads the temperature from the AD9363 chip.
        The AD9363 has an internal temperature sensor.
        The IIO attribute is typically 'temp0_input' or similar for the ad9361-phy device.
        Returns:
            float: Temperature in degrees Celsius, or None if an error occurs or not supported.
        """
        if not self._phy_device:
            # Try to find it again if not found during init (e.g. SDR connected later)
            self._find_phy_device()
            if not self._phy_device:
                print("ThermalMonitor: Phy device not available. Cannot read temperature.")
                return None

        temp_attr_names = ["temp0_input", "temp_input", "in_temp_input"] # Common IIO attribute names for temperature
        
        temperature_value = None
        found_attr_name = None

        for attr_name in temp_attr_names:
            if hasattr(self._phy_device, attr_name):
                found_attr_name = attr_name
                break
        
        if found_attr_name:
            try:
                # Temperature is usually reported in millidegrees Celsius by the IIO driver
                raw_temp = int(getattr(self._phy_device, found_attr_name))
                self.current_temperature_c = raw_temp / 1000.0
                # print(f"ThermalMonitor: Raw temp ({found_attr_name}): {raw_temp}, Celsius: {self.current_temperature_c:.2f}°C")
                return self.current_temperature_c
            except Exception as e:
                print(f"ThermalMonitor: Error reading temperature attribute '{found_attr_name}': {e}")
                self.current_temperature_c = None
                return None
        else:
            print(f"ThermalMonitor: No known temperature attribute found on device '{self._phy_device.name}'. Checked: {temp_attr_names}")
            print("Available attributes:")
            try:
                for attr in self._phy_device.attrs:
                    print(f"  - {attr}")
            except Exception as e:
                print(f"  Error listing attributes: {e}")
            self.current_temperature_c = None
            return None

    def monitor_temperature_loop(self, callback=None, duration_sec=None, stop_event=None):
        """
        Continuously monitors temperature and calls a callback function.
        
        Args:
            callback (function): A function to call with the temperature value. 
                                 It will receive one argument: temperature in Celsius (or None).
            duration_sec (int, optional): How long to monitor for. Runs indefinitely if None.
            stop_event (threading.Event, optional): An event to signal stopping the loop.
        """
        print(f"Starting temperature monitoring loop. Interval: {self.update_interval}s")
        start_time = time.time()
        while True:
            if stop_event and stop_event.is_set():
                print("Temperature monitoring loop: Stop event received.")
                break
            if duration_sec and (time.time() - start_time > duration_sec):
                print("Temperature monitoring loop: Duration reached.")
                break

            temp_c = self.get_temperature()
            if temp_c is not None:
                print(f"ThermalMonitor: Current SDR Temperature: {temp_c:.2f}°C")
            else:
                print("ThermalMonitor: Failed to read SDR temperature.")
            
            if callback:
                try:
                    callback(temp_c)
                except Exception as e:
                    print(f"ThermalMonitor: Error in temperature callback: {e}")
            
            time.sleep(self.update_interval)
        print("Temperature monitoring loop finished.")

# Example Usage (for testing, comment out when integrating):
if __name__ == '__main__':
    # This example assumes sdr_controller.py is in the same directory or accessible
    # and an SDR is connected.
    # from sdr_controller import SDRController # If in the same dir for standalone test

    # Mock SDRController and its sdr.phy for standalone testing if no real SDR
    class MockPhyDevice:
        def __init__(self):
            self.name = "mock_ad9361-phy"
            self.temp0_input = "25000" # Simulate 25.0 C
            self.attrs = ["temp0_input", "some_other_attr"]

    class MockSDRDevice:
        def __init__(self):
            self.uri = "ip:mock.local"
            self.phy = MockPhyDevice() # AD9361 phy device
            self.ctx = None # Not deeply mocked for this example
            print("MockSDRDevice initialized.")

    class MockSDRController:
        def __init__(self, sdr_ip="ip:mock.local"):
            self.sdr_ip = sdr_ip
            self.sdr = MockSDRDevice()
            print(f"MockSDRController initialized with {sdr_ip}")

    print("--- Testing ThermalMonitor with MockSDRController ---")
    mock_sdr_ctrl = MockSDRController()
    thermal_mon = ThermalMonitor(sdr_controller=mock_sdr_ctrl, update_interval_sec=2)

    def my_temp_callback(temperature):
        if temperature is not None:
            print(f"CALLBACK: SDR Temperature is {temperature:.2f}°C")
            if temperature > 60:
                print("CALLBACK: WARNING! SDR Temperature is high!")
        else:
            print("CALLBACK: SDR Temperature could not be read.")

    # Test single read
    temp = thermal_mon.get_temperature()
    if temp is not None:
        print(f"Single read: SDR Temperature: {temp:.2f}°C")
    else:
        print("Single read: Failed to get SDR temperature.")

    # Test monitoring loop for a short duration
    print("\nStarting temperature monitoring loop for 6 seconds...")
    try:
        thermal_mon.monitor_temperature_loop(callback=my_temp_callback, duration_sec=6)
    except KeyboardInterrupt:
        print("Monitoring interrupted by user.")
    print("--- Test complete ---")

