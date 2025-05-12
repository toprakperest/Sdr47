# Ground Penetrating Radar (GPR) System

## 1. Overview

This document provides a comprehensive guide to the Ground Penetrating Radar (GPR) system software. The system is designed for real-time data acquisition from Software Defined Radio (SDR) hardware (specifically tested with Analog Devices PlutoSDR), signal processing, and subsurface visualization. It features a PyQt5-based graphical user interface (GUI) for controlling the SDR, managing scan parameters, performing calibrations, and viewing processed GPR data.

The software is structured into several key modules:

*   **Hardware (`ground_radar/hardware/`)**: Manages communication and control of the SDR (e.g., PlutoSDR) and antenna configurations. Includes `sdr_controller.py` for direct SDR interaction and `antenna_manager.py` for abstracting antenna setups.
*   **Algorithms (`ground_radar/algorithms/`)**: Contains modules for signal processing (`signal_processing.py`), AI-based data analysis (`ai_models.py`), and hybrid analysis techniques (`hybrid_analysis.py`).
*   **Configuration (`ground_radar/config/`)**: Handles loading of system settings (`settings.yaml`) and frequency profiles (`freq_profiles.json`) via `config_loader.py`.
*   **Data (`ground_radar/data/`)**: Manages data logging (`logger.py`), calibration procedures and data (`calibration.py`), and potentially geological databases (`geology_db.py`).
*   **UI (`ground_radar/ui/`)**: Implements the user interface, including the main application window (`main_window.py`) and data visualization components (`visualization.py`).
*   **Main (`ground_radar/main.py`)**: The main entry point to launch the GPR application.

This revised version of the software has been professionally refactored to ensure robustness, direct real-hardware compatibility (eliminating simulation-only code), and improved maintainability.

## 2. System Requirements

### 2.1. Hardware

*   **Software Defined Radio (SDR)**: An Analog Devices PlutoSDR is recommended and has been the primary target for development and testing. Other SDRs supported by `pyadi-iio` might work with modifications to the `sdr_controller.py` and configuration settings.
*   **Antennas**: Appropriate GPR antennas (transmitter and receiver(s)) compatible with your SDR and target frequency range.
*   **Host Computer**: A computer capable of running Python and the required libraries, with USB connectivity for the SDR.

### 2.2. Software

*   **Operating System**: Linux (recommended, development was on Ubuntu 22.04), macOS, or Windows with appropriate SDR drivers.
*   **Python**: Python 3.8 or newer.
*   **Libraries**: The core dependencies are listed in `requirements.txt`. These include:
    *   `PyQt5`: For the graphical user interface.
    *   `numpy`: For numerical operations, especially array manipulations.
    *   `pyadi-iio`: Analog Devices library for interfacing with SDRs like PlutoSDR. This often requires `libiio` to be installed on the system.
    *   `PyYAML`: For loading YAML configuration files.
    *   (Optional, for AI features if fully implemented and models provided): `tensorflow` or `keras`, `scikit-learn`.

## 3. Installation and Setup

### 3.1. Prerequisites

1.  **Install `libiio`**: This library is crucial for `pyadi-iio` to function. Installation methods vary by OS:
    *   **Linux (Debian/Ubuntu)**: `sudo apt-get install libiio-utils libiio-dev`
    *   **Windows/macOS**: Refer to the Analog Devices `libiio` wiki for installers or build instructions.

2.  **Install Python**: If not already installed, download and install Python from [python.org](https://python.org).

### 3.2. Setting up the Project

1.  **Extract the Project**: Unzip the provided project archive (e.g., `ground_radar_project.zip`) to a desired location on your computer. This will create a directory, for example, `ground_radar_extracted/`.

2.  **Create a Virtual Environment (Recommended)**: Open a terminal or command prompt, navigate into the extracted project's root directory (e.g., `cd path/to/ground_radar_extracted/`), and create a virtual environment:
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment**:
    *   **Linux/macOS**:
        ```bash
        source venv/bin/activate
        ```
    *   **Windows**:
        ```bash
        venv\Scripts\activate
        ```

4.  **Install Dependencies**: With the virtual environment activated, install the required Python packages from the `requirements.txt` file located in the `ground_radar_extracted` directory:
    ```bash
    pip install -r requirements.txt
    ```
    If you intend to use AI features and have the models, you might need to install `tensorflow` or `keras` separately if they are not included in `requirements.txt` for size reasons.

### 3.3. SDR Hardware Setup

1.  **Connect the SDR**: Connect your PlutoSDR (or other compatible SDR) to your computer via USB.
2.  **Verify SDR Detection**: Ensure your operating system detects the SDR. For PlutoSDR, it typically appears as a USB Ethernet device. You might need to configure its network settings or ensure drivers are correctly installed as per Analog Devices documentation.
    *   The default IP address used by the software for PlutoSDR is `ip:192.168.2.1`. If your PlutoSDR uses a different IP (e.g., `usb:X.X.X` for direct USB context or a different static IP), you will need to update this in the `ground_radar/config/settings.yaml` file or ensure your PlutoSDR is accessible at the default IP.

## 4. Configuration

The primary configuration file for the system is `ground_radar/config/settings.yaml`. This file allows you to customize various parameters without modifying the source code.

### 4.1. `settings.yaml` Key Sections:

*   **`sdr_settings`**: Crucial for hardware interaction.
    *   `default_ip`: The IP address or URI for your SDR (e.g., `"ip:192.168.2.1"` for PlutoSDR over network, or `"usb:1.2.3"` for a specific USB context if supported by `pyadi-iio` and your setup).
    *   `default_sample_rate`: Default sampling rate in Samples per Second (Sps).
    *   `default_rx_buffer_size`: Number of samples per data capture.
    *   `default_center_freq`: Default center frequency in Hertz (e.g., `433000000` for 433 MHz).
    *   `default_rx_rf_bandwidth`: Default receiver RF bandwidth in Hertz.
    *   `default_gain_control_mode_chanX`: Gain mode for RX channel X (`"manual"`, `"slow_attack"`, `"fast_attack"`).
    *   `default_rx_hardware_gain_chanX`: Manual gain value in dB for RX channel X.
    *   `default_tx_hardware_gain_chanX`: TX attenuation value in dB for TX channel X (for PlutoSDR, this is attenuation, so 0 is max power, negative values increase attenuation).

*   **`scan_parameters`**: Settings related to the scanning process.
    *   `default_scan_mode`: Pre-defined scan mode (though the UI might allow dynamic selection).
    *   `default_depth_m`, `trace_interval_cm`, `scan_speed_cm_s`: Informational parameters for planning scans.

*   **`algorithm_settings`**: Parameters for signal processing algorithms (STFT, wavelet, etc.).

*   **`ai_model_paths`**: Paths to pre-trained AI models if used.

*   **`ui_settings`**: UI-specific configurations like update intervals.

*   **`logging`**: Configuration for system and data logging.

*   **`calibration`**: Settings for calibration procedures.

*   **`geology`**: Path to geology database files.

*   **`default_frequency_profile`**: Name of a default frequency profile to load from `freq_profiles.json`.

### 4.2. `freq_profiles.json`

Located in `ground_radar/config/`, this JSON file stores predefined sets of SDR parameters (frequency, bandwidth, gain, etc.) optimized for different scanning scenarios or targets. The UI may allow selecting these profiles.

## 5. Running the Application

1.  **Ensure Virtual Environment is Active**: If you closed your terminal, reopen it, navigate to the project directory (`ground_radar_extracted/`), and activate the virtual environment (`source venv/bin/activate` or `venv\Scripts\activate`).

2.  **Navigate to the `ground_radar` Subdirectory**: The main script is inside the `ground_radar` package directory.
    ```bash
    cd ground_radar
    ```

3.  **Run `main.py`**: Execute the main application script:
    ```bash
    python main.py
    ```
    Alternatively, from the `ground_radar_extracted` directory, you can run it as a module (if your Python path is set up correctly or if you've installed it as a package):
    ```bash
    python -m ground_radar.main
    ```

### 5.1. Using the GUI

The main window (`BD-GPR Arayüzü`) provides the following functionalities:

*   **SDR Bağlantı (SDR Connection)**:
    *   **Bağlan (Connect)**: Attempts to connect to the SDR using the IP address specified in `settings.yaml`. Upon successful connection, this button will be disabled, and its text will change to "Bağlı" (Connected). Scan controls will become active.
*   **Tarama Ayarları (Scan Settings)**:
    *   **Merkez Frekans (Center Frequency)**: Set the SDR's center frequency in MHz.
    *   **RX Kazancı (RX Gain)**: Set the SDR's receiver gain in dB (for manual mode).
    *   **Tarama Başlat (Start Scan)**: Initiates the GPR scanning process. Data will be acquired, processed, and visualized in real-time (or near real-time).
    *   **Tarama Durdur (Stop Scan)**: Stops the ongoing scan.
*   **Kalibrasyon (Calibration)**:
    *   **Zemin Kalibrasyonu (Ground Calibration)**: Performs a ground calibration routine (e.g., using RX2 antenna if configured for it).
*   **Analiz (Analysis)**:
    *   **Zemin Türü (Terrain Type)**: Allows selection of the ground type, which may influence processing algorithms or AI interpretation.
*   **Visualization Panel**: The right side of the GUI displays the processed GPR data, typically as a B-scan (radargram) and potentially identified subsurface layers or anomalies.
*   **Status Bar**: Displays messages about the system's current status, connection progress, scan updates, and errors.

### 5.2. Standalone Hardware Tests

The `sdr_controller.py` and `antenna_manager.py` files contain `if __name__ == '__main__':` blocks with test scripts. These can be run independently to test basic SDR functionality before launching the full GUI application. Ensure your SDR is connected and configured.

Navigate to the `ground_radar/hardware/` directory and run:

*   For SDR Controller tests:
    ```bash
    python sdr_controller.py
    ```
*   For Antenna Manager tests (this also initializes and uses `SDRController`):
    ```bash
    python antenna_manager.py
    ```
Review the console output for connection status, configuration details, and any error messages.

## 6. Development and Code Structure

(As described in Section 1. Overview)

### Key Refinements in this Version:

*   **`SDRController` (`hardware/sdr_controller.py`)**: Completely refactored for clarity, robustness, and direct hardware interaction. Redundant methods were removed, and a single, comprehensive `__init__` method is used. Configuration and control methods are more explicit and include better error handling and logging.
*   **`AntennaManager` (`hardware/antenna_manager.py`)**: Revised to work seamlessly with the new `SDRController`. It provides a clear abstraction for antenna configurations (e.g., main scan, calibration scan) and manages the activation/deactivation of SDR TX/RX channels based on antenna names.
*   **Configuration Loading**: `config_loader.py` remains robust for loading `settings.yaml` and `freq_profiles.json`.
*   **Real Hardware Focus**: All simulation-specific code paths have been removed or refactored to ensure the system is built for real hardware deployment.
*   **Error Handling**: Improved error handling and status messaging throughout the hardware interface and UI.

## 7. Troubleshooting

*   **SDR Connection Issues**:
    *   Verify the SDR is powered on and properly connected via USB.
    *   Check the IP address in `settings.yaml` matches your SDR's actual IP address. For PlutoSDR, ensure it's accessible on the network (e.g., `ping 192.168.2.1`).
    *   Ensure `libiio` and `pyadi-iio` are correctly installed. Missing `libiio` is a common cause of `pyadi-iio` failing to find or connect to the SDR.
    *   Check system logs or dmesg (Linux) for SDR detection issues.
*   **PyQt5 Issues**:
    *   Ensure PyQt5 is installed in your active Python environment.
    *   If you encounter display issues, ensure your graphics drivers are up to date.
*   **No Data Captured / Processing Fails**:
    *   Check SDR antenna connections.
    *   Verify SDR parameters (frequency, gain, bandwidth) are appropriate for your setup and target.
    *   Review console output and log files (`logs/system_activity.log`) for error messages from `SDRController` or processing algorithms.
*   **Slow Performance**:
    *   GPR data processing can be computationally intensive. Ensure your host computer meets reasonable performance requirements.
    *   Adjust `rx_buffer_size` or processing parameters if necessary.

## 8. Packaging for Distribution (Developer Note)

To package this application for distribution (e.g., using PyInstaller or Nuitka), you would typically:

1.  Ensure all dependencies are captured in `requirements.txt`.
2.  Use the chosen packaging tool to create an executable, making sure to include:
    *   All Python scripts.
    *   The `config` directory (with `settings.yaml`, `freq_profiles.json`).
    *   The `ui` directory and its contents if not fully compiled in.
    *   Any AI models (`models/` directory) if used.
    *   Any other necessary assets (icons, etc.).
3.  Test the packaged application thoroughly on a clean system.

This step is beyond the current scope but is a common next step for distributing Python applications.

---
This documentation provides a guide to the refactored GPR system. For further development or advanced troubleshooting, refer to the source code comments and individual module docstrings.
