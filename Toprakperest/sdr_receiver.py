#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced SDR Receiver for Underground Detection System

Version: 2.2
Author: AI Assistant
Last Updated: 2023-11-15

Features:
- Robust SDR data acquisition with SoapySDR
- Advanced DSP processing (bandpass/notch filtering, adaptive gain)
- Real-time data publishing via ZMQ
- Comprehensive error handling and recovery
- Thread-safe design with performance monitoring
- Built-in test mode with scenario simulation
"""

import os
import sys
import time
import argparse
import logging
import json
import numpy as np
import zmq
from threading import Thread, Event, Lock
from queue import Queue, Full
import scipy.signal as signal
from datetime import datetime
from collections import deque
import psutil

# Conditional imports for SDR libraries
try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32, SOAPY_SDR_CS16

    SOAPY_AVAILABLE = True
except ImportError:
    SOAPY_AVAILABLE = False
    print("SoapySDR library not found. Running in test mode.")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            "sdr_receiver.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=3
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SDR_RECEIVER")


class SDRReceiver:
    """High-performance SDR receiver with advanced signal processing."""

    def __init__(self, args):
        """
        Initialize SDR receiver with configuration parameters.

        Args:
            args: Command line arguments (see parse_args())
        """
        self.args = args
        self.running = Event()
        self.shutdown_event = Event()

        # Hardware configuration
        self.device_args = args.device_args
        self.freq = self._parse_frequency(args.freq)
        self.sample_rate = self._parse_frequency(args.rate)
        self.bandwidth = self._parse_frequency(args.bandwidth) if args.bandwidth else None
        self.gain = args.gain
        self.antenna = args.antenna
        self.channel = args.channel
        self.test_mode = args.test_mode or not SOAPY_AVAILABLE
        self.lna_enabled = args.lna_enabled
        self.lna_gain = args.lna_gain

        # DSP configuration
        self.bandpass_low = self._parse_frequency(args.bandpass_low)
        self.bandpass_high = self._parse_frequency(args.bandpass_high)
        self.notch_freq = self._parse_frequency(args.notch_freq)
        self.notch_width = self._parse_frequency(args.notch_width)

        # Data buffers and queues
        self.raw_queue = Queue(maxsize=100)
        self.processed_queue = Queue(maxsize=100)
        self.spectrum_buffer = deque(maxlen=10)  # Last 10 spectra

        # Thread synchronization
        self.lock = Lock()
        self.threads = {}

        # ZMQ configuration
        self.context = zmq.Context()
        self.data_socket = self.context.socket(zmq.PUB)
        self.data_socket.bind(f"tcp://*:{args.data_port}")

        # Control channel
        self.control_socket = self.context.socket(zmq.REP)
        self.control_socket.bind(f"tcp://*:{args.control_port}")
        self.control_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout

        # SDR device
        self.sdr = None
        self.rx_stream = None

        # Performance metrics
        self.metrics = {
            'start_time': time.time(),
            'samples_processed': 0,
            'dropped_packets': 0,
            'processing_time': 0,
            'snr_estimates': [],
            'current_gain': self.gain,
            'hardware': {}
        }

        # Initialize DSP filters
        self._init_filters()

        logger.info(f"SDR Receiver initialized (Test Mode: {self.test_mode})")

    def _parse_frequency(self, freq_str):
        """Parse frequency string with unit suffixes (e.g., 1.5G, 20M)."""
        units = {'k': 1e3, 'K': 1e3, 'M': 1e6, 'G': 1e9}
        if isinstance(freq_str, (int, float)):
            return float(freq_str)

        if freq_str[-1] in units:
            return float(freq_str[:-1]) * units[freq_str[-1]]
        return float(freq_str)

    def _init_filters(self):
        """Initialize DSP filters based on configuration."""
        # Bandpass filter
        if 0 < self.bandpass_low < self.bandpass_high < self.sample_rate / 2:
            nyq = self.sample_rate / 2
            low = self.bandpass_low / nyq
            high = self.bandpass_high / nyq
            self.bandpass_taps = signal.firwin(101, [low, high], pass_zero=False)
            logger.info(f"Bandpass filter: {self.bandpass_low / 1e6:.2f}-{self.bandpass_high / 1e6:.2f} MHz")
        else:
            self.bandpass_taps = None
            logger.warning("Invalid bandpass range, filter disabled")

        # Notch filter
        if 0 < self.notch_freq < self.sample_rate / 2 and self.notch_width > 0:
            nyq = self.sample_rate / 2
            freq_norm = self.notch_freq / nyq
            q_factor = self.notch_freq / self.notch_width
            self.notch_b, self.notch_a = signal.iirnotch(freq_norm, q_factor)
            logger.info(f"Notch filter: {self.notch_freq / 1e6:.2f} MHz, Q={q_factor:.1f}")
        else:
            self.notch_b = self.notch_a = None
            logger.warning("Invalid notch params, filter disabled")

    def start(self):
        """Start all receiver components."""
        if not self.test_mode:
            if not self._init_sdr():
                logger.error("SDR initialization failed")
                return False

        self.running.set()

        # Start worker threads
        self.threads = {
            'acquisition': Thread(target=self._acquisition_worker, name='Acquisition'),
            'processing': Thread(target=self._processing_worker, name='Processing'),
            'control': Thread(target=self._control_worker, name='Control'),
            'monitor': Thread(target=self._monitor_worker, name='Monitor')
        }

        for name, thread in self.threads.items():
            thread.daemon = True
            thread.start()
            logger.info(f"Started {name} thread")

        logger.info(f"SDR Receiver started (Freq: {self.freq / 1e6:.2f} MHz, Rate: {self.sample_rate / 1e6:.2f} MS/s)")
        return True

    def stop(self):
        """Gracefully shutdown all components."""
        if not self.running.is_set():
            return

        logger.info("Initiating shutdown...")
        self.running.clear()
        self.shutdown_event.set()

        # Stop threads
        for name, thread in self.threads.items():
            if thread.is_alive():
                thread.join(timeout=2.0)
                logger.info(f"Stopped {name} thread")

        # Close SDR resources
        if not self.test_mode and self.sdr:
            self._close_sdr()

        # Close ZMQ sockets
        self.data_socket.setsockopt(zmq.LINGER, 0)
        self.control_socket.setsockopt(zmq.LINGER, 0)
        self.data_socket.close()
        self.control_socket.close()
        self.context.term()

        # Save final metrics
        self._log_metrics()
        logger.info("SDR Receiver shutdown complete")

    def _init_sdr(self):
        """Initialize SDR hardware."""
        try:
            self.sdr = SoapySDR.Device(self.device_args)

            # Configure RX channel
            self.sdr.setSampleRate(SOAPY_SDR_RX, self.channel, self.sample_rate)
            self.sdr.setFrequency(SOAPY_SDR_RX, self.channel, self.freq)

            if self.bandwidth:
                self.sdr.setBandwidth(SOAPY_SDR_RX, self.channel, self.bandwidth)

            if self.antenna:
                self.sdr.setAntenna(SOAPY_SDR_RX, self.channel, self.antenna)

            # Configure gain
            self.sdr.setGain(SOAPY_SDR_RX, self.channel, self.gain)

            # Configure LNA if available
            if self.lna_enabled and "LNA" in self.sdr.listGains(SOAPY_SDR_RX, self.channel):
                self.sdr.setGain(SOAPY_SDR_RX, self.channel, "LNA", self.lna_gain)

            # Setup stream
            self.rx_stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [self.channel])

            # Get hardware info
            self.metrics['hardware'] = self.sdr.getHardwareInfo()
            logger.info(f"SDR initialized: {self.metrics['hardware']}")
            return True

        except Exception as e:
            logger.error(f"SDR init failed: {str(e)}")
            return False

    def _close_sdr(self):
        """Release SDR resources."""
        try:
            if self.rx_stream:
                self.sdr.deactivateStream(self.rx_stream)
                self.sdr.closeStream(self.rx_stream)
            logger.info("SDR resources released")
        except Exception as e:
            logger.error(f"SDR cleanup error: {str(e)}")

    def _acquisition_worker(self):
        """Thread for acquiring samples from SDR."""
        logger.info("Acquisition worker started")

        if self.test_mode:
            self._test_data_generator()
            return

        try:
            self.sdr.activateStream(self.rx_stream)
            buffer = np.zeros(1024, dtype=np.complex64)

            while self.running.is_set():
                try:
                    # Read from SDR
                    sr = self.sdr.readStream(self.rx_stream, [buffer], len(buffer), timeoutUs=100000)

                    if sr.ret > 0:  # Valid data
                        samples = buffer[:sr.ret].copy()

                        # Adaptive gain control
                        self._adaptive_gain_control(samples)

                        # Put in queue (non-blocking)
                        try:
                            self.raw_queue.put_nowait(samples)
                        except Full:
                            with self.lock:
                                self.metrics['dropped_packets'] += 1

                    elif sr.ret == SoapySDR.SOAPY_SDR_OVERFLOW:
                        logger.warning("SDR overflow detected")
                    elif sr.ret == SoapySDR.SOAPY_SDR_TIMEOUT:
                        continue
                    else:
                        logger.error(f"SDR read error: {SoapySDR.errToStr(sr.ret)}")
                        time.sleep(0.1)

                except Exception as e:
                    logger.error(f"Acquisition error: {str(e)}")
                    time.sleep(1)

        except Exception as e:
            logger.critical(f"Acquisition worker failed: {str(e)}")
            self.running.clear()
        finally:
            logger.info("Acquisition worker stopped")

    def _adaptive_gain_control(self, samples):
        """Adjust gain dynamically based on signal levels."""
        if not self.args.adaptive_gain:
            return

        rms = np.sqrt(np.mean(np.abs(samples) ** 2))

        with self.lock:
            if rms < 0.1:  # Low signal
                new_gain = min(self.gain + 3, 50)
            elif rms > 0.9:  # Overload
                new_gain = max(self.gain - 5, 0)
            else:
                return

            if new_gain != self.gain:
                self.gain = new_gain
                self.sdr.setGain(SOAPY_SDR_RX, self.channel, self.gain)
                self.metrics['current_gain'] = self.gain
                logger.debug(f"Gain adjusted to {self.gain} dB (RMS: {rms:.2f})")


def _test_data_generator(self):
    """Generate test data when in test mode."""
    logger.info("Test data generator started")

    # Test signal parameters
    t = np.arange(1024) / self.sample_rate
    targets = [
        {'type': 'metal', 'freq': 0.05, 'amp': 0.8, 'delay': 0.2},
        {'type': 'void', 'freq': 0.03, 'amp': 0.5, 'delay': 0.5},
        {'type': 'mineral', 'freq': 0.08, 'amp': 0.3, 'delay': 0.7}
    ]

    while self.running.is_set():
        try:
            # Base signal with noise
            noise_level = 0.1 + 0.05 * np.sin(time.time() / 10)  # Varying noise
            noise = noise_level * (np.random.randn(1024) + 1j * np.random.randn(1024))

            # Add targets
            signal_data = noise.copy()
            for target in targets:
                pulse = target['amp'] * np.exp(-((t - target['delay']) ** 2) / 0.01)
                signal_data += pulse * np.exp(1j * 2 * np.pi * target['freq'] * t)

            # Add ground reflection
            signal_data += 0.3 * np.exp(-t / 2) * np.sin(2 * np.pi * 0.01 * t)

            # Put in queue
            try:
                self.raw_queue.put_nowait(signal_data.astype(np.complex64))
                time.sleep(0.1)  # Simulate real-time
            except Full:
                with self.lock:
                    self.metrics['dropped_packets'] += 1

        except Exception as e:
            logger.error(f"Test data error: {str(e)}")
            time.sleep(1)

    logger.info("Test data generator stopped")


def _processing_worker(self):
    """Thread for processing acquired samples."""
    logger.info("Processing worker started")

    prev_samples = None  # For feed-through cancellation

    while self.running.is_set():
        try:
            # Get raw data
            if self.raw_queue.empty():
                time.sleep(0.01)
                continue

            start_time = time.time()
            data = self.raw_queue.get()

            # Process data
            processed = self._process_samples(data, prev_samples)
            prev_samples = data[-100:]  # Keep last samples for next iteration

            # Calculate spectrum
            spectrum = np.fft.fftshift(np.abs(np.fft.fft(processed)))
            self.spectrum_buffer.append(spectrum)

            # Estimate SNR
            snr = self._estimate_snr(processed)
            with self.lock:
                self.metrics['snr_estimates'].append(snr)
                if len(self.metrics['snr_estimates']) > 100:
                    self.metrics['snr_estimates'].pop(0)

            # Publish results
            self._publish_data(processed)

            # Update metrics
            with self.lock:
                self.metrics['samples_processed'] += len(data)
                self.metrics['processing_time'] += time.time() - start_time

        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            time.sleep(0.1)

    logger.info("Processing worker stopped")


def _process_samples(self, samples, prev_samples=None):
    """Apply all DSP processing to samples."""
    processed = samples.copy()

    # Bandpass filter
    if self.bandpass_taps is not None:
        processed = signal.lfilter(self.bandpass_taps, 1.0, processed)

    # Notch filter
    if self.notch_b is not None:
        processed = signal.lfilter(self.notch_b, self.notch_a, processed)

    # Feed-through cancellation
    if prev_samples is not None and len(prev_samples) > 0:
        min_len = min(len(processed), len(prev_samples))
        processed[:min_len] -= 0.3 * prev_samples[-min_len:]

    return processed


def _estimate_snr(self, samples):
    """Estimate SNR in dB."""
    power = np.mean(np.abs(samples) ** 2)
    noise_floor = np.percentile(np.abs(samples), 10) ** 2
    return 10 * np.log10(power / noise_floor) if noise_floor > 0 else 0


def _publish_data(self, data):
    """Publish processed data via ZMQ."""
    try:
        msg = {
            'timestamp': time.time(),
            'samples': data.tolist(),
            'sample_rate': self.sample_rate,
            'center_freq': self.freq,
            'snr': self.metrics['snr_estimates'][-1] if self.metrics['snr_estimates'] else 0
        }
        self.data_socket.send_json(msg, flags=zmq.NOBLOCK)
    except Exception as e:
        logger.error(f"Publish error: {str(e)}")


def _control_worker(self):
    """Thread for handling control commands."""
    logger.info("Control worker started")

    while self.running.is_set():
        try:
            # Check for control messages
            try:
                msg = self.control_socket.recv_json(flags=zmq.NOBLOCK)
                self._handle_control_message(msg)
            except zmq.Again:
                time.sleep(0.1)
                continue
            except Exception as e:
                logger.error(f"Control message error: {str(e)}")
                self.control_socket.send_json({'status': 'error', 'message': str(e)})
                continue

        except Exception as e:
            logger.error(f"Control worker error: {str(e)}")
            time.sleep(1)

    logger.info("Control worker stopped")


def _handle_control_message(self, msg):
    """Process incoming control messages."""
    cmd = msg.get('command', '').lower()

    try:
        if cmd == 'get_status':
            response = {
                'status': 'running' if self.running.is_set() else 'stopped',
                'metrics': self._get_metrics(),
                'hardware': self.metrics['hardware']
            }

        elif cmd == 'set_gain':
            new_gain = float(msg['value'])
            with self.lock:
                self.gain = max(0, min(new_gain, 50))
                if not self.test_mode:
                    self.sdr.setGain(SOAPY_SDR_RX, self.channel, self.gain)
                self.metrics['current_gain'] = self.gain
            response = {'status': 'ok', 'new_gain': self.gain}

        elif cmd == 'get_spectrum':
            response = {
                'status': 'ok',
                'spectrum': self.spectrum_buffer[-1].tolist() if self.spectrum_buffer else []
            }

        else:
            response = {'status': 'error', 'message': 'Unknown command'}

        self.control_socket.send_json(response)

    except Exception as e:
        logger.error(f"Command '{cmd}' failed: {str(e)}")
        self.control_socket.send_json({'status': 'error', 'message': str(e)})


def _monitor_worker(self):
    """Thread for monitoring system health."""
    logger.info("Monitor worker started")

    while self.running.is_set():
        try:
            # Log periodic status
            if int(time.time()) % 30 == 0:  # Every 30 seconds
                self._log_metrics()

            # Check queue health
            if self.raw_queue.qsize() > 90:
                logger.warning(f"Raw queue filling up: {self.raw_queue.qsize()}/100")

            time.sleep(1)

        except Exception as e:
            logger.error(f"Monitor error: {str(e)}")
            time.sleep(5)

    logger.info("Monitor worker stopped")


def _log_metrics(self):
    """Log current performance metrics."""
    with self.lock:
        metrics = self._get_metrics()
        logger.info(
            f"Metrics: Samples={metrics['samples_processed']}, "
            f"Dropped={metrics['dropped_packets']}, "
            f"SNR={metrics['snr_avg']:.1f}dB, "
            f"Gain={metrics['current_gain']}dB, "
            f"CPU={metrics['cpu_usage']}%"
        )


def _get_metrics(self):
    """Get current metrics snapshot."""
    with self.lock:
        metrics = self.metrics.copy()
        metrics.update({
            'snr_avg': np.mean(metrics['snr_estimates']) if metrics['snr_estimates'] else 0,
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'uptime': time.time() - metrics['start_time']
        })
        return metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Advanced SDR Receiver for Underground Detection")

    # SDR hardware parameters
    parser.add_argument("--device-args", default="", help="SDR device arguments")
    parser.add_argument("--freq", default="1.5G", help="Center frequency (e.g., 1.5G)")
    parser.add_argument("--rate", default="20M", help="Sample rate (e.g., 20M)")
    parser.add_argument("--bandwidth", default="", help="Frontend bandwidth")
    parser.add_argument("--gain", type=float, default=30, help="RX gain in dB")
    parser.add_argument("--antenna", default="", help="RX antenna port")
    parser.add_argument("--channel", type=int, default=0, help="RX channel index")

    # DSP parameters
    parser.add_argument("--bandpass-low", default="1M", help="Bandpass lower cutoff")
    parser.add_argument("--bandpass-high", default="3M", help="Bandpass upper cutoff")
    parser.add_argument("--notch-freq", default="0", help="Notch center frequency")
    parser.add_argument("--notch-width", default="100K", help="Notch bandwidth")

    # System parameters
    parser.add_argument("--data-port", type=int, default=5555, help="ZMQ data publish port")
    parser.add_argument("--control-port", type=int, default=5556, help="ZMQ control port")
    parser.add_argument("--test-mode", action="store_true", help="Enable test data generation")
    parser.add_argument("--lna-enabled", action="store_true", help="Enable LNA gain stage")
    parser.add_argument("--lna-gain", type=float, default=20, help="LNA gain in dB")
    parser.add_argument("--adaptive-gain", action="store_true", help="Enable automatic gain control")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    receiver = SDRReceiver(args)

    try:
        if receiver.start():
            # Main loop
            while receiver.running.is_set():
                time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()