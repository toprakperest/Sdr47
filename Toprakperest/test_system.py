#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Underground Detection System Test Module

Version: 2.1
Author: AI Assistant
Last Updated: 2023-11-15
"""

import os
import sys
import time
import json
import argparse
import logging
import threading
import random
import numpy as np
import matplotlib.pyplot as plt
import zmq
from datetime import datetime
import csv
import subprocess
from scipy import signal
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import psutil
import signal as sys_signal
from queue import Queue
from json.decoder import JSONDecodeError
from typing import Dict, List, Optional, Tuple

# Configure module path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Constants
MAX_THREADS = 5
SOCKET_TIMEOUT_MS = 1000
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 3
DEFAULT_SAMPLE_RATE = 20e6  # 20 MHz
DEFAULT_CENTER_FREQ = 1500e6  # 1.5 GHz
DETECTION_CONFIDENCE_THRESHOLD = 0.5
POSITION_TOLERANCE = 0.5  # meters

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            "test_system.log",
            maxBytes=MAX_LOG_SIZE,
            backupCount=LOG_BACKUP_COUNT
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UG_TEST_SYSTEM")


class TestSystem:
    """Enhanced underground detection test system with thread safety and improved reliability."""

    def __init__(self, args: argparse.Namespace):
        """Initialize the test system with configuration."""
        self.args = args
        self.mode = args.mode
        self.scenario = args.scenario
        self.data_file = os.path.abspath(args.data_file) if args.data_file else None
        self.duration = args.duration
        self.output_dir = os.path.abspath(args.output_dir)
        self.debug = args.debug

        # Thread safety
        self.lock = threading.Lock()
        self.threads = []
        self.thread_queues = {}

        # System state
        self.running = threading.Event()
        self.shutdown_flag = threading.Event()

        # ZMQ context
        self.context = None
        self.sockets = {}

        # Test results
        self.results = {
            "metadata": {
                "version": "2.1",
                "start_time": datetime.now().isoformat(),
                "mode": self.mode,
                "scenario": self.scenario,
                "config": vars(args)
            },
            "performance": {},
            "detections": [],
            "ground_truth": [],
            "metrics": {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "false_positive_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_detection_time_ms": 0.0
            }
        }

        # Setup signal handlers
        self._setup_signal_handlers()

        # Initialize directories
        self._init_directories()

        logger.info(f"Test system initialized in {self.mode} mode")

    def _setup_signal_handlers(self) -> None:
        """Setup system signal handlers for graceful shutdown."""

        def handler(signum, frame):
            logger.warning(f"Received signal {signum}, initiating shutdown...")
            self.stop()

        sys_signal.signal(sys_signal.SIGINT, handler)
        sys_signal.signal(sys_signal.SIGTERM, handler)

    def _init_directories(self) -> None:
        """Initialize required directories."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "raw_data"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        except OSError as e:
            logger.error(f"Directory creation failed: {str(e)}")
            raise

    def start(self) -> bool:
        """Start the test system."""
        try:
            # Initialize ZMQ
            self.context = zmq.Context()

            # Start mode-specific components
            if self.mode == "synthetic":
                success = self._start_synthetic_mode()
            elif self.mode == "replay":
                success = self._start_replay_mode()
            elif self.mode == "benchmark":
                success = self._start_benchmark_mode()
            else:
                logger.error(f"Invalid test mode: {self.mode}")
                success = False

            if success:
                self.running.set()
                logger.info("Test system started successfully")
                return True

            logger.error("Test system failed to start")
            return False

        except Exception as e:
            logger.critical(f"Startup failed: {str(e)}", exc_info=True)
            return False

    def stop(self) -> bool:
        """Stop the test system gracefully."""
        if not self.running.is_set():
            return True

        logger.info("Initiating shutdown sequence...")
        self.shutdown_flag.set()
        self.running.clear()

        # Stop all threads
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2.0)

        # Close sockets
        self._close_sockets()

        # Save final results
        self._analyze_results()
        self._save_results()

        logger.info("Test system shutdown complete")
        return True

    def _close_sockets(self) -> None:
        """Close all ZMQ sockets safely."""
        if not hasattr(self, 'sockets'):
            return

        # Close subscriber sockets first
        for name in list(self.sockets.keys()):
            if name.endswith('_sub'):
                try:
                    self.sockets[name].setsockopt(zmq.LINGER, 0)
                    self.sockets[name].close()
                    del self.sockets[name]
                except Exception as e:
                    logger.error(f"Error closing {name}: {str(e)}")

        # Then close publisher sockets
        for name in list(self.sockets.keys()):
            if name.endswith('_pub'):
                try:
                    self.sockets[name].setsockopt(zmq.LINGER, 0)
                    self.sockets[name].close()
                    del self.sockets[name]
                except Exception as e:
                    logger.error(f"Error closing {name}: {str(e)}")

        # Finally terminate context
        if self.context:
            self.context.term()
            self.context = None

    def _start_synthetic_mode(self) -> bool:
        """Start synthetic data generation mode."""
        try:
            # Setup ZMQ sockets
            self.sockets["sdr_pub"] = self.context.socket(zmq.PUB)
            self.sockets["sdr_pub"].bind("tcp://*:5555")
            self.sockets["sdr_pub"].setsockopt(zmq.SNDHWM, 100)

            self.sockets["ai_sub"] = self.context.socket(zmq.SUB)
            self.sockets["ai_sub"].connect("tcp://localhost:5559")
            self.sockets["ai_sub"].setsockopt(zmq.SUBSCRIBE, b"")
            self.sockets["ai_sub"].setsockopt(zmq.RCVTIMEO, SOCKET_TIMEOUT_MS)

            self.sockets["control_sub"] = self.context.socket(zmq.SUB)
            self.sockets["control_sub"].connect("tcp://localhost:5563")
            self.sockets["control_sub"].setsockopt(zmq.SUBSCRIBE, b"")
            self.sockets["control_sub"].setsockopt(zmq.RCVTIMEO, SOCKET_TIMEOUT_MS)

            # Generate ground truth
            if not self._generate_ground_truth():
                return False

            # Start worker threads
            self._start_thread(self._synthetic_data_worker, "DataGenerator")
            self._start_thread(self._result_collector_worker, "ResultCollector")

            return True

        except Exception as e:
            logger.error(f"Synthetic mode startup failed: {str(e)}", exc_info=True)
            return False

    def _start_replay_mode(self) -> bool:
        """Start replay mode with recorded data."""
        try:
            # Verify data file
            if not os.path.isfile(self.data_file):
                logger.error(f"Data file not found: {self.data_file}")
                return False

            # Setup ZMQ sockets
            self.sockets["sdr_pub"] = self.context.socket(zmq.PUB)
            self.sockets["sdr_pub"].bind("tcp://*:5555")
            self.sockets["sdr_pub"].setsockopt(zmq.SNDHWM, 100)

            self.sockets["ai_sub"] = self.context.socket(zmq.SUB)
            self.sockets["ai_sub"].connect("tcp://localhost:5559")
            self.sockets["ai_sub"].setsockopt(zmq.SUBSCRIBE, b"")
            self.sockets["ai_sub"].setsockopt(zmq.RCVTIMEO, SOCKET_TIMEOUT_MS)

            self.sockets["control_sub"] = self.context.socket(zmq.SUB)
            self.sockets["control_sub"].connect("tcp://localhost:5563")
            self.sockets["control_sub"].setsockopt(zmq.SUBSCRIBE, b"")
            self.sockets["control_sub"].setsockopt(zmq.RCVTIMEO, SOCKET_TIMEOUT_MS)

            # Load ground truth
            if not self._load_ground_truth():
                return False

            # Start worker threads
            self._start_thread(self._replay_data_worker, "DataReplay")
            self._start_thread(self._result_collector_worker, "ResultCollector")

            return True

        except Exception as e:
            logger.error(f"Replay mode startup failed: {str(e)}", exc_info=True)
            return False

    def _start_benchmark_mode(self) -> bool:
        """Start performance benchmarking mode."""
        try:
            # Start main system process
            self.main_process = subprocess.Popen(
                ["python", "main.py", "--mode=test", "--test-type=synthetic", "--optimize"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Setup control socket
            self.sockets["control_sub"] = self.context.socket(zmq.SUB)
            self.sockets["control_sub"].connect("tcp://localhost:5563")
            self.sockets["control_sub"].setsockopt(zmq.SUBSCRIBE, b"")
            self.sockets["control_sub"].setsockopt(zmq.RCVTIMEO, SOCKET_TIMEOUT_MS)

            # Start worker threads
            self._start_thread(self._performance_monitor_worker, "PerfMonitor")
            self._start_thread(self._result_collector_worker, "ResultCollector")

            # Start shutdown timer if duration is set
            if self.duration > 0:
                self._start_thread(
                    self._shutdown_timer_worker,
                    "ShutdownTimer",
                    kwargs={'duration': self.duration}
                )

            return True

        except Exception as e:
            logger.error(f"Benchmark mode startup failed: {str(e)}", exc_info=True)
            return False

    def _start_thread(self, target: callable, name: str, kwargs: dict = None) -> None:
        """Start a managed thread."""
        if kwargs is None:
            kwargs = {}

        # Create a communication queue for the thread
        queue = Queue()
        self.thread_queues[name] = queue

        # Start the thread
        thread = threading.Thread(
            target=target,
            name=name,
            kwargs=kwargs,
            daemon=True
        )
        thread.start()
        self.threads.append(thread)

    def _generate_ground_truth(self) -> bool:
        """Generate synthetic ground truth data based on scenario."""
        scenarios = {
            "metal_only": self._generate_metal_objects,
            "void_only": self._generate_void_objects,
            "mineral_only": self._generate_mineral_objects,
            "mixed": self._generate_mixed_objects,
            "noisy": self._generate_noisy_objects
        }

        if self.scenario not in scenarios:
            logger.error(f"Invalid scenario: {self.scenario}")
            return False

        try:
            # Generate objects
            self.results["ground_truth"] = scenarios[self.scenario]()

            # Save ground truth
            truth_file = os.path.join(self.output_dir, "ground_truth.json")
            with open(truth_file, 'w') as f:
                json.dump(self.results["ground_truth"], f, indent=2)

            logger.info(f"Generated {len(self.results['ground_truth'])} ground truth objects")
            return True

        except Exception as e:
            logger.error(f"Ground truth generation failed: {str(e)}", exc_info=True)
            return False

    def _generate_metal_objects(self) -> List[Dict]:
        """Generate metal objects for ground truth."""
        num_objects = random.randint(5, 15)
        objects = []

        for i in range(num_objects):
            obj = {
                "id": f"metal_{i:03d}",
                "type": "metal",
                "subtype": random.choice(["iron", "copper", "aluminum", "gold", "silver"]),
                "position": {
                    "x": round(random.uniform(0, 10), 2),
                    "y": round(random.uniform(0, 10), 2),
                    "z": round(random.uniform(-3, -0.1), 2)  # Include surface objects
                },
                "size": round(random.uniform(0.05, 0.8), 2),
                "reflection_strength": round(random.uniform(0.6, 1.0), 2),
                "conductivity": round(random.uniform(1e6, 6e7), 2)  # S/m
            }
            objects.append(obj)

        return objects

    def _generate_void_objects(self) -> List[Dict]:
        """Generate void objects for ground truth."""
        num_objects = random.randint(3, 10)
        objects = []

        for i in range(num_objects):
            obj = {
                "id": f"void_{i:03d}",
                "type": "void",
                "subtype": random.choice(["air", "water", "gas"]),
                "position": {
                    "x": round(random.uniform(0, 10), 2),
                    "y": round(random.uniform(0, 10), 2),
                    "z": round(random.uniform(-4, -0.5), 2)
                },
                "size": round(random.uniform(0.2, 1.5), 2),
                "reflection_strength": round(random.uniform(0.3, 0.8), 2),
                "permittivity": round(random.uniform(1, 80), 2)  # Relative permittivity
            }
            objects.append(obj)

        return objects

    def _generate_mineral_objects(self) -> List[Dict]:
        """Generate mineral objects for ground truth."""
        num_objects = random.randint(4, 12)
        objects = []

        for i in range(num_objects):
            obj = {
                "id": f"mineral_{i:03d}",
                "type": "mineral",
                "subtype": random.choice(["quartz", "granite", "limestone", "marble", "coal"]),
                "position": {
                    "x": round(random.uniform(0, 10), 2),
                    "y": round(random.uniform(0, 10), 2),
                    "z": round(random.uniform(-5, -1.0), 2)
                },
                "size": round(random.uniform(0.3, 2.0), 2),
                "reflection_strength": round(random.uniform(0.2, 0.7), 2),
                "density": round(random.uniform(1000, 3000), 2)  # kg/m³
            }
            objects.append(obj)

        return objects

    def _generate_mixed_objects(self) -> List[Dict]:
        """Generate mixed objects for ground truth."""
        objects = []

        # Metals (15-25%)
        num_metals = random.randint(2, 5)
        objects.extend(self._generate_metal_objects()[:num_metals])

        # Voids (25-35%)
        num_voids = random.randint(3, 6)
        objects.extend(self._generate_void_objects()[:num_voids])

        # Minerals (40-60%)
        num_minerals = random.randint(4, 8)
        objects.extend(self._generate_mineral_objects()[:num_minerals])

        return objects

    def _generate_noisy_objects(self) -> List[Dict]:
        """Generate noisy environment with challenging objects."""
        objects = self._generate_mixed_objects()

        # Reduce reflection strengths
        for obj in objects:
            if obj["type"] == "metal":
                obj["reflection_strength"] = round(obj["reflection_strength"] * 0.7, 2)
            elif obj["type"] == "void":
                obj["reflection_strength"] = round(obj["reflection_strength"] * 0.6, 2)
            else:
                obj["reflection_strength"] = round(obj["reflection_strength"] * 0.5, 2)

        return objects

    def _load_ground_truth(self) -> bool:
        """Load ground truth from data file."""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)

            if isinstance(data, dict) and "ground_truth" in data:
                self.results["ground_truth"] = data["ground_truth"]
            elif isinstance(data, list):
                self.results["ground_truth"] = data
            else:
                logger.error("Invalid ground truth data format")
                return False

            logger.info(f"Loaded {len(self.results['ground_truth'])} ground truth objects")
            return True

        except JSONDecodeError as e:
            logger.error(f"Invalid JSON in data file: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Failed to load ground truth: {str(e)}")
            return False

    def _synthetic_data_worker(self) -> None:
        """Worker thread for synthetic data generation."""
        logger.info("Synthetic data generator started")

        try:
            sample_rate = DEFAULT_SAMPLE_RATE
            center_freq = DEFAULT_CENTER_FREQ
            samples_per_chunk = 1024

            # Ground properties
            ground_properties = {
                "conductivity": round(random.uniform(0.001, 0.1), 4),
                "permittivity": round(random.uniform(2, 30), 2),
                "roughness": round(random.uniform(0.1, 1.0), 2)
            }

            while self.running.is_set() and not self.shutdown_flag.is_set():
                try:
                    # Time base
                    t = np.arange(samples_per_chunk) / sample_rate

                    # Carrier wave
                    carrier = np.sin(2 * np.pi * center_freq * t)

                    # Noise (adaptive based on ground roughness)
                    noise_level = 0.05 + (ground_properties["roughness"] * 0.1)
                    noise = np.random.normal(0, noise_level, samples_per_chunk)

                    # Reflections
                    reflections = np.zeros(samples_per_chunk)

                    for obj in self.results["ground_truth"]:
                        # Calculate delay based on depth and ground properties
                        distance = abs(obj["position"]["z"])
                        wave_velocity = 3e8 / np.sqrt(ground_properties["permittivity"])
                        delay = (2 * distance) / wave_velocity
                        delay_samples = int(delay * sample_rate)

                        if 0 <= delay_samples < samples_per_chunk:
                            # Reflection amplitude calculation
                            amplitude = (
                                    obj["reflection_strength"] *
                                    (1 / (1 + distance ** 2)) *  # Distance attenuation
                                    (1 - ground_properties["roughness"])  # Surface roughness
                            )

                            # Add reflection pulse (with some width)
                            pulse_width = max(1, int(sample_rate * obj["size"] / wave_velocity))
                            start = max(0, delay_samples - pulse_width // 2)
                            end = min(samples_per_chunk, delay_samples + pulse_width // 2)

                            import numpy as np
                            import time
                            from scipy import signal

                            # Örnek parametreler – Gerçek senaryoya göre değiştirilmeli
                            sample_rate = 1_000_000  # 1 MHz
                            center_freq = 2.4e9  # 2.4 GHz
                            amplitude = 1.0
                            pulse_width = 100
                            start = 100
                            end = 200
                            delay_samples = 150
                            num_samples = 1024
                            carrier_freq = 1e6  # 1 MHz
                            ground_properties = {}
                            start_time = time.time()

                            # Reflections, carrier ve noise sinyalleri tanımlanıyor
                            reflections = np.zeros(num_samples)
                            pulse = amplitude * np.exp(
                                -0.5 * ((np.arange(start, end) - delay_samples) ** 2 / (pulse_width / 4) ** 2)
                            )
                            reflections[start:end] += pulse

                            # Yere ait filtre uygulanıyor
                            fir_taps = signal.firwin(101, cutoff=0.3, window="hamming")
                            reflections = signal.lfilter(fir_taps, 1.0, reflections)

                            # Taşıyıcı sinyal ve gürültü
                            carrier = np.exp(1j * 2 * np.pi * carrier_freq * np.arange(num_samples) / sample_rate)
                            noise = np.random.normal(0, 0.1, num_samples) + 1j * np.random.normal(0, 0.1, num_samples)

                            # I/Q verisi oluşturuluyor
                            iq_data = carrier * (1 + reflections) + noise

                            # Simülasyon amaçlı örnek sınıf (gerçek kodda sınıf içindeyse "self" zaten tanımlı olur)
                            class DummyContext:
                                def __init__(self):
                                    self.scenario = "default"
                                    self.results = {"ground_truth": [1, 2, 3]}  # Örnek nesne sayısı
                                    from types import SimpleNamespace
                                    self.sockets = {
                                        "sdr_pub": SimpleNamespace(send_json=lambda x: print("Data sent:", x))}

                            context = DummyContext()

                            # Veri paketi oluşturuluyor
                            data_packet = {
                                "timestamp": time.time(),
                                "sample_rate": sample_rate,
                                "center_freq": center_freq,
                                "samples": iq_data.tolist(),
                                "ground_properties": ground_properties,
                                "metadata": {
                                    "scenario": context.scenario,
                                    "objects_count": len(context.results["ground_truth"])
                                }
                            }

                            # Veri gönderiliyor
                            context.sockets["sdr_pub"].send_json(data_packet)

                            # Uyarlanabilir bekleme
                            time.sleep(max(0.005, 0.01 - (time.time() - start_time)))


                except zmq.ZMQError as e:
                    logger.error(f"ZMQ error in data worker: {str(e)}")
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error in data generation: {str(e)}")
                    time.sleep(1)

        except Exception as e:
            logger.critical(f"Data worker failed: {str(e)}", exc_info=True)
        finally:
            logger.info("Synthetic data generator stopped")

    def _replay_data_worker(self) -> None:
        """Worker thread for replaying recorded data."""
        logger.info(f"Data replay worker started for {self.data_file}")

        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)

            if "frames" not in data:
                logger.error("Invalid data file format: missing 'frames' key")
                return

            start_time = time.time()
            first_timestamp = data["frames"][0]["timestamp"]
            frame_count = len(data["frames"])
            current_frame = 0

            while (self.running.is_set() and
                   not self.shutdown_flag.is_set() and
                   current_frame < frame_count):

                frame = data["frames"][current_frame]

                # Time synchronization
                elapsed = time.time() - start_time
                frame_elapsed = frame["timestamp"] - first_timestamp

                if elapsed < frame_elapsed:
                    time.sleep(frame_elapsed - elapsed)

                # Send frame
                try:
                    self.sockets["sdr_pub"].send_json(frame)
                    current_frame += 1

                    # Progress reporting
                    if current_frame % 100 == 0:
                        progress = (current_frame / frame_count) * 100
                        logger.info(f"Replay progress: {progress:.1f}%")

                except zmq.ZMQError as e:
                    logger.error(f"Frame send error: {str(e)}")
                    time.sleep(1)

            logger.info("Data replay completed")

        except Exception as e:
            logger.critical(f"Replay worker failed: {str(e)}", exc_info=True)
        finally:
            logger.info("Data replay worker stopped")
            self.running.clear()

    def _result_collector_worker(self) -> None:
        """Worker thread for collecting and processing results."""
        logger.info("Result collector started")

        try:
            while self.running.is_set() and not self.shutdown_flag.is_set():
                # Process AI results
                try:
                    ai_msg = self.sockets["ai_sub"].recv_json(flags=zmq.NOBLOCK)
                    self._process_ai_message(ai_msg)
                except zmq.Again:
                    pass
                except zmq.ZMQError as e:
                    logger.error(f"AI result ZMQ error: {str(e)}")
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"AI result processing error: {str(e)}")

                # Process control messages
                try:
                    control_msg = self.sockets["control_sub"].recv_json(flags=zmq.NOBLOCK)
                    self._process_control_message(control_msg)
                except zmq.Again:
                    pass
                except zmq.ZMQError as e:
                    logger.error(f"Control message ZMQ error: {str(e)}")
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Control message processing error: {str(e)}")

                # Adaptive sleep
                time.sleep(0.01)

        except Exception as e:
            logger.critical(f"Result collector failed: {str(e)}", exc_info=True)
        finally:
            logger.info("Result collector stopped")

    def _performance_monitor_worker(self) -> None:
        """Worker thread for performance monitoring."""
        logger.info("Performance monitor started")

        try:
            cpu_readings = []
            memory_readings = []
            last_report = time.time()

            while self.running.is_set() and not self.shutdown_flag.is_set():
                try:
                    # Capture metrics
                    cpu_readings.append(psutil.cpu_percent(interval=1))
                    memory_readings.append(psutil.virtual_memory().percent)

                    # Periodic reporting
                    if time.time() - last_report > 5:
                        with self.lock:
                            self.results["performance"].update({
                                "cpu_avg": round(np.mean(cpu_readings), 1),
                                "cpu_max": round(np.max(cpu_readings), 1),
                                "memory_avg": round(np.mean(memory_readings), 1),
                                "memory_max": round(np.max(memory_readings), 1),
                                "timestamp": time.time()
                            })

                        # Reset readings
                        cpu_readings = []
                        memory_readings = []
                        last_report = time.time()

                    time.sleep(1)

                except Exception as e:
                    logger.error(f"Performance monitoring error: {str(e)}")
                    time.sleep(5)

        except Exception as e:
            logger.critical(f"Performance monitor failed: {str(e)}", exc_info=True)
        finally:
            logger.info("Performance monitor stopped")

    def _shutdown_timer_worker(self, duration: int) -> None:
        """Worker thread for timed shutdown."""
        logger.info(f"Shutdown timer started for {duration} seconds")

        try:
            time.sleep(duration)
            logger.info("Shutdown timer expired, initiating shutdown")
            self.stop()
        except Exception as e:
            logger.error(f"Shutdown timer error: {str(e)}")
        finally:
            logger.info("Shutdown timer stopped")

    def _process_ai_message(self, message: Dict) -> None:
        """Process a message from the AI subsystem."""
        try:
            if "detections" not in message:
                return

            timestamp = message.get("timestamp", time.time())
            processing_time = message.get("processing_time_ms", 0)

            with self.lock:
                for detection in message["detections"]:
                    # Basic validation
                    if not all(k in detection for k in ["type", "confidence", "position"]):
                        logger.warning(f"Incomplete detection: {detection}")
                        continue

                    # Confidence threshold
                    if detection["confidence"] < DETECTION_CONFIDENCE_THRESHOLD:
                        continue

                    # Add to results
                    self.results["detections"].append({
                        "timestamp": timestamp,
                        "type": detection["type"],
                        "subtype": detection.get("subtype", ""),
                        "confidence": round(float(detection["confidence"]), 4),
                        "position": {
                            "x": round(float(detection["position"]["x"]), 3),
                            "y": round(float(detection["position"]["y"]), 3),
                            "z": round(float(detection["position"]["z"]), 3)
                        },
                        "size": round(float(detection.get("size", 0)), 3),
                        "processing_time_ms": processing_time
                    })

                # Record processing time if available
                if processing_time > 0:
                    if "detection_times" not in self.results["metrics"]:
                        self.results["metrics"]["detection_times"] = []
                    self.results["metrics"]["detection_times"].append(processing_time)

        except Exception as e:
            logger.error(f"AI message processing failed: {str(e)}")

    def _process_control_message(self, message: Dict) -> None:
        """Process a message from the control subsystem."""
        try:
            if "event" not in message:
                return

            event_type = message["event"]

            with self.lock:
                if event_type == "system_status":
                    # Update performance metrics
                    if "performance" in message:
                        self.results["performance"].update(message["performance"])

                elif event_type == "error":
                    # Log system errors
                    if "errors" not in self.results:
                        self.results["errors"] = []
                    self.results["errors"].append(message)
                    logger.error(f"System error: {message.get('message', 'Unknown')}")

                elif event_type == "warning":
                    # Log system warnings
                    if "warnings" not in self.results:
                        self.results["warnings"] = []
                    self.results["warnings"].append(message)
                    logger.warning(f"System warning: {message.get('message', 'Unknown')}")

        except Exception as e:
            logger.error(f"Control message processing failed: {str(e)}")

    def _analyze_results(self) -> None:
        """Analyze and calculate test metrics."""
        try:
            if not self.results["ground_truth"] or not self.results["detections"]:
                logger.warning("Insufficient data for analysis")
                return

            logger.info("Starting results analysis...")

            # Prepare ground truth data
            truth_positions = np.array([
                [obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]]
                for obj in self.results["ground_truth"]
            ])
            truth_types = [obj["type"] for obj in self.results["ground_truth"]]
            truth_ids = [obj["id"] for obj in self.results["ground_truth"]]

            # Prepare detection data
            det_positions = np.array([
                [det["position"]["x"], det["position"]["y"], det["position"]["z"]]
                for det in self.results["detections"]
            ])
            det_types = [det["type"] for det in self.results["detections"]]
            det_confidences = [det["confidence"] for det in self.results["detections"]]
            det_times = [
                det["processing_time_ms"]
                for det in self.results["detections"]
                if "processing_time_ms" in det
            ]

            # Initialize matching
            matched_truth = set()
            matched_detections = set()
            true_positives = 0
            detection_distances = []

            # Match detections to ground truth
            for i, det_pos in enumerate(det_positions):
                # Find nearest ground truth object of same type
                same_type_mask = np.array([t == det_types[i] for t in truth_types])
                if not np.any(same_type_mask):
                    continue  # No matching type in ground truth

                # Calculate distances to same-type objects
                distances = np.linalg.norm(
                    truth_positions[same_type_mask] - det_pos,
                    axis=1
                )
                min_dist_idx = np.argmin(distances)
                min_distance = distances[min_dist_idx]

                # Get original index
                original_indices = np.where(same_type_mask)[0]
                truth_idx = original_indices[min_dist_idx]

                # Check if match meets criteria
                if (min_distance <= POSITION_TOLERANCE and
                        truth_idx not in matched_truth):
                    true_positives += 1
                    matched_truth.add(truth_idx)
                    matched_detections.add(i)
                    detection_distances.append(min_distance)

            # Calculate metrics
            false_positives = len(self.results["detections"]) - len(matched_detections)
            false_negatives = len(self.results["ground_truth"]) - len(matched_truth)

            precision = true_positives / (true_positives + false_positives) if (
                                                                                           true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (
                                                                                        true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            avg_confidence = np.mean(det_confidences) if det_confidences else 0
            avg_distance = np.mean(detection_distances) if detection_distances else 0
            avg_detection_time = np.mean(det_times) if det_times else 0

            # Update results
            with self.lock:
                self.results["metrics"].update({
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1_score": round(f1_score, 4),
                    "false_positive_rate": round(false_positives / len(det_positions), 4) if det_positions else 0,
                    "avg_confidence": round(avg_confidence, 4),
                    "avg_position_error_m": round(avg_distance, 4),
                    "avg_detection_time_ms": round(avg_detection_time, 2),
                    "matched_objects": len(matched_truth),
                    "total_objects": len(self.results["ground_truth"]),
                    "total_detections": len(self.results["detections"])
                })

                # Add confusion matrix
                self._create_confusion_matrix(truth_types, det_types)

            logger.info("Results analysis completed")

        except Exception as e:
            logger.error(f"Results analysis failed: {str(e)}", exc_info=True)

    def _create_confusion_matrix(self, truth_types: List[str], pred_types: List[str]) -> None:
        """Create and save a confusion matrix."""
        try:
            # Get all unique classes
            classes = sorted(set(truth_types + pred_types))

            # Calculate confusion matrix
            cm = confusion_matrix(truth_types, pred_types, labels=classes)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
            plt.colorbar(im, ax=ax)

            # Set labels
            ax.set_xticks(np.arange(len(classes)))
            ax.set_yticks(np.arange(len(classes)))
            ax.set_xticklabels(classes, rotation=45)
            ax.set_yticklabels(classes)
            ax.set_xlabel('Predicted Type')
            ax.set_ylabel('Actual Type')
            ax.set_title('Normalized Confusion Matrix')

            # Add text annotations
            thresh = cm_normalized.max() / 2.
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, f"{cm_normalized[i, j]:.2f}\n({cm[i, j]})",
                            ha="center", va="center",
                            color="white" if cm_normalized[i, j] > thresh else "black")

            plt.tight_layout()

            # Save to file
            cm_path = os.path.join(self.output_dir, "plots", "confusion_matrix.png")
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Add to results
            with self.lock:
                self.results["metrics"]["confusion_matrix"] = cm_path
                self.results["metrics"]["confusion_matrix_data"] = {
                    "matrix": cm.tolist(),
                    "normalized": cm_normalized.tolist(),
                    "classes": classes
                }

        except Exception as e:
            logger.error(f"Confusion matrix creation failed: {str(e)}")

    def _save_results(self) -> None:
        """Save test results to output files."""
        try:
            # Finalize results
            self.results["metadata"]["end_time"] = datetime.now().isoformat()
            self.results["metadata"]["duration_sec"] = round(
                (datetime.fromisoformat(self.results["metadata"]["end_time"]) -
                 datetime.fromisoformat(self.results["metadata"]["start_time"])).total_seconds(),
                2
            )

            # Save JSON results
            json_path = os.path.join(self.output_dir, "test_results.json")
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)

            # Save CSV summary
            csv_path = os.path.join(self.output_dir, "test_summary.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # Write metadata
                writer.writerow(["Category", "Key", "Value"])
                for key, value in self.results["metadata"].items():
                    writer.writerow(["Metadata", key, str(value)])

                # Write metrics
                for key, value in self.results["metrics"].items():
                    if isinstance(value, (int, float, str)):
                        writer.writerow(["Metrics", key, str(value)])

                # Write performance
                if "performance" in self.results:
                    for key, value in self.results["performance"].items():
                        if isinstance(value, (int, float, str)):
                            writer.writerow(["Performance", key, str(value)])

            logger.info(f"Results saved to {json_path} and {csv_path}")

        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")


def main():
    """Main entry point for the test system."""
    parser = argparse.ArgumentParser(
        description="Underground Detection System Test Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["synthetic", "replay", "benchmark"],
        default="synthetic",
        help="Test operation mode"
    )

    # Scenario selection
    parser.add_argument(
        "--scenario",
        choices=["metal_only", "void_only", "mineral_only", "mixed", "noisy"],
        default="mixed",
        help="Test scenario for synthetic mode"
    )

    # Data file
    parser.add_argument(
        "--data-file",
        type=str,
        default="test_data.json",
        help="Path to recorded data file for replay mode"
    )

    # Duration
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Test duration in seconds for benchmark mode"
    )

    # Output directory
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_results",
        help="Directory to save test results"
    )

    # Debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Configure debug logging
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Create and run test system
    tester = TestSystem(args)

    try:
        if tester.start():
            # Main loop
            while tester.running.is_set():
                time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
    finally:
        tester.stop()

    logger.info("Test system exited")


if __name__ == "__main__":
    main()