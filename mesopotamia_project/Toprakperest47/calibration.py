import os
import sys
import time
import argparse
import numpy as np
import zmq
import json
import logging
import logging.handlers
import pickle
from datetime import datetime
from threading import Thread, Event, Lock
from queue import Queue, Empty, Full
import scipy.signal as signal
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Tuple, Union

# Yerel modüller
from config import (
    GroundType, COLOR_MAP_GROUND, PORT_SDR_DATA, PORT_SDR_CONTROL,
    PORT_CALIBRATION_RESULT, PORT_CALIBRATION_CONTROL,
    DEFAULT_SCAN_FREQ_START_MHZ, DEFAULT_SCAN_FREQ_END_MHZ,
    DEFAULT_CALIBRATION_DIR, DEFAULT_PERMITTIVITY, DEFAULT_SDR_SAMPLE_RATE
)

def setup_calibration_logger():
    """Kalibrasyon logger'ını kurar"""
    logger = logging.getLogger("CALIBRATION")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

calibration_logger = setup_calibration_logger()
FEATURE_VECTOR_SIZE = 10
MIN_SAMPLES_FOR_CALIBRATION = 5

class CalibrationModule:
    """Zemin kalibrasyonunu yönetir."""

    GROUND_TYPE_SIGNATURES = {
        GroundType.DRY_SOIL: np.array([0.8, 0.1, 0.5, 0.2, 0.7] + [0] * (FEATURE_VECTOR_SIZE - 5)),
        GroundType.WET_SOIL: np.array([0.6, 0.2, 1.5, 0.5, 0.3] + [0] * (FEATURE_VECTOR_SIZE - 5)),
        GroundType.LIMESTONE: np.array([0.7, 0.1, 0.8, 0.3, 0.8] + [0] * (FEATURE_VECTOR_SIZE - 5)),
        GroundType.IRON_OXIDE: np.array([0.5, 0.3, 1.2, 0.6, 0.2] + [0] * (FEATURE_VECTOR_SIZE - 5)),
        GroundType.ROCKY: np.array([0.75, 0.15, 0.7, 0.3, 0.75] + [0] * (FEATURE_VECTOR_SIZE - 5)),
        GroundType.UNKNOWN: np.array([0.5] * FEATURE_VECTOR_SIZE)
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.calibration_time_sec = float(config.get("calibration_time_sec", 60))
        self.calibration_sweeps = int(config.get("calibration_sweeps", 20))
        self.reference_data_path = config.get("reference_data_path",
                                            os.path.join(DEFAULT_CALIBRATION_DIR, "reference.pkl"))
        self.noise_antenna = config.get("noise_antenna", "Dipole_RX2")
        self.scan_antenna = config.get("scan_antenna", "UWB_RX1")
        self.min_freq_mhz = float(config.get("min_scan_freq_mhz", DEFAULT_SCAN_FREQ_START_MHZ))
        self.max_freq_mhz = float(config.get("max_scan_freq_mhz", DEFAULT_SCAN_FREQ_END_MHZ))
        self.noise_monitor_interval_sec = float(config.get("noise_monitor_interval_sec", 300))
        self.sample_rate = float(config.get("sample_rate", DEFAULT_SDR_SAMPLE_RATE))

        # Durum değişkenleri
        self.calibration_complete = False
        self.ground_type = GroundType.UNKNOWN
        self.estimated_permittivity = DEFAULT_PERMITTIVITY
        self.ground_params: Dict[str, Any] = {}
        self.calibration_progress = 0.0
        self.background_noise_db = -100.0
        self.interference_freqs: List[float] = []

        # Veri yapıları
        self.calibration_data_buffer = defaultdict(lambda: deque(maxlen=50))
        self.feature_vectors = defaultdict(list)
        self.reference_signatures = self._load_reference_signatures()
        self.scaler = StandardScaler()

        # ZMQ bağlantıları
        self.context = zmq.Context()
        self.sdr_data_socket = self.context.socket(zmq.SUB)
        self.sdr_data_socket.connect(f"tcp://localhost:{config.get('input_port', PORT_SDR_DATA)}")
        self.sdr_data_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sdr_data_socket.setsockopt(zmq.CONFLATE, 1)

        self.result_socket = self.context.socket(zmq.PUB)
        self.result_socket.bind(f"tcp://*:{config.get('result_port', PORT_CALIBRATION_RESULT)}")

        self.control_socket = self.context.socket(zmq.REP)
        self.control_socket.bind(f"tcp://*:{config.get('control_port', PORT_CALIBRATION_CONTROL)}")
        self.control_socket.setsockopt(zmq.RCVTIMEO, 1000)

        self.sdr_control_socket = self.context.socket(zmq.REQ)
        self.sdr_control_socket.connect(f"tcp://localhost:{PORT_SDR_CONTROL}")

        # Thread yönetimi
        self.running = Event()
        self.shutdown_event = Event()
        self.lock = Lock()
        self.data_queue = Queue(maxsize=500)
        self.threads: Dict[str, Optional[Thread]] = {}

        calibration_logger.info("Kalibrasyon modülü başlatıldı.")

    def _load_reference_signatures(self):
        """Referans imzalarını yükler"""
        if os.path.exists(self.reference_data_path):
            with open(self.reference_data_path, "rb") as f:
                return pickle.load(f)
        else:
            return self.GROUND_TYPE_SIGNATURES

    def _save_calibration_result(self, result: Dict[str, Any]):
        """Kalibrasyon sonuçlarını kaydeder"""
        with open(self.reference_data_path, "wb") as f:
            pickle.dump(result, f)

    def _send_sdr_command(self, command: str):
        """SDR cihazına komut gönderir"""
        self.sdr_control_socket.send_string(command)
        response = self.sdr_control_socket.recv_string()
        return response

    def start(self):
        """Kalibrasyon modülünü başlatır"""
        self.running.set()
        self.threads["receiver"] = Thread(target=self._data_receiver_worker, name="ReceiverThread", daemon=True)
        self.threads["processor"] = Thread(target=self._processing_worker, name="ProcessorThread", daemon=True)
        self.threads["control"] = Thread(target=self._control_worker, name="ControlThread", daemon=True)
        self.threads["noise_monitor"] = Thread(target=self._noise_monitoring_worker, name="NoiseMonitor",
                                               daemon=True)

        for t in self.threads.values():
            if t:
                t.start()

        return True

    def stop(self):
        """Kalibrasyon modülünü durdurur"""
        self.running.clear()
        self.shutdown_event.set()
        time.sleep(1)
        self.context.term()
        calibration_logger.info("Kalibrasyon modülü durduruldu.")

    def _data_receiver_worker(self):
        while self.running.is_set():
            try:
                raw = self.sdr_data_socket.recv(flags=zmq.NOBLOCK)
                if raw:
                    self.data_queue.put(raw, timeout=0.1)
            except (zmq.Again, Full):
                continue

    def _processing_worker(self):
        while self.running.is_set():
            try:
                raw = self.data_queue.get(timeout=0.5)
                sample = self._parse_sample(raw)
                features = self._extract_features(sample)
                freq = sample.get("frequency_mhz")
                self.feature_vectors[freq].append(features)

                self.calibration_progress = min(1.0, sum(len(v) for v in self.feature_vectors.values()) / (
                            self.calibration_sweeps * len(self.feature_vectors)))
                if self.calibration_progress >= 1.0 and not self.calibration_complete:
                    self._finalize_calibration()
            except Empty:
                continue

    def _parse_sample(self, raw: bytes) -> Dict[str, Any]:
        return json.loads(raw.decode("utf-8"))

    def _extract_features(self, sample: Dict[str, Any]) -> np.ndarray:
        signal_data = np.array(sample.get("data", []))
        features = [
            np.mean(signal_data),
            np.std(signal_data),
            np.max(signal_data),
            np.min(signal_data),
            np.median(signal_data),
            skew(signal_data),
            kurtosis(signal_data),
            np.percentile(signal_data, 25),
            np.percentile(signal_data, 75),
            np.ptp(signal_data)
        ]
        return np.array(features)

    def _finalize_calibration(self):
        all_vectors = []
        for freq in self.feature_vectors:
            all_vectors.extend(self.feature_vectors[freq])
        if len(all_vectors) < MIN_SAMPLES_FOR_CALIBRATION:
            calibration_logger.warning("Yeterli veri toplanamadı.")
            return

        all_vectors = np.array(all_vectors)
        self.scaler.fit(all_vectors)
        avg_vector = np.mean(all_vectors, axis=0)
        ground_type = self._classify_ground_type(avg_vector)
        self.ground_type = ground_type
        self.estimated_permittivity = self._estimate_permittivity(avg_vector)

        result = {
            "ground_type": self.ground_type.name,
            "timestamp": datetime.utcnow().isoformat(),
            "estimated_permittivity": self.estimated_permittivity
        }
        self._save_calibration_result(result)
        self._publish_calibration_result(result)
        self.calibration_complete = True
        calibration_logger.info(f"Kalibrasyon tamamlandı: {result}")

    def _classify_ground_type(self, feature_vector: np.ndarray) -> GroundType:
        distances = {}
        for ground_type, signature in self.reference_signatures.items():
            dist = np.linalg.norm(
                self.scaler.transform([feature_vector])[0] - self.scaler.transform([signature])[0])
            distances[ground_type] = dist
        return min(distances, key=distances.get)

    def _estimate_permittivity(self, feature_vector: np.ndarray) -> float:
        return np.clip(1 + feature_vector[2] * 0.1, 1, 20)

    def _publish_calibration_result(self, result: Dict[str, Any]):
        self.result_socket.send_json(result)

    def _noise_monitoring_worker(self):
        while not self.shutdown_event.is_set():
            self.background_noise_db = self._measure_background_noise()
            time.sleep(self.noise_monitor_interval_sec)

    def _measure_background_noise(self) -> float:
        return np.random.uniform(-110, -90)

    def _control_worker(self):
        while self.running.is_set():
            try:
                msg = self.control_socket.recv_string()
                response = self._handle_control_command(msg)
                self.control_socket.send_json(response)
            except zmq.Again:
                continue

    def _handle_control_command(self, cmd: str) -> Dict[str, Any]:
        if cmd == "status":
            return {
                "running": self.running.is_set(),
                "complete": self.calibration_complete,
                "ground_type": self.ground_type.name,
                "progress": self.calibration_progress
            }
        elif cmd == "stop":
            self.stop()
            return {"status": "stopped"}
        return {"error": "unknown_command"}

    def is_alive(self):
        return self.running.is_set()

    def _run_calibration_sequence(self):
        """Kalibrasyon dizisini otomatik başlatır"""
        self.calibration_complete = False
        self.feature_vectors.clear()
        self.calibration_data_buffer.clear()
        start_time = time.time()
        while time.time() - start_time < self.calibration_time_sec and not self.calibration_complete:
            time.sleep(0.5)

def main():
    parser = argparse.ArgumentParser(description="Ground Calibration Module")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to system JSON configuration file")
    parser.add_argument("--run-cal", action="store_true",
                        help="Start calibration sequence immediately after launch")

    args = parser.parse_args()
    config = {}
    if args.config:
        try:
            with open(args.config, "r") as f:
                file_config = json.load(f)
                config = file_config.get("module_configs", {}).get("calibration", {})
                calibration_logger.info(f"Yapılandırma dosyası yüklendi: {args.config}")
        except FileNotFoundError:
            calibration_logger.error(f"Yapılandırma dosyası bulunamadı: {args.config}")
        except json.JSONDecodeError as e:
            calibration_logger.error(f"Yapılandırma dosyası okunamadı: {e}")

    calibration_module = CalibrationModule(config)

    def shutdown_handler(signum, frame):
        print("\nKapatma sinyali alındı...")
        calibration_module.stop()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    if calibration_module.start():
        calibration_logger.info("Kalibrasyon Modülü çalışıyor.")
        if args.run_cal:
            calibration_logger.info("Otomatik kalibrasyon başlatılıyor...")
            auto_cal_thread = Thread(target=calibration_module._run_calibration_sequence, name="AutoCalRunner")
            auto_cal_thread.daemon = True
            auto_cal_thread.start()

        while calibration_module.running.is_set():
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                break
    else:
        calibration_logger.critical("Kalibrasyon Modülü başlatılamadı.")

    calibration_logger.info("Kalibrasyon Modülü programı sonlandı.")

if __name__ == "__main__":
    main()