#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
logger.py - Gelişmiş SDR Yeraltı Tespit Sistemi Logger Modülü

Bu modül, sistemin tüm verilerini toplar, işler ve analiz eder.
Profesyonel seviyede kayıt tutma, analiz ve raporlama özellikleri içerir.
"""

import os
import sys
import time
import json
import argparse
import logging
import threading
import queue
import zmq
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import hashlib
import gzip
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import csv

# Yapılandırılmış logging
import structlog
from opentelemetry import trace

# Sistem yollarını ayarla
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Yerel modüller
from utilities.data_models import Detection, CalibrationData, SDRData
from utilities.database import DatabaseManager
from utilities.performance import PerformanceMonitor
from utilities.security import DataEncryptor

# Logging konfigürasyonu
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class LogLevel(Enum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class DetectionType(Enum):
    METAL = auto()
    VOID = auto()
    MINERAL = auto()
    UNKNOWN = auto()


class DataSource(Enum):
    SDR = auto()
    CALIBRATION = auto()
    AI = auto()
    CONTROL = auto()


@dataclass
class SystemStats:
    """Sistem istatistiklerini tutan veri yapısı"""
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    sdr_count: int = 0
    calibration_count: int = 0
    ai_count: int = 0
    detection_count: int = 0
    false_positive_count: int = 0
    true_positive_count: int = 0
    detection_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    detection_by_confidence: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


class Logger:
    """Gelişmiş veri kayıt ve analiz sistemi"""

    def __init__(self, args):
        self.args = args
        self.running = threading.Event()
        self.data_queue = queue.Queue(maxsize=5000)
        self.context = zmq.Context()
        self.tracer = trace.get_tracer(__name__)

        # Yapılandırma
        self._setup_directories()
        self._init_components()

        logger.info("Logger initialized", config=vars(args))

    def _setup_directories(self):
        """Gerekli dizin yapısını oluşturur"""
        self.log_dir = Path(self.args.log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Alt dizinler
        self.subdirs = {
            DataSource.SDR: self.log_dir / "sdr",
            DataSource.CALIBRATION: self.log_dir / "calibration",
            DataSource.AI: self.log_dir / "ai",
            "detections": self.log_dir / "detections",
            "stats": self.log_dir / "stats",
            "plots": self.log_dir / "plots",
            "backups": self.log_dir / "backups"
        }

        for dir_path in self.subdirs.values():
            dir_path.mkdir(exist_ok=True)

        # Günlük dosya yolları
        self.current_date = datetime.now().strftime("%Y%m%d")
        self.log_files = {
            DataSource.SDR: self.subdirs[DataSource.SDR] / f"sdr_{self.current_date}.jsonl",
            DataSource.CALIBRATION: self.subdirs[DataSource.CALIBRATION] / f"calibration_{self.current_date}.jsonl",
            DataSource.AI: self.subdirs[DataSource.AI] / f"ai_{self.current_date}.jsonl",
            "detections": self.subdirs["detections"] / f"detections_{self.current_date}.jsonl",
            "stats": self.subdirs["stats"] / f"stats_{self.current_date}.json"
        }

    def _init_components(self):
        """Sistem bileşenlerini başlatır"""
        # Veritabanı yöneticisi
        self.db_manager = DatabaseManager(self.args.db_file)

        # Performans izleyici
        self.performance_monitor = PerformanceMonitor()

        # Güvenlik bileşeni
        self.encryptor = DataEncryptor(self.args.encryption_key)

        # İstatistikler
        self.stats = SystemStats()

        # Tespit geçmişi
        self.detection_history = deque(maxlen=1000)
        self.detection_map = defaultdict(list)

        # ZMQ soketleri
        self.sockets = {
            DataSource.SDR: self.context.socket(zmq.SUB),
            DataSource.CALIBRATION: self.context.socket(zmq.SUB),
            DataSource.AI: self.context.socket(zmq.SUB),
            DataSource.CONTROL: self.context.socket(zmq.SUB)
        }

        # Soket bağlantıları
        self.sockets[DataSource.SDR].connect(f"tcp://localhost:{self.args.sdr_port}")
        self.sockets[DataSource.SDR].setsockopt_string(zmq.SUBSCRIBE, "")

        self.sockets[DataSource.CALIBRATION].connect(f"tcp://localhost:{self.args.calibration_port}")
        self.sockets[DataSource.CALIBRATION].setsockopt_string(zmq.SUBSCRIBE, "")

        self.sockets[DataSource.AI].connect(f"tcp://localhost:{self.args.ai_port}")
        self.sockets[DataSource.AI].setsockopt_string(zmq.SUBSCRIBE, "")

        self.sockets[DataSource.CONTROL].connect(f"tcp://localhost:{self.args.control_port}")
        self.sockets[DataSource.CONTROL].setsockopt_string(zmq.SUBSCRIBE, "")

    def start(self):
        """Sistemi başlatır"""
        with self.tracer.start_as_current_span("logger_start"):
            self.running.set()

            # İş parçacıklarını başlat
            self.threads = {
                "sdr_monitor": threading.Thread(target=self._sdr_monitor, daemon=True),
                "calibration_monitor": threading.Thread(target=self._calibration_monitor, daemon=True),
                "ai_monitor": threading.Thread(target=self._ai_monitor, daemon=True),
                "control_monitor": threading.Thread(target=self._control_monitor, daemon=True),
                "data_processor": threading.Thread(target=self._process_data, daemon=True),
                "stats_processor": threading.Thread(target=self._process_stats, daemon=True),
                "archiver": threading.Thread(target=self._archive_data, daemon=True)
            }

            for name, thread in self.threads.items():
                thread.start()
                logger.info(f"Thread started", thread_name=name)

            logger.info("Logger system started")
            return True

    def stop(self):
        """Sistemi durdurur"""
        with self.tracer.start_as_current_span("logger_stop"):
            self.running.clear()

            # İş parçacıklarını durdur
            for name, thread in self.threads.items():
                if thread.is_alive():
                    thread.join(timeout=5.0)
                    logger.info(f"Thread stopped", thread_name=name)

            # Kaynakları temizle
            for socket in self.sockets.values():
                socket.close()

            self.context.term()
            self.db_manager.close()

            logger.info("Logger system stopped")
            return True

    def _sdr_monitor(self):
        """SDR verilerini izler"""
        while self.running.is_set():
            with self.tracer.start_as_current_span("sdr_monitoring"):
                try:
                    try:
                        data = self.sockets[DataSource.SDR].recv_json(flags=zmq.NOBLOCK)
                        self.data_queue.put((DataSource.SDR, data))
                    except zmq.Again:
                        pass

                    time.sleep(0.01)

                except Exception as e:
                    logger.error("SDR monitoring error", error=str(e))
                    self.stats.error_count["sdr_monitor"] += 1
                    time.sleep(1)

    def _calibration_monitor(self):
        """Kalibrasyon verilerini izler"""
        while self.running.is_set():
            with self.tracer.start_as_current_span("calibration_monitoring"):
                try:
                    try:
                        data = self.sockets[DataSource.CALIBRATION].recv_json(flags=zmq.NOBLOCK)
                        self.data_queue.put((DataSource.CALIBRATION, data))
                    except zmq.Again:
                        pass

                    time.sleep(0.01)

                except Exception as e:
                    logger.error("Calibration monitoring error", error=str(e))
                    self.stats.error_count["calibration_monitor"] += 1
                    time.sleep(1)

    def _ai_monitor(self):
        """AI verilerini izler"""
        while self.running.is_set():
            with self.tracer.start_as_current_span("ai_monitoring"):
                try:
                    try:
                        data = self.sockets[DataSource.AI].recv_json(flags=zmq.NOBLOCK)
                        self.data_queue.put((DataSource.AI, data))
                    except zmq.Again:
                        pass

                    time.sleep(0.01)

                except Exception as e:
                    logger.error("AI monitoring error", error=str(e))
                    self.stats.error_count["ai_monitor"] += 1
                    time.sleep(1)

    def _control_monitor(self):
        """Kontrol verilerini izler"""
        while self.running.is_set():
            with self.tracer.start_as_current_span("control_monitoring"):
                try:
                    try:
                        data = self.sockets[DataSource.CONTROL].recv_json(flags=zmq.NOBLOCK)
                        self.data_queue.put((DataSource.CONTROL, data))
                    except zmq.Again:
                        pass

                    time.sleep(0.01)

                except Exception as e:
                    logger.error("Control monitoring error", error=str(e))
                    self.stats.error_count["control_monitor"] += 1
                    time.sleep(1)

    def _process_data(self):
        """Verileri işler"""
        while self.running.is_set():
            with self.tracer.start_as_current_span("data_processing"):
                try:
                    source, data = self.data_queue.get(timeout=0.1)

                    if source == DataSource.SDR:
                        self._handle_sdr_data(data)
                    elif source == DataSource.CALIBRATION:
                        self._handle_calibration_data(data)
                    elif source == DataSource.AI:
                        self._handle_ai_data(data)
                    elif source == DataSource.CONTROL:
                        self._handle_control_data(data)

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error("Data processing error", error=str(e))
                    self.stats.error_count["data_processor"] += 1

    def _handle_sdr_data(self, data):
        """SDR verilerini işler"""
        with self.tracer.start_as_current_span("process_sdr_data"):
            try:
                # Veri modelini oluştur
                sdr_data = SDRData(**data)

                # Dosyaya yaz
                self._write_to_file(self.log_files[DataSource.SDR], sdr_data.dict())

                # Veritabanına kaydet
                self.db_manager.save_sdr_data(sdr_data)

                # İstatistikleri güncelle
                self.stats.sdr_count += 1
                self.stats.last_update = datetime.now()

            except Exception as e:
                logger.error("SDR data handling error", error=str(e))
                self.stats.error_count["sdr_data"] += 1

    def _handle_calibration_data(self, data):
        """Kalibrasyon verilerini işler"""
        with self.tracer.start_as_current_span("process_calibration_data"):
            try:
                # Veri modelini oluştur
                cal_data = CalibrationData(**data)

                # Dosyaya yaz
                self._write_to_file(self.log_files[DataSource.CALIBRATION], cal_data.dict())

                # Veritabanına kaydet
                self.db_manager.save_calibration_data(cal_data)

                # İstatistikleri güncelle
                self.stats.calibration_count += 1
                self.stats.last_update = datetime.now()

            except Exception as e:
                logger.error("Calibration data handling error", error=str(e))
                self.stats.error_count["calibration_data"] += 1

    def _handle_ai_data(self, data):
        """AI verilerini işler"""
        with self.tracer.start_as_current_span("process_ai_data"):
            try:
                # Tespitleri işle
                for detection_data in data.get("detections", []):
                    self._handle_detection(detection_data)

                # İstatistikleri güncelle
                self.stats.ai_count += 1
                self.stats.last_update = datetime.now()

            except Exception as e:
                logger.error("AI data handling error", error=str(e))
                self.stats.error_count["ai_data"] += 1

    def _handle_detection(self, detection_data):
        """Tespit verilerini işler"""
        with self.tracer.start_as_current_span("process_detection"):
            try:
                # Veri modelini oluştur
                detection = Detection(**detection_data)

                # Dosyaya yaz
                self._write_to_file(self.log_files["detections"], detection.dict())

                # Veritabanına kaydet
                self.db_manager.save_detection(detection)

                # İstatistikleri güncelle
                self._update_detection_stats(detection)

                # Tespit geçmişine ekle
                self.detection_history.append(detection)

                # Konum haritasını güncelle
                self._update_detection_map(detection)

            except Exception as e:
                logger.error("Detection handling error", error=str(e))
                self.stats.error_count["detection_data"] += 1

    def _update_detection_stats(self, detection: Detection):
        """Tespit istatistiklerini günceller"""
        self.stats.detection_count += 1

        # Tip bazlı istatistik
        det_type = detection.type.value if isinstance(detection.type, Enum) else detection.type
        self.stats.detection_by_type[det_type] += 1

        # Güven seviyesi bazlı istatistik
        conf_level = int(detection.confidence * 10)
        conf_range = f"{conf_level}0-{conf_level + 1}0%"
        self.stats.detection_by_confidence[conf_range] += 1

        # Doğruluk istatistikleri
        if detection.is_false_positive:
            self.stats.false_positive_count += 1
        else:
            self.stats.true_positive_count += 1

        self.stats.last_update = datetime.now()

    def _update_detection_map(self, detection: Detection):
        """Tespit haritasını günceller"""
        pos_key = f"{detection.position.x:.1f}_{detection.position.y:.1f}"
        self.detection_map[pos_key].append(detection)

    def _handle_control_data(self, data):
        """Kontrol verilerini işler"""
        with self.tracer.start_as_current_span("process_control_data"):
            try:
                # Sistem komutlarını işle
                command = data.get("command")

                if command == "shutdown":
                    logger.info("Received shutdown command")
                    self.stop()
                elif command == "get_stats":
                    return self._get_current_stats()
                elif command == "export_data":
                    self._export_data(data.get("format", "json"))

            except Exception as e:
                logger.error("Control data handling error", error=str(e))
                self.stats.error_count["control_data"] += 1

    def _process_stats(self):
        """İstatistikleri işler"""
        while self.running.is_set():
            with self.tracer.start_as_current_span("stats_processing"):
                try:
                    # İstatistikleri kaydet
                    self._save_stats()

                    # CSV dışa aktarımı
                    if self.args.csv_export:
                        self._export_stats_to_csv()

                    # Grafik oluştur
                    if self.args.plot_stats:
                        self._generate_stat_plots()

                    # Performansı izle
                    self._monitor_performance()

                    # 5 dakikada bir çalış
                    time.sleep(300)

                except Exception as e:
                    logger.error("Stats processing error", error=str(e))
                    self.stats.error_count["stats_processor"] += 1
                    time.sleep(60)

    def _save_stats(self):
        """İstatistikleri kaydeder"""
        stats_file = self.subdirs["stats"] / f"stats_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(stats_file, 'w') as f:
            json.dump({
                "start_time": self.stats.start_time.isoformat(),
                "last_update": self.stats.last_update.isoformat(),
                "sdr_count": self.stats.sdr_count,
                "calibration_count": self.stats.calibration_count,
                "ai_count": self.stats.ai_count,
                "detection_count": self.stats.detection_count,
                "false_positive_count": self.stats.false_positive_count,
                "true_positive_count": self.stats.true_positive_count,
                "detection_by_type": dict(self.stats.detection_by_type),
                "detection_by_confidence": dict(self.stats.detection_by_confidence),
                "error_count": dict(self.stats.error_count)
            }, f, indent=2)

    def _export_stats_to_csv(self):
        """İstatistikleri CSV'ye aktarır"""
        csv_file = self.subdirs["stats"] / f"stats_{datetime.now().strftime('%Y%m%d')}.csv"

        data = {
            "timestamp": datetime.now().isoformat(),
            "sdr_count": self.stats.sdr_count,
            "calibration_count": self.stats.calibration_count,
            "ai_count": self.stats.ai_count,
            "detection_count": self.stats.detection_count,
            "false_positive_count": self.stats.false_positive_count,
            "true_positive_count": self.stats.true_positive_count
        }

        # CSV başlık kontrolü
        write_header = not csv_file.exists()

        with open(csv_file, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(data)

    def _generate_stat_plots(self):
        """İstatistik grafikleri oluşturur"""
        try:
            # Tespit tipi dağılımı
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=list(self.stats.detection_by_type.keys()),
                y=list(self.stats.detection_by_type.values())
            )
            plt.title("Detection Type Distribution")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.subdirs["plots"] / "detection_types.png")
            plt.close()

            # Güven seviyesi dağılımı
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=list(self.stats.detection_by_confidence.keys()),
                y=list(self.stats.detection_by_confidence.values())
            )
            plt.title("Detection Confidence Distribution")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.subdirs["plots"] / "detection_confidence.png")
            plt.close()

        except Exception as e:
            logger.error("Plot generation error", error=str(e))

    def _monitor_performance(self):
        """Sistem performansını izler"""
        cpu_usage = self.performance_monitor.get_cpu_usage()
        mem_usage = self.performance_monitor.get_memory_usage()

        if cpu_usage > 90:
            logger.warning("High CPU usage", cpu_usage=cpu_usage)
        if mem_usage > 90:
            logger.warning("High memory usage", memory_usage=mem_usage)

    def _archive_data(self):
        """Verileri arşivler"""
        while self.running.is_set():
            try:
                # Eski logları temizle
                self._cleanup_old_logs()

                # Veritabanı yedeği al
                self._backup_database()

                # 24 saatte bir çalış
                time.sleep(24 * 60 * 60)

            except Exception as e:
                logger.error("Archiving error", error=str(e))
                time.sleep(60)

    def _cleanup_old_logs(self):
        """Eski log dosyalarını temizler"""
        cutoff_date = datetime.now() - timedelta(days=self.args.retention_days)

        for dir_path in self.subdirs.values():
            for file_path in dir_path.glob("*"):
                if file_path.is_file() and datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_date:
                    try:
                        if self.args.compress_before_delete:
                            self._compress_file(file_path)
                        file_path.unlink()
                        logger.info("Deleted old file", file_path=str(file_path))
                    except Exception as e:
                        logger.error("File deletion error", file_path=str(file_path), error=str(e))

    def _compress_file(self, file_path: Path):
        """Dosyayı sıkıştırır"""
        compressed_path = file_path.with_suffix(file_path.suffix + ".gz")
        try:
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info("File compressed", original=str(file_path), compressed=str(compressed_path))
        except Exception as e:
            logger.error("File compression error", file_path=str(file_path), error=str(e))

    def _backup_database(self):
        """Veritabanı yedeği alır"""
        backup_path = self.subdirs["backups"] / f"detections_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.db"
        try:
            self.db_manager.backup(backup_path)
            logger.info("Database backup created", backup_path=str(backup_path))
        except Exception as e:
            logger.error("Database backup error", error=str(e))

    def _write_to_file(self, file_path: Path, data: Dict):
        """Veriyi dosyaya yazar"""
        try:
            with open(file_path, 'a') as f:
                json.dump(data, f)
                f.write('\n')
        except Exception as e:
            logger.error("File write error", file_path=str(file_path), error=str(e))

    def _get_current_stats(self) -> Dict:
        """Güncel istatistikleri döndürür"""
        return {
            "status": "running",
            "uptime": str(datetime.now() - self.stats.start_time),
            "stats": {
                "sdr_count": self.stats.sdr_count,
                "calibration_count": self.stats.calibration_count,
                "ai_count": self.stats.ai_count,
                "detection_count": self.stats.detection_count,
                "false_positive_rate": (
                    self.stats.false_positive_count / self.stats.detection_count
                    if self.stats.detection_count > 0 else 0
                )
            }
        }

    def _export_data(self, format: str = "json"):
        """Verileri dışa aktarır"""
        export_dir = self.log_dir / "exports"
        export_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        if format == "json":
            export_path = export_dir / f"export_{timestamp}.json"
            data = {
                "stats": self._get_current_stats(),
                "recent_detections": [d.dict() for d in list(self.detection_history)[-100:]]
            }
            with open(export_path, 'w') as f:
                json.dump(data, f, indent=2)

        elif format == "csv":
            export_path = export_dir / f"export_{timestamp}.csv"
            detections = list(self.detection_history)[-100:]
            if detections:
                df = pd.DataFrame([d.dict() for d in detections])
                df.to_csv(export_path, index=False)

        logger.info("Data exported", format=format, path=str(export_path))
        return str(export_path)


def parse_arguments():
    """Komut satırı argümanlarını ayrıştırır"""
    parser = argparse.ArgumentParser(
        description="SDR Tabanlı Yeraltı Tespit Sistemi Logger",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Temel ayarlar
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Log dosyalarının kaydedileceği dizin"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Log seviyesi"
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=30,
        help="Logların saklanma süresi (gün)"
    )
    parser.add_argument(
        "--compress-before-delete",
        action="store_true",
        help="Silmeden önce dosyaları sıkıştır"
    )

    # Veritabanı ayarları
    parser.add_argument(
        "--db-file",
        type=str,
        default="logs/detections.db",
        help="SQLite veritabanı dosya yolu"
    )
    parser.add_argument(
        "--backup-interval",
        type=int,
        default=24,
        help="Veritabanı yedekleme aralığı (saat)"
    )

    # Dışa aktarma ayarları
    parser.add_argument(
        "--csv-export",
        action="store_true",
        help="CSV dışa aktarımı yap"
    )
    parser.add_argument(
        "--plot-stats",
        action="store_true",
        help="İstatistik grafikleri oluştur"
    )
    parser.add_argument(
        "--export-interval",
        type=int,
        default=60,
        help="Dışa aktarma aralığı (dakika)"
    )

    # ZMQ bağlantı ayarları
    parser.add_argument(
        "--sdr-port",
        type=int,
        default=5555,
        help="SDR modülü ZMQ portu"
    )
    parser.add_argument(
        "--calibration-port",
        type=int,
        default=5557,
        help="Kalibrasyon modülü ZMQ portu"
    )
    parser.add_argument(
        "--ai-port",
        type=int,
        default=5559,
        help="AI modülü ZMQ portu"
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=5563,
        help="Kontrol merkezi ZMQ portu"
    )

    # Güvenlik ayarları
    parser.add_argument(
        "--encryption-key",
        type=str,
        default="",
        help="Veri şifreleme anahtarı"
    )

    return parser.parse_args()


def main():
    """Ana uygulama giriş noktası"""
    args = parse_arguments()

    try:
        logger = Logger(args)
        logger.start()

        # Ana döngü
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.stop()
    except Exception as e:
        logger.critical("System error", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()