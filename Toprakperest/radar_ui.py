#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
radar_ui.py - Gelişmiş Radar Arayüzü

SDR tabanlı yeraltı tespit sistemi için tam özellikli 2D/3D radar arayüzü.

Özellikler:
- Gerçek zamanlı veri simülasyon modu
- 2D ve 3D görüntüleme desteği
- Çoklu dil desteği (İngilizce, Türkçe)
- Gelişmiş veri işleme ve filtreleme
- Plugin sistemi ile genişletilebilirlik
- Detaylı loglama ve hata yönetimi
- Performans izleme ve optimizasyon
"""

import os
import sys
import time
import argparse
import numpy as np
import zmq
import json
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any
from datetime import datetime
from threading import Thread, Event, Lock
from queue import Queue, Empty
import matplotlib
import signal

# GUI backend seçimi
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, Polygon, Wedge
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Sabitler
DEFAULT_UPDATE_INTERVAL = 100  # ms
DEFAULT_MAX_RANGE = 50  # metre
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
MAX_QUEUE_SIZE = 1000
MAX_HISTORY_LENGTH = 100
SIMULATION_DATA_RATE = 0.1  # saniye


# Enumlar
class DetectionType(Enum):
    METAL = auto()
    VOID = auto()
    MINERAL = auto()
    UNKNOWN = auto()


class GroundType(Enum):
    UNKNOWN = auto()
    SOIL = auto()
    ROCK = auto()
    SAND = auto()
    CLAY = auto()


class VisualizationMode(Enum):
    _2D = auto()
    _3D = auto()


class Language(Enum):
    ENGLISH = "en"
    TURKISH = "tr"


# Veri yapıları
@dataclass
class Detection:
    angle: float  # derece
    distance: float  # metre
    confidence: float  # 0.0-1.0
    type: DetectionType
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundProperties:
    type: GroundType
    density: float  # g/cm³
    conductivity: float  # S/m
    permittivity: float  # Dielectric sabiti


# Dil desteği
TRANSLATIONS = {
    Language.ENGLISH: {
        "window_title": "Advanced Radar UI",
        "start": "Start",
        "stop": "Stop",
        "file": "File",
        "view": "View",
        "help": "Help",
        "status_ready": "Ready",
        "status_scanning": "Scanning...",
    },
    Language.TURKISH: {
        "window_title": "Gelişmiş Radar Arayüzü",
        "start": "Başlat",
        "stop": "Durdur",
        "file": "Dosya",
        "view": "Görünüm",
        "help": "Yardım",
        "status_ready": "Hazır",
        "status_scanning": "Taranıyor...",
    }
}

# PyQt5 importları
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QLabel, QPushButton, QComboBox,
                                 QSlider, QCheckBox, QGroupBox, QGridLayout,
                                 QTabWidget, QSplitter, QFrame, QFileDialog,
                                 QMessageBox, QDockWidget, QToolBar, QAction,
                                 QStatusBar, QSizePolicy, QProgressBar)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize, QThread, QLocale, QObject
    from PyQt5.QtGui import QIcon, QFont, QColor, QPalette, QPixmap, QKeySequence

    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

# PyQtGraph importları
try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl

    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False


# Logging konfigürasyonu
def setup_logging():
    """Gelişmiş logging konfigürasyonu"""
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger("radar_ui")
    logger.setLevel(logging.DEBUG)

    # Dosya handler'ı
    file_handler = logging.FileHandler("radar_ui.log", encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Konsol handler'ı
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)

    # Dönen log handler'ı
    rotating_handler = logging.handlers.RotatingFileHandler(
        "radar_ui_rotating.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    rotating_handler.setFormatter(log_formatter)
    rotating_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(rotating_handler)

    return logger


logger = setup_logging()


class DataFilter:
    """Veri filtreleme sınıfı"""

    @staticmethod
    def kalman_filter(data, process_noise=0.01, measurement_noise=0.1):
        """Basit Kalman filtresi uygular"""
        if not data:
            return []

        estimated = data[0]
        estimation_error = 1.0
        filtered_data = []

        for value in data:
            # Tahmin adımı
            estimation_error += process_noise

            # Güncelleme adımı
            kalman_gain = estimation_error / (estimation_error + measurement_noise)
            estimated += kalman_gain * (value - estimated)
            estimation_error *= (1 - kalman_gain)

            filtered_data.append(estimated)

        return filtered_data

    @staticmethod
    def moving_average(data, window_size=3):
        """Hareketli ortalama filtresi"""
        if len(data) < window_size:
            return data

        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)


class PluginManager:
    """Plugin yönetim sistemi"""

    def __init__(self):
        self.plugins = []
        self.lock = Lock()

    def register_plugin(self, plugin):
        """Yeni plugin ekler"""
        with self.lock:
            self.plugins.append(plugin)
            logger.info(f"Plugin registered: {plugin.__class__.__name__}")

    def notify_plugins(self, event, *args, **kwargs):
        """Plugınlara event bildirir"""
        with self.lock:
            for plugin in self.plugins:
                try:
                    getattr(plugin, f"on_{event}")(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Plugin notification error: {str(e)}")


class BasePlugin:
    """Tüm pluginler için temel sınıf"""

    def __init__(self, radar_ui):
        self.radar_ui = radar_ui

    def on_detection(self, detection):
        """Yeni tespit algılandığında çağrılır"""
        pass

    def on_start(self):
        """Sistem başladığında çağrılır"""
        pass

    def on_stop(self):
        """Sistem durduğunda çağrılır"""
        pass


class PerformanceMonitor:
    """Performans izleme sınıfı"""

    def __init__(self):
        self.frame_times = []
        self.start_time = time.time()
        self.frame_count = 0

    def start_frame(self):
        """Yeni frame başlangıcını işaretler"""
        self.frame_start = time.time()

    def end_frame(self):
        """Frame sonunu işaretler ve istatistikleri günceller"""
        frame_time = time.time() - self.frame_start
        self.frame_times.append(frame_time)
        self.frame_count += 1

        # Eski verileri temizle
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)

        return frame_time

    def get_fps(self):
        """Saniyedeki frame sayısını hesaplar"""
        if not self.frame_times:
            return 0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0

    def get_stats(self):
        """Performans istatistiklerini döndürür"""
        uptime = time.time() - self.start_time
        fps = self.get_fps()
        return {
            "uptime": uptime,
            "fps": fps,
            "frame_count": self.frame_count,
            "avg_frame_time": 1.0 / fps if fps > 0 else 0
        }


class RadarUI(QObject if PYQT_AVAILABLE else object):
    """Gelişmiş Radar Arayüzü Sınıfı"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self._validate_args()
        self._init_settings()
        self._init_data_structures()
        self._init_zmq()
        self._setup_directories()

        # Performans izleme
        self.performance_monitor = PerformanceMonitor()

        # Plugin yöneticisi
        self.plugin_manager = PluginManager()

        # Sinyal işleme
        if PYQT_AVAILABLE:
            self.update_signal = pyqtSignal()

        logger.info("Radar UI initialized")

    def _validate_args(self):
        """Argümanları doğrular"""
        if not 0 < self.args.confidence_threshold <= 1.0:
            raise ValueError("Güven eşiği 0-1 arasında olmalıdır")
        if self.args.max_range <= 0:
            raise ValueError("Maksimum menzil pozitif olmalıdır")

    def _init_settings(self):
        """Ayarları başlatır"""
        self.running = Event()
        self.simulation_mode = self.args.simulation
        self.visualization_mode = VisualizationMode._3D if self.args._3d_mode else VisualizationMode._2D
        self.language = Language(self.args.language)
        self.dark_mode = self.args.dark_mode
        self.radar_angle = 0
        self.radar_speed = 2  # derece/güncelleme

    def _init_data_structures(self):
        """Veri yapılarını başlatır"""
        self.data_queue = Queue(maxsize=MAX_QUEUE_SIZE)
        self.detections = []
        self.ground_properties = GroundProperties(GroundType.UNKNOWN, 0, 0, 0)
        self.scan_history = []
        self.lock = Lock()

    def _init_zmq(self):
        """ZMQ bağlantılarını başlatır"""
        self.context = zmq.Context()

        if not self.simulation_mode:
            self._init_zmq_real_mode()
        else:
            logger.info("Simülasyon modu aktif - ZMQ bağlantıları devre dışı")

    def _init_zmq_real_mode(self):
        """Gerçek mod için ZMQ soketlerini ayarlar"""
        # SDR veri soketi
        self.sdr_socket = self.context.socket(zmq.SUB)
        self.sdr_socket.connect(f"tcp://localhost:{self.args.sdr_port}")
        self.sdr_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # AI sınıflandırma soketi
        self.ai_socket = self.context.socket(zmq.SUB)
        self.ai_socket.connect(f"tcp://localhost:{self.args.ai_port}")
        self.ai_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Kontrol soketi
        self.control_socket = self.context.socket(zmq.REP)
        self.control_socket.bind(f"tcp://*:{self.args.control_port}")

    def _setup_directories(self):
        """Gerekli dizinleri oluşturur"""
        os.makedirs(self.args.save_path, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("plugins", exist_ok=True)

    def start(self):
        """Radar sistemini başlatır"""
        try:
            logger.info("Radar sistemini başlatılıyor...")

            # Pluginleri yükle
            self._load_plugins()

            # Thread'leri başlat
            self._start_threads()

            # UI'ı başlat
            self._start_ui()

            # Plugin'lere bildirim gönder
            self.plugin_manager.notify_plugins("start")

            logger.info("Radar sistemi başarıyla başlatıldı")
            return True

        except Exception as e:
            logger.critical(f"Radar başlatma hatası: {str(e)}")
            self.stop()
            return False

    def _load_plugins(self):
        """Pluginleri yükler"""

        # Örnek bir plugin ekleyelim
        class ExamplePlugin(BasePlugin):
            def on_detection(self, detection):
                logger.debug(f"ExamplePlugin: New detection at {detection.angle}°")

        self.plugin_manager.register_plugin(ExamplePlugin(self))

        # plugins dizinindeki diğer pluginleri yükle
        # ...

    def _start_threads(self):
        """Tüm çalışan thread'leri başlatır"""
        if not self.simulation_mode:
            self._start_real_mode_threads()
        else:
            self._start_simulation_thread()

    def _start_real_mode_threads(self):
        """Gerçek mod için thread'leri başlatır"""
        self.running.set()

        threads = [
            Thread(target=self._sdr_worker, daemon=True, name="SDRWorker"),
            Thread(target=self._ai_worker, daemon=True, name="AIWorker"),
            Thread(target=self._control_worker, daemon=True, name="ControlWorker"),
            Thread(target=self._data_processor, daemon=True, name="DataProcessor")
        ]

        for t in threads:
            t.start()

        logger.info(f"{len(threads)} çalışan thread başlatıldı")

    def _start_simulation_thread(self):
        """Simülasyon thread'ini başlatır"""
        self.running.set()
        Thread(target=self._simulation_worker, daemon=True, name="SimulationWorker").start()
        logger.info("Simülasyon thread'i başlatıldı")

    def _start_ui(self):
        """UI'ı başlatır"""
        if PYQT_AVAILABLE:
            self._start_qt_interface()
        else:
            self._start_matplotlib_interface()

    def _start_qt_interface(self):
        """PyQt tabanlı arayüzü başlatır"""
        self.app = QApplication(sys.argv)

        # Karanlık mod ayarı
        if self.dark_mode:
            self._apply_dark_theme()

        self.main_window = RadarMainWindow(self)

        # Tam ekran modu
        if self.args.fullscreen:
            self.main_window.showFullScreen()
        else:
            self.main_window.show()

        # Sinyal bağlantıları
        if hasattr(self, 'update_signal'):
            self.update_signal.connect(self.main_window.update_plot)

        # Graceful shutdown için sinyal handler
        signal.signal(signal.SIGINT, self._handle_signal)

        # Uygulamayı başlat
        sys.exit(self.app.exec_())

    def _start_matplotlib_interface(self):
        """Matplotlib tabanlı basit arayüzü başlatır"""
        self.fig, self.ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})
        self.fig.canvas.manager.set_window_title('SDR Radar Görüntüleyici')

        # Tam ekran modu
        if self.args.fullscreen:
            plt.get_current_fig_manager().full_screen_toggle()

        # Radar görünümünü ayarla
        self._setup_radar_view()

        # Animasyonu başlat
        self.animation = FuncAnimation(
            self.fig, self._update_plot,
            interval=self.args.update_interval,
            blit=False
        )

        # Görselleştirmeyi göster
        plt.show()

    def _apply_dark_theme(self):
        """Karanlık tema uygular"""
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        QApplication.setPalette(dark_palette)

    def _handle_signal(self, signum, frame):
        """Sinyal işleme fonksiyonu"""
        logger.info(f"{signal.Signals(signum).name} sinyali alındı, kapatılıyor...")
        self.stop()
        QApplication.quit()

    def stop(self):
        """Radar sistemini durdurur"""
        logger.info("Radar sistemi durduruluyor...")

        # Plugin'lere bildirim gönder
        self.plugin_manager.notify_plugins("stop")

        # Thread'leri durdur
        self.running.clear()

        # Soketleri kapat
        self.context.destroy(linger=0)

        logger.info("Radar sistemi başarıyla durduruldu")

    def _sdr_worker(self):
        """SDR veri işçi thread'i"""
        logger.info("SDR worker thread başlatıldı")

        while self.running.is_set():
            try:
                # SDR verilerini dinle (non-blocking)
                try:
                    message = self.sdr_socket.recv_json(flags=zmq.NOBLOCK)

                    # Veriyi işleme kuyruğuna ekle
                    if not self.data_queue.full():
                        self.data_queue.put(("sdr_data", message))
                    else:
                        logger.warning("Veri kuyruğu dolu, SDR verisi atlandı")

                except zmq.Again:
                    # Veri yok, devam et
                    pass

                time.sleep(0.01)

            except Exception as e:
                logger.error(f"SDR worker hatası: {str(e)}")
                time.sleep(0.1)

    def _ai_worker(self):
        """AI veri işçi thread'i"""
        logger.info("AI worker thread başlatıldı")

        while self.running.is_set():
            try:
                # AI verilerini dinle (non-blocking)
                try:
                    message = self.ai_socket.recv_json(flags=zmq.NOBLOCK)

                    # Veriyi işleme kuyruğuna ekle
                    if not self.data_queue.full():
                        self.data_queue.put(("ai_data", message))
                    else:
                        logger.warning("Veri kuyruğu dolu, AI verisi atlandı")

                except zmq.Again:
                    # Veri yok, devam et
                    pass

                time.sleep(0.01)

            except Exception as e:
                logger.error(f"AI worker hatası: {str(e)}")
                time.sleep(0.1)

    def _control_worker(self):
        """Kontrol işçi thread'i"""
        logger.info("Control worker thread başlatıldı")

        while self.running.is_set():
            try:
                # Kontrol komutlarını dinle (non-blocking)
                try:
                    message = self.control_socket.recv_json(flags=zmq.NOBLOCK)
                    command = message.get("command", "")

                    response = {"status": "error", "message": "Bilinmeyen komut"}

                    if command == "status":
                        response = {
                            "status": "ok",
                            "running": self.running.is_set(),
                            "detections_count": len(self.detections),
                            "ground_type": self.ground_properties.type.name,
                            "performance": self.performance_monitor.get_stats()
                        }
                    elif command == "set_confidence_threshold":
                        threshold = message.get("value", 0)
                        if 0 < threshold <= 1.0:
                            self.args.confidence_threshold = threshold
                            response = {"status": "ok", "message": f"Güven eşiği {threshold} olarak ayarlandı"}
                    elif command == "set_max_range":
                        max_range = message.get("value", 0)
                        if max_range > 0:
                            self.args.max_range = max_range
                            response = {"status": "ok", "message": f"Maksimum menzil {max_range}m olarak ayarlandı"}
                    elif command == "save_screenshot":
                        filename = message.get("filename", "")
                        if not filename:
                            filename = f"radar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

                        save_path = os.path.join(self.args.save_path, filename)

                        if hasattr(self, 'main_window'):
                            pixmap = self.main_window.grab()
                            pixmap.save(save_path)
                            response = {"status": "ok", "path": save_path}
                        else:
                            response = {"status": "error", "message": "UI başlatılmadı"}

                    # Yanıtı gönder
                    self.control_socket.send_json(response)

                except zmq.Again:
                    # Komut yok, devam et
                    pass

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Control worker hatası: {str(e)}")
                time.sleep(0.1)

    def _simulation_worker(self):
        """Simülasyon veri üretici thread'i"""
        logger.info("Simülasyon worker thread başlatıldı")

        # Simüle edilmiş veri üretimi
        while self.running.is_set():
            try:
                # Rastgele tespitler üret
                if np.random.random() > 0.7:  # %30 olasılıkla tespit üret
                    detection = Detection(
                        angle=np.random.uniform(0, 360),
                        distance=np.random.uniform(1, self.args.max_range),
                        confidence=np.random.uniform(0.5, 1.0),
                        type=np.random.choice(list(DetectionType)),
                        metadata={"simulated": True}
                    )

                    with self.lock:
                        self.detections.append(detection)
                        self.plugin_manager.notify_plugins("detection", detection)

                # Radar açısını güncelle
                self.radar_angle = (self.radar_angle + self.radar_speed) % 360

                # UI güncelleme sinyali gönder
                if hasattr(self, 'update_signal'):
                    self.update_signal.emit()

                time.sleep(SIMULATION_DATA_RATE)

            except Exception as e:
                logger.error(f"Simülasyon worker hatası: {str(e)}")
                time.sleep(0.1)

    def _data_processor(self):
        """Veri işleme thread'i"""
        logger.info("Data processor thread başlatıldı")

        while self.running.is_set():
            try:
                # Kuyruktan veri al
                try:
                    data_type, data = self.data_queue.get(timeout=0.1)

                    if data_type == "sdr_data":
                        self._process_sdr_data(data)
                    elif data_type == "ai_data":
                        self._process_ai_data(data)

                except Empty:
                    # Kuyruk boş, devam et
                    continue

            except Exception as e:
                logger.error(f"Data processor hatası: {str(e)}")
                time.sleep(0.1)

    def _process_sdr_data(self, data):
        """SDR verilerini işler"""
        # SDR veri işleme mantığı
        # ...
        pass

    def _process_ai_data(self, data):
        """AI verilerini işler"""
        try:
            if "detections" in data:
                with self.lock:
                    # Yeni tespitleri ekle
                    for det in data["detections"]:
                        if det["confidence"] >= self.args.confidence_threshold:
                            detection = Detection(
                                angle=det.get("angle", 0),
                                distance=det.get("distance", 0),
                                confidence=det.get("confidence", 0),
                                type=DetectionType[det.get("type", "UNKNOWN")],
                                metadata=det.get("metadata", {})
                            )
                            self.detections.append(detection)
                            self.plugin_manager.notify_plugins("detection", detection)

                    # Eski tespitleri temizle (30 saniyeden eski olanlar)
                    current_time = time.time()
                    self.detections = [
                        d for d in self.detections
                        if current_time - d.timestamp < 30
                    ]

            if "ground_type" in data:
                self.ground_properties = GroundProperties(
                    type=GroundType[data["ground_type"]],
                    density=data.get("density", 0),
                    conductivity=data.get("conductivity", 0),
                    permittivity=data.get("permittivity", 0)
                )

            # UI güncelleme sinyali gönder
            if hasattr(self, 'update_signal'):
                self.update_signal.emit()

        except Exception as e:
            logger.error(f"AI veri işleme hatası: {str(e)}")

    def _setup_radar_view(self):
        """Radar görünümünü ayarlar"""
        if not hasattr(self, 'ax'):
            return

        self.ax.clear()

        # Polar eksen ayarları
        self.ax.set_theta_zero_location('N')  # 0 derece kuzeyde
        self.ax.set_theta_direction(-1)  # Saat yönünde
        self.ax.set_rlim(0, self.args.max_range)

        # Izgara ve etiketler
        self.ax.grid(self.show_grid)

        # Başlık ve etiketler
        self.ax.set_title("2D Radar Görünümü", pad=20)
        self.ax.set_xlabel("Açı (derece)")
        self.ax.set_ylabel("Menzil (m)", labelpad=20)

        # Derinlik halkaları
        if self.show_depth_rings:
            for r in np.arange(0, self.args.max_range, self.args.max_range / 5):
                circle = Circle((0, 0), r, fill=False, linestyle='--', alpha=0.3)
                self.ax.add_patch(circle)

    def _update_plot(self, frame):
        """Radar görüntüsünü günceller"""
        self.performance_monitor.start_frame()

        try:
            # Radar açısını güncelle
            self.radar_angle = (self.radar_angle + self.radar_speed) % 360

            # Eksenleri temizle ve yeniden ayarla
            self._setup_radar_view()

            # Tespitleri çiz
            if self.detections:
                angles = []
                ranges = []
                confidences = []
                colors = []
                markers = []

                with self.lock:
                    for detection in self.detections:
                        if detection.confidence >= self.args.confidence_threshold:
                            angle_rad = np.deg2rad(detection.angle)
                            angles.append(angle_rad)
                            ranges.append(detection.distance)
                            confidences.append(detection.confidence)

                            # Nesne türüne göre stil belirle
                            if detection.type == DetectionType.METAL:
                                colors.append('red')
                                markers.append('o')  # Daire
                            elif detection.type == DetectionType.VOID:
                                colors.append('blue')
                                markers.append('s')  # Kare
                            else:  # Mineral/Kaya/Bilinmeyen
                                colors.append('green')
                                markers.append('^')  # Üçgen

                # Tespitleri çiz
                for angle, rng, color, marker in zip(angles, ranges, colors, markers):
                    self.ax.plot(angle, rng, marker=marker, color=color, markersize=10)

                    # Etiketleri göster
                    if self.show_labels:
                        self.ax.text(angle, rng, f"{rng:.1f}m", color=color, fontsize=8)

            # Tarama çizgisini çiz
            scan_line = self.ax.plot(
                [np.deg2rad(self.radar_angle), np.deg2rad(self.radar_angle)],
                [0, self.args.max_range],
                'r-',
                alpha=0.5
            )[0]

            # Tarama geçmişini güncelle
            self.scan_history.append(self.radar_angle)
            if len(self.scan_history) > MAX_HISTORY_LENGTH:
                self.scan_history.pop(0)

            # Tarama geçmişini çiz
            if self.show_history:
                for angle in self.scan_history:
                    self.ax.plot(
                        [np.deg2rad(angle), np.deg2rad(angle)],
                        [0, self.args.max_range * 0.1],
                        'y-',
                        alpha=0.1
                    )

            return [scan_line]

        finally:
            self.performance_monitor.end_frame()

    def save_screenshot(self, filename=None):
        """Ekran görüntüsünü kaydeder"""
        if not filename:
            filename = f"radar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        save_path = os.path.join(self.args.save_path, filename)

        if hasattr(self, 'main_window'):
            # PyQt5 için
            pixmap = self.main_window.grab()
            pixmap.save(save_path)
        elif hasattr(self, 'fig'):
            # Matplotlib için
            self.fig.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            raise RuntimeError("Kayıt için uygun arayüz bulunamadı")

        logger.info(f"Ekran görüntüsü kaydedildi: {save_path}")
        return save_path


class RadarMainWindow(QMainWindow):
    """Ana radar penceresi sınıfı"""

    def __init__(self, radar_ui):
        super().__init__()
        self.radar_ui = radar_ui
        self._init_ui()
        self._setup_connections()

        # Performans güncelleme timer'ı
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(1000)  # 1 saniyede bir

        logger.info("Radar ana penceresi başlatıldı")

    def _init_ui(self):
        """Kullanıcı arayüzünü başlatır"""
        self.setWindowTitle(TRANSLATIONS[self.radar_ui.language]["window_title"])
        self.setGeometry(100, 100, 1200, 800)

        # Merkez widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Ana layout
        main_layout = QHBoxLayout(central_widget)

        # Radar görüntüleme alanı
        self._setup_visualization(main_layout)

        # Kontrol paneli
        self._setup_control_panel(main_layout)

        # Menü çubuğu
        self._create_menu_bar()

        # Durum çubuğu
        self.statusBar().showMessage(TRANSLATIONS[self.radar_ui.language]["status_ready"])

        # Araç çubuğu
        self._create_toolbar()

    def _setup_visualization(self, main_layout):
        """Görselleştirme alanını ayarlar"""
        if self.radar_ui.visualization_mode == VisualizationMode._3D and PYQTGRAPH_AVAILABLE:
            self._setup_3d_visualization(main_layout)
        else:
            self._setup_2d_visualization(main_layout)

    def _setup_3d_visualization(self, main_layout):
        """3D görselleştirme alanını ayarlar"""
        try:
            # PyQtGraph GLViewWidget oluştur
            self.gl_view = gl.GLViewWidget()
            self.gl_view.setCameraPosition(distance=self.radar_ui.args.max_range * 2)

            # 3D grid ekle
            grid = gl.GLGridItem()
            grid.scale(10, 10, 1)
            self.gl_view.addItem(grid)

            # Eksenleri ekle
            axis = gl.GLAxisItem()
            self.gl_view.addItem(axis)

            # 3D scatter plot için veri yapısı
            self.scatter_plot = gl.GLScatterPlotItem()
            self.gl_view.addItem(self.scatter_plot)

            main_layout.addWidget(self.gl_view, stretch=1)

            logger.info("3D görselleştirme başlatıldı")

        except Exception as e:
            logger.error(f"3D görselleştirme hatası: {str(e)}")
            self.radar_ui.visualization_mode = VisualizationMode._2D
            self._setup_2d_visualization(main_layout)

    def _setup_2d_visualization(self, main_layout):
        """2D görselleştirme alanını ayarlar"""
        # Matplotlib figure ve canvas oluştur
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='polar')

        # Radar görünümünü ayarla
        self.radar_ui.fig = self.figure
        self.radar_ui.ax = self.ax
        self.radar_ui._setup_radar_view()

        # Navigation toolbar ekle
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Layout'a ekle
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        main_layout.addWidget(container, stretch=1)

        logger.info("2D görselleştirme başlatıldı")

    def _setup_control_panel(self, main_layout):
        """Kontrol panelini ayarlar"""
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setFixedWidth(300)

        layout = QVBoxLayout(panel)

        # Görüntüleme seçenekleri
        display_group = QGroupBox("Görüntüleme Seçenekleri")
        display_layout = QVBoxLayout()

        self.grid_check = QCheckBox("Izgara Göster")
        self.grid_check.setChecked(True)
        display_layout.addWidget(self.grid_check)

        self.history_check = QCheckBox("Tarama Geçmişi Göster")
        self.history_check.setChecked(True)
        display_layout.addWidget(self.history_check)

        self.labels_check = QCheckBox("Etiketleri Göster")
        self.labels_check.setChecked(True)
        display_layout.addWidget(self.labels_check)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # Radar kontrolü
        radar_group = QGroupBox("Radar Kontrolleri")
        radar_layout = QVBoxLayout()

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 10)
        self.speed_slider.setValue(2)
        radar_layout.addWidget(QLabel("Tarama Hızı:"))
        radar_layout.addWidget(self.speed_slider)

        self.range_slider = QSlider(Qt.Horizontal)
        self.range_slider.setRange(10, 100)
        self.range_slider.setValue(self.radar_ui.args.max_range)
        radar_layout.addWidget(QLabel("Maksimum Menzil:"))
        radar_layout.addWidget(self.range_slider)

        radar_group.setLayout(radar_layout)
        layout.addWidget(radar_group)

        # Performans bilgisi
        perf_group = QGroupBox("Performans")
        perf_layout = QVBoxLayout()

        self.fps_label = QLabel("FPS: 0")
        perf_layout.addWidget(self.fps_label)

        self.frame_time_label = QLabel("Frame Süresi: 0ms")
        perf_layout.addWidget(self.frame_time_label)

        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)

        # Boşluk ekle
        layout.addStretch()

        main_layout.addWidget(panel)

    def _create_menu_bar(self):
        """Menü çubuğunu oluşturur"""
        menubar = self.menuBar()

        # Dosya menüsü
        file_menu = menubar.addMenu(TRANSLATIONS[self.radar_ui.language]["file"])

        save_action = QAction("Ekran Görüntüsü Kaydet", self)
        save_action.triggered.connect(self.save_screenshot)
        file_menu.addAction(save_action)

        exit_action = QAction("Çıkış", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Görünüm menüsü
        view_menu = menubar.addMenu(TRANSLATIONS[self.radar_ui.language]["view"])

        mode_action = QAction("3D Moda Geç", self)
        mode_action.setCheckable(True)
        mode_action.setChecked(self.radar_ui.visualization_mode == VisualizationMode._3D)
        mode_action.triggered.connect(self.toggle_visualization_mode)
        view_menu.addAction(mode_action)

        # Yardım menüsü
        help_menu = menubar.addMenu(TRANSLATIONS[self.radar_ui.language]["help"])

        about_action = QAction("Hakkında", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _create_toolbar(self):
        """Araç çubuğunu oluşturur"""
        toolbar = self.addToolBar("Araçlar")

        start_action = QAction(QIcon.fromTheme("media-playback-start"),
                               TRANSLATIONS[self.radar_ui.language]["start"], self)
        start_action.triggered.connect(self.start_scan)
        toolbar.addAction(start_action)

        stop_action = QAction(QIcon.fromTheme("media-playback-stop"),
                              TRANSLATIONS[self.radar_ui.language]["stop"], self)
        stop_action.triggered.connect(self.stop_scan)
        toolbar.addAction(stop_action)

        toolbar.addSeparator()

        save_action = QAction(QIcon.fromTheme("document-save"), "Kaydet", self)
        save_action.triggered.connect(self.save_screenshot)
        toolbar.addAction(save_action)

    def _setup_connections(self):
        """Sinyal bağlantılarını ayarlar"""
        self.grid_check.stateChanged.connect(self.toggle_grid)
        self.history_check.stateChanged.connect(self.toggle_history)
        self.labels_check.stateChanged.connect(self.toggle_labels)
        self.speed_slider.valueChanged.connect(self.change_speed)
        self.range_slider.valueChanged.connect(self.change_range)

    def update_plot(self):
        """Radar görüntüsünü günceller"""
        if self.radar_ui.visualization_mode == VisualizationMode._3D:
            self._update_3d_plot()
        else:
            self.radar_ui._update_plot(None)
            self.canvas.draw()

    def _update_3d_plot(self):
        """3D radar görüntüsünü günceller"""
        if not PYQTGRAPH_AVAILABLE:
            return

        points = []
        colors = []
        sizes = []

        with self.radar_ui.lock:
            for detection in self.radar_ui.detections:
                if detection.confidence >= self.radar_ui.args.confidence_threshold:
                    # Polar koordinatları kartezyene çevir
                    angle_rad = np.deg2rad(detection.angle)
                    x = detection.distance * np.sin(angle_rad)
                    y = detection.distance * np.cos(angle_rad)
                    z = 0

                    points.append([x, y, z])

                    # Renk ve boyut ayarla
                    if detection.type == DetectionType.METAL:
                        colors.append([1, 0, 0, 1])  # Kırmızı
                        sizes.append(10)
                    elif detection.type == DetectionType.VOID:
                        colors.append([0, 0, 1, 1])  # Mavi
                        sizes.append(8)
                    else:  # Mineral/Kaya
                        colors.append([0, 1, 0, 1])  # Yeşil
                        sizes.append(12)

        if points:
            self.scatter_plot.setData(
                pos=np.array(points),
                color=np.array(colors),
                size=np.array(sizes)
            )

    def update_stats(self):
        """Performans istatistiklerini günceller"""
        stats = self.radar_ui.performance_monitor.get_stats()
        self.fps_label.setText(f"FPS: {stats['fps']:.1f}")
        self.frame_time_label.setText(f"Frame Süresi: {stats['avg_frame_time'] * 1000:.1f}ms")

    def toggle_grid(self, state):
        """Izgara görünürlüğünü değiştirir"""
        self.radar_ui.show_grid = (state == Qt.Checked)
        if hasattr(self.radar_ui, 'ax'):
            self.radar_ui._setup_radar_view()
            if hasattr(self, 'canvas'):
                self.canvas.draw()

    def toggle_history(self, state):
        """Tarama geçmişi görünürlüğünü değiştirir"""
        self.radar_ui.show_history = (state == Qt.Checked)

    def toggle_labels(self, state):
        """Etiket görünürlüğünü değiştirir"""
        self.radar_ui.show_labels = (state == Qt.Checked)

    def change_speed(self, value):
        """Tarama hızını değiştirir"""
        self.radar_ui.radar_speed = value

    def change_range(self, value):
        """Maksimum menzili değiştirir"""
        self.radar_ui.args.max_range = value
        if hasattr(self.radar_ui, 'ax'):
            self.radar_ui._setup_radar_view()
            if hasattr(self, 'canvas'):
                self.canvas.draw()

    def toggle_visualization_mode(self, checked):
        """Görselleştirme modunu değiştirir"""
        if checked and PYQTGRAPH_AVAILABLE:
            self.radar_ui.visualization_mode = VisualizationMode._3D
        else:
            self.radar_ui.visualization_mode = VisualizationMode._2D

        # Görselleştirmeyi yeniden başlat
        self._setup_visualization(self.centralWidget().layout())

    def start_scan(self):
        """Tarama başlat"""
        self.statusBar().showMessage(TRANSLATIONS[self.radar_ui.language]["status_scanning"])

    def stop_scan(self):
        """Tarama durdur"""
        self.statusBar().showMessage(TRANSLATIONS[self.radar_ui.language]["status_ready"])

    def save_screenshot(self):
        """Ekran görüntüsü kaydet"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Ekran Görüntüsü Kaydet",
            os.path.join(self.radar_ui.args.save_path, f"radar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"),
            "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg)"
        )

        if filename:
            try:
                self.radar_ui.save_screenshot(filename)
                QMessageBox.information(self, "Başarılı", "Ekran görüntüsü kaydedildi")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Kayıt başarısız: {str(e)}")

    def show_about(self):
        """Hakkında penceresini göster"""
        about_text = """
        <h2>Radar UI</h2>
        <p>SDR tabanlı yeraltı tespit sistemi için gelişmiş radar arayüzü</p>
        <p>Versiyon: 1.0.0</p>
        <p>Geliştirici: Your Name</p>
        <p>Lisans: MIT</p>
        """
        QMessageBox.about(self, "Hakkında", about_text)

    def closeEvent(self, event):
        """Pencere kapatılırken temizlik yapar"""
        self.radar_ui.stop()
        event.accept()


if __name__ == "__main__":
    # Komut satırı argümanlarını işle
    parser = argparse.ArgumentParser(description="Gelişmiş Radar Arayüzü")

    parser.add_argument('--fullscreen', action='store_true',
                        help='Tam ekran modunda başlat')
    parser.add_argument('--3d-mode', action='store_true',
                        help='3D görselleştirme modunu kullan')
    parser.add_argument('--simulation', action='store_true',
                        help='Simülasyon modunda çalıştır')
    parser.add_argument('--update-interval', type=int,
                        default=DEFAULT_UPDATE_INTERVAL,
                        help='Güncelleme aralığı (ms)')
    parser.add_argument('--max-range', type=float,
                        default=DEFAULT_MAX_RANGE,
                        help='Maksimum tespit menzili (metre)')
    parser.add_argument('--confidence-threshold', type=float,
                        default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help='Güven eşik değeri (0.0-1.0)')
    parser.add_argument('--language', type=str, default='en',
                        choices=['en', 'tr'],
                        help='Arayüz dili')
    parser.add_argument('--dark-mode', action='store_true',
                        help='Karanlık tema kullan')
    parser.add_argument('--save-path', type=str, default='screenshots',
                        help='Ekran görüntülerinin kaydedileceği dizin')
    parser.add_argument('--sdr-port', type=int, default=5555,
                        help='SDR veri portu')
    parser.add_argument('--ai-port', type=int, default=5556,
                        help='AI sınıflandırma portu')
    parser.add_argument('--control-port', type=int, default=5557,
                        help='Kontrol portu')

    args = parser.parse_args()

    try:
        # Radar UI örneğini oluştur ve başlat
        radar_ui = RadarUI(args)
        if radar_ui.start():
            logger.info("Radar UI başarıyla başlatıldı")
        else:
            logger.error("Radar UI başlatılamadı")
            sys.exit(1)

    except Exception as e:
        logger.critical(f"Kritik hata: {str(e)}")
        sys.exit(1)