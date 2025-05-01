#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
radar_ui.py - Enhanced Radar UI Module

Version: 2.0
Author: AI Assistant (Manus)
Last Updated: 2025-04-30

SDR tabanlı yeraltı tespit sistemi için ana kullanıcı arayüzü.
PyQt5 tabanlıdır ve 3D görselleştirmeyi (PyVista), 2D haritaları,
kontrol düğmelerini ve durum göstergelerini entegre eder.

Özellikler:
- İki panelli düzen (3D Görünüm + 2D Harita/Durum).
- PyVista ile entegre 3D yeraltı görünümü.
- 2D sinyal yoğunluğu/derinlik haritası (placeholder).
- Gerçek zamanlı hedef listesi ve güven skorları.
- Sistem durumu, zemin tipi, gürültü seviyesi göstergeleri.
- Kalibrasyon, frekans seçimi, tarama kontrol düğmeleri.
- ZMQ üzerinden diğer modüllerle iletişim.
"""

import os
import sys
import time
import argparse
import numpy as np
import zmq
import json
import logging
import logging.handlers
from enum import Enum
from typing import List, Dict, Optional, Any
from threading import Thread, Event, Lock
from queue import Queue, Empty
import signal
from config import (
    GroundType, COLOR_MAP_GROUND, COLOR_MAP_DETECTION, COLOR_MAP_TARGET,
    PORT_PREPROCESSING_OUTPUT, PORT_CALIBRATION_RESULT, PORT_AI_OUTPUT,
    PORT_SDR_CONTROL, PORT_CALIBRATION_CONTROL, PORT_PREPROCESSING_CONTROL,
    PORT_AI_CONTROL, PORT_VISUALIZATION_CONTROL
    # SCAN_FREQUENCY_RANGE_MHZ varsa config.py'de tanımlı olmalı, yoksa kaldır
)
# 3D Görünüm modülü
from depth_3d_view import Depth3DView # 3D görünüm mantığı için

# PyQt5 importları
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QComboBox, QSlider, QCheckBox, QGroupBox,
        QGridLayout, QTabWidget, QSplitter, QFrame, QFileDialog,
        QMessageBox, QDockWidget, QToolBar, QAction, QStatusBar,
        QSizePolicy, QListWidget, QListWidgetItem, QTextEdit, QProgressBar
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QSize
    from PyQt5.QtGui import QIcon, QFont, QColor, QPalette, QPixmap
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt5 kütüphanesi bulunamadı. Arayüz çalıştırılamaz.")
    sys.exit(1)

# PyVista/PyVistaQT importları
try:
    import pyvista as pv
    from pyvistaqt import BackgroundPlotter
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("PyVista/PyVistaQT bulunamadı. 3D görünüm kullanılamaz.")

# Matplotlib (2D harita için)
try:
    import matplotlib
    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib bulunamadı. 2D harita kullanılamaz.")

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s',
    handlers=[
        logging.FileHandler("radar_ui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("radar_ui")

# --- Yardımcı Sınıflar ve Fonksiyonlar --- 

class ZMQListener(QObject):
    """ZMQ mesajlarını dinleyen ve sinyal yayan QObject."""
    message_received = pyqtSignal(dict)
    
    def __init__(self, port: int, topic: str = "", conflate: bool = True):
        super().__init__()
        self.port = port
        self.topic = topic
        self.conflate = conflate
        self.running = False
        self.context = zmq.Context.instance() # Global context kullan
        self.socket = None
        self.thread = QThread(self)
        self.moveToThread(self.thread)
        self.thread.started.connect(self.run)

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.quit()
        self.thread.wait()

    def run(self):
        self.socket = self.context.socket(zmq.SUB)
        try:
            self.socket.connect(f"tcp://localhost:{self.port}")
            self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
            if self.conflate:
                self.socket.setsockopt(zmq.CONFLATE, 1)
            logger.info(f"ZMQ Listener başlatıldı: Port={self.port}, Topic=\'{self.topic}\'")
            
            poller = zmq.Poller()
            poller.register(self.socket, zmq.POLLIN)
            
            while self.running:
                # 100ms timeout ile bekle
                socks = dict(poller.poll(100))
                if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                    try:
                        message = self.socket.recv_json()
                        self.message_received.emit(message)
                    except json.JSONDecodeError:
                        logger.error(f"Port {self.port}: Geçersiz JSON alındı.")
                    except zmq.ZMQError as e:
                        if e.errno == zmq.ETERM:
                            logger.info(f"Port {self.port}: ZMQ context sonlandırıldı.")
                            break
                        else:
                            logger.error(f"Port {self.port}: ZMQ hatası: {e}")
                            time.sleep(1) # Hata durumunda bekle
                    except Exception as e:
                         logger.error(f"Port {self.port}: Listener hatası: {e}", exc_info=True)
        finally:
            if self.socket:
                self.socket.close()
            logger.info(f"ZMQ Listener durdu: Port={self.port}")

class ZMQCommunicator:
    """Diğer modüllere ZMQ komutları gönderir."""
    def __init__(self):
        self.context = zmq.Context.instance()
        self.sockets: Dict[str, zmq.Socket] = {}
        self.ports = {
            "sdr": PORT_SDR_CONTROL,
            "calibration": PORT_CALIBRATION_CONTROL,
            "preprocessing": PORT_PREPROCESSING_CONTROL,
            "ai": PORT_AI_CONTROL,
            "visualization": PORT_VISUALIZATION_CONTROL
        }
        self.lock = Lock()

    def _get_socket(self, module_name: str) -> Optional[zmq.Socket]:
        """İlgili modül için REQ soketi oluşturur veya alır."""
        with self.lock:
            if module_name not in self.sockets:
                port = self.ports.get(module_name)
                if port:
                    try:
                        socket = self.context.socket(zmq.REQ)
                        socket.connect(f"tcp://localhost:{port}")
                        # Timeout ayarları
                        socket.setsockopt(zmq.LINGER, 0) # Kapatırken bekleme
                        socket.setsockopt(zmq.RCVTIMEO, 2000) # 2 saniye cevap bekleme
                        socket.setsockopt(zmq.SNDTIMEO, 1000) # 1 saniye gönderme bekleme
                        self.sockets[module_name] = socket
                        logger.info(f"{module_name} modülü için ZMQ REQ soketi oluşturuldu (Port: {port})")
                    except Exception as e:
                        logger.error(f"{module_name} soketi oluşturulamadı: {e}")
                        return None
                else:
                    logger.error(f"Bilinmeyen modül adı: {module_name}")
                    return None
            return self.sockets[module_name]

    def send_command(self, module_name: str, command: dict) -> Optional[dict]:
        """Belirtilen modüle komut gönderir ve cevabı bekler."""
        socket = self._get_socket(module_name)
        if not socket:
            return None
            
        try:
            logger.debug(f"Komut gönderiliyor -> {module_name}: {command}")
            socket.send_json(command)
            response = socket.recv_json()
            logger.debug(f"Cevap alındı <- {module_name}: {response}")
            return response
        except zmq.Again:
            logger.error(f"{module_name} modülünden cevap zaman aşımına uğradı ({command}).")
            # Soketi yeniden bağlamayı dene?
            self._reset_socket(module_name)
            return {"status": "error", "message": "Timeout"}
        except Exception as e:
            logger.error(f"{module_name} modülüne komut gönderirken hata ({command}): {e}")
            self._reset_socket(module_name)
            return {"status": "error", "message": str(e)}
            
    def _reset_socket(self, module_name: str):
         """Sorunlu soketi kapatıp yeniden oluşturur."""
         with self.lock:
             if module_name in self.sockets:
                 try:
                     self.sockets[module_name].close()
                 except: pass
                 del self.sockets[module_name]
                 logger.warning(f"{module_name} modülü için ZMQ soketi sıfırlandı.")

    def close_all(self):
        """Tüm soketleri kapatır."""
        with self.lock:
            for name, socket in self.sockets.items():
                try:
                    socket.close()
                except: pass
            self.sockets.clear()
            logger.info("Tüm ZMQ REQ soketleri kapatıldı.")

# --- Matplotlib 2D Harita Widget --- 

class MplCanvas(FigureCanvas):
    """Matplotlib figürü için Qt widget\ı."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

class SignalMapWidget(QWidget):
    """2D Sinyal Yoğunluk Haritasını gösteren widget."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.data = None # Gösterilecek veri (örn. numpy array)
        self.cmap = plt.cm.viridis # Renk haritası
        self.image = None # Matplotlib image nesnesi
        self._setup_plot()

    def _setup_plot(self):
        self.canvas.axes.set_title("2D Sinyal Yoğunluk Haritası")
        self.canvas.axes.set_xlabel("Yatay Konum (m)")
        self.canvas.axes.set_ylabel("Derinlik (m)")
        # Eksen limitleri dinamik olarak ayarlanabilir
        self.canvas.axes.invert_yaxis() # Derinlik aşağı doğru artsın
        self.canvas.fig.tight_layout()

    def update_map(self, data: np.ndarray, extent: List[float]):
        """
        Haritayı yeni verilerle günceller.
        
        Args:
            data (np.ndarray): 2D numpy array (yoğunluk değerleri).
            extent (List[float]): [xmin, xmax, ymin, ymax] harita sınırları.
        """
        if data is None or data.ndim != 2:
            logger.warning("Geçersiz 2D harita verisi.")
            return
            
        self.data = data
        if self.image is None:
            self.image = self.canvas.axes.imshow(self.data, aspect='auto', cmap=self.cmap, extent=extent)
            self.canvas.fig.colorbar(self.image, ax=self.canvas.axes)
        else:
            self.image.set_data(self.data)
            self.image.set_extent(extent)
            # Veri sınırlarına göre renk limitlerini güncelle
            self.image.set_clim(vmin=np.min(data), vmax=np.max(data))
            
        # Eksen limitlerini güncelle
        self.canvas.axes.set_xlim(extent[0], extent[1])
        self.canvas.axes.set_ylim(extent[3], extent[2]) # imshow y eksenini ters çevirir
        
        self.canvas.draw_idle()

# --- Ana Arayüz Penceresi --- 

class RadarMainWindow(QMainWindow):
    """Ana radar arayüz penceresi."""
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.zmq_comm = ZMQCommunicator()
        self.listeners = {}
        self.latest_preprocessing_data = None
        self.latest_calibration_data = None
        self.latest_ai_data = None
        self.data_lock = Lock()
        
        # 3D Görünüm için Depth3DView örneği (plotter içerir)
        self.depth_view = None
        if PYVISTA_AVAILABLE:
             # Depth3DView\a UI config\ini ver
             vis_config = config.get("module_configs", {}).get("visualization", {})
             vis_config["backend"] = "pyvista" # Emin ol
             self.depth_view = Depth3DView(vis_config)
             # PyVista plotter\ını al (BackgroundPlotter olmalı)
             # Depth3DView.start() içinde plotter oluşturulacak
        
        self._init_ui()
        self._setup_listeners()
        self._connect_signals()
        
        # Periyodik UI güncelleme timer\ı
        self.ui_update_timer = QTimer(self)
        self.ui_update_timer.timeout.connect(self.update_ui_elements)
        self.ui_update_timer.start(config.get("ui_update_interval_ms", 500)) # 500ms\de bir

    def _init_ui(self):
        self.setWindowTitle("Mesopotamia GPR - Radar Arayüzü")
        self.setGeometry(100, 100, 1400, 900)
        self.setWindowIcon(QIcon("icon.png")) # İkon dosyası eklenmeli

        # Ana Widget ve Layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Splitter ile iki panel oluştur
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- Sol Panel: 3D Görünüm --- 
        left_panel = QFrame(self)
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        if self.depth_view and PYVISTA_AVAILABLE:
            # Depth3DView.start() içinde plotter oluşturulur ve widget\a eklenir
            # Şimdilik bir placeholder ekleyelim
            self.plotter_widget = QWidget() # Gerçek plotter buraya gelecek
            left_layout.addWidget(self.plotter_widget)
            logger.info("PyVista 3D görünüm alanı oluşturuldu.")
        else:
            left_layout.addWidget(QLabel("3D Görünüm kullanılamıyor (PyVista/PyVistaQT eksik)"))
            
        splitter.addWidget(left_panel)

        # --- Sağ Panel: Kontroller, 2D Harita, Durum --- 
        right_panel = QFrame(self)
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_panel)
        splitter.addWidget(right_panel)

        # Sağ paneli de ikiye böl (üst: kontroller/durum, alt: 2D harita/hedefler)
        right_splitter = QSplitter(Qt.Vertical)
        right_layout.addWidget(right_splitter)

        # Sağ Üst Panel: Kontroller ve Durum
        top_right_panel = QWidget()
        top_right_layout = QVBoxLayout(top_right_panel)
        right_splitter.addWidget(top_right_panel)

        # Kontrol Grubu
        control_group = QGroupBox("Kontroller")
        control_layout = QGridLayout(control_group)
        self.btn_start_scan = QPushButton("Taramayı Başlat")
        self.btn_stop_scan = QPushButton("Taramayı Durdur")
        self.btn_start_calibration = QPushButton("Kalibrasyon Başlat")
        self.combo_frequency = QComboBox()
        # Frekansları config\den veya SDR modülünden al?
        self.combo_frequency.addItems(["500 MHz", "1 GHz", "1.5 GHz", "2 GHz", "Tarama Modu"])
        control_layout.addWidget(self.btn_start_scan, 0, 0)
        control_layout.addWidget(self.btn_stop_scan, 0, 1)
        control_layout.addWidget(self.btn_start_calibration, 1, 0)
        control_layout.addWidget(QLabel("Frekans:"), 1, 1)
        control_layout.addWidget(self.combo_frequency, 1, 2)
        top_right_layout.addWidget(control_group)

        # Durum Grubu
        status_group = QGroupBox("Sistem Durumu")
        status_layout = QGridLayout(status_group)
        self.lbl_system_status = QLabel("Beklemede")
        self.lbl_ground_type = QLabel("Zemin Tipi: Bilinmiyor")
        self.lbl_noise_level = QLabel("Gürültü: - dB")
        self.lbl_target_count = QLabel("Hedef Sayısı: 0")
        self.lbl_mobile_sync = QLabel("Mobil Senk: Bağlı Değil") # Placeholder
        self.progress_calibration = QProgressBar()
        self.progress_calibration.setRange(0, 100)
        self.progress_calibration.setValue(0)
        self.progress_calibration.setTextVisible(True)
        self.progress_calibration.setFormat("Kalibrasyon: %p%")
        status_layout.addWidget(QLabel("Durum:"), 0, 0)
        status_layout.addWidget(self.lbl_system_status, 0, 1)
        status_layout.addWidget(QLabel("Zemin:"), 1, 0)
        status_layout.addWidget(self.lbl_ground_type, 1, 1)
        status_layout.addWidget(QLabel("Gürültü:"), 2, 0)
        status_layout.addWidget(self.lbl_noise_level, 2, 1)
        status_layout.addWidget(QLabel("Hedefler:"), 3, 0)
        status_layout.addWidget(self.lbl_target_count, 3, 1)
        status_layout.addWidget(QLabel("Mobil:"), 4, 0)
        status_layout.addWidget(self.lbl_mobile_sync, 4, 1)
        status_layout.addWidget(self.progress_calibration, 5, 0, 1, 2)
        top_right_layout.addWidget(status_group)
        top_right_layout.addStretch(1)

        # Sağ Alt Panel: 2D Harita ve Hedef Listesi
        bottom_right_panel = QWidget()
        bottom_right_layout = QVBoxLayout(bottom_right_panel)
        right_splitter.addWidget(bottom_right_panel)

        # Tab Widget
        tab_widget = QTabWidget()
        bottom_right_layout.addWidget(tab_widget)

        # Tab 1: 2D Harita
        self.map_widget = QWidget()
        map_layout = QVBoxLayout(self.map_widget)
        if MATPLOTLIB_AVAILABLE:
            self.signal_map = SignalMapWidget(self.map_widget)
            map_layout.addWidget(self.signal_map)
        else:
            map_layout.addWidget(QLabel("2D Harita kullanılamıyor (Matplotlib eksik)"))
        tab_widget.addTab(self.map_widget, "2D Sinyal Haritası")

        # Tab 2: Hedef Listesi
        self.target_list_widget = QListWidget()
        tab_widget.addTab(self.target_list_widget, "Tespit Edilen Hedefler")
        
        # Tab 3: Log Mesajları
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        tab_widget.addTab(self.log_text_edit, "Log Mesajları")
        # TODO: Log mesajlarını buraya yönlendir

        # Splitter boyutlarını ayarla
        splitter.setSizes([800, 600])
        right_splitter.setSizes([300, 600])

        # Status Bar
        self.statusBar().showMessage("Arayüz başlatıldı.")

    def _setup_listeners(self):
        """ZMQ dinleyicilerini başlatır."""
        self.listeners["preprocessing"] = ZMQListener(PORT_PREPROCESSING_OUTPUT)
        self.listeners["preprocessing"].message_received.connect(self._handle_preprocessing_message)
        self.listeners["preprocessing"].start()
        
        self.listeners["calibration"] = ZMQListener(PORT_CALIBRATION_RESULT)
        self.listeners["calibration"].message_received.connect(self._handle_calibration_message)
        self.listeners["calibration"].start()
        
        # AI Listener (varsa)
        if PORT_AI_OUTPUT:
             self.listeners["ai"] = ZMQListener(PORT_AI_OUTPUT)
             self.listeners["ai"].message_received.connect(self._handle_ai_message)
             self.listeners["ai"].start()

    def _connect_signals(self):
        """Buton sinyallerini ilgili slotlara bağlar."""
        self.btn_start_scan.clicked.connect(self.start_scan)
        self.btn_stop_scan.clicked.connect(self.stop_scan)
        self.btn_start_calibration.clicked.connect(self.start_calibration)
        self.combo_frequency.currentIndexChanged.connect(self.set_frequency)

    # --- Slotlar ve Komut Gönderme --- 

    def start_scan(self):
        logger.info("Tarama başlatma komutu gönderiliyor...")
        # SDR modülüne taramayı başlatma komutu gönder
        response = self.zmq_comm.send_command("sdr", {"command": "start_scan"}) # Komut adı varsayımsal
        if response and response.get("status") == "ok":
            self.statusBar().showMessage("Tarama başlatıldı.")
            self.lbl_system_status.setText("Tarama Yapılıyor...")
        else:
            self.statusBar().showMessage("Tarama başlatılamadı!")
            QMessageBox.warning(self, "Hata", f"Tarama başlatılamadı: {response.get('message') if response else 'İletişim hatası'}")

    def stop_scan(self):
        logger.info("Tarama durdurma komutu gönderiliyor...")
        # SDR modülüne durdurma komutu gönder
        response = self.zmq_comm.send_command("sdr", {"command": "stop_scan"}) # Komut adı varsayımsal
        if response and response.get("status") == "ok":
            self.statusBar().showMessage("Tarama durduruldu.")
            self.lbl_system_status.setText("Beklemede")
        else:
            self.statusBar().showMessage("Tarama durdurulamadı!")
            QMessageBox.warning(self, "Hata", f"Tarama durdurulamadı: {response.get('message') if response else 'İletişim hatası'}")

def start_calibration(self):
    logger.info("Kalibrasyon başlatma komutu gönderiliyor...")
    # Kalibrasyon modülüne başlatma komutu gönder
    response = self.zmq_comm.send_command("calibration", {"command": "start_calibration"})
    if response and response.get("status") == "ok":
        self.statusBar().showMessage("Kalibrasyon başlatıldı.")
        self.lbl_system_status.setText("Kalibrasyon Yapılıyor...")
        self.progress_calibration.setValue(0)
    else:
        self.statusBar().showMessage("Kalibrasyon başlatılamadı!")
        QMessageBox.warning(self, "Hata", f"Kalibrasyon başlatılamadı: {response.get('message') if response else 'İletişim hatası'}")


    def set_frequency(self, index):
        freq_text = self.combo_frequency.itemText(index)
        logger.info(f"Frekans değiştirme komutu gönderiliyor: {freq_text}")
        command = {"command": "set_frequency", "value": 0} # Varsayılan
        scan_mode_command = {"command": "set_scan_mode", "value": "single_frequency"}
        
        if "MHz" in freq_text:
            try:
                freq_mhz = float(freq_text.split()[0])
                command["value"] = freq_mhz * 1e6
            except ValueError:
                 logger.error(f"Geçersiz frekans formatı: {freq_text}")
                 return
        elif "GHz" in freq_text:
             try:
                freq_ghz = float(freq_text.split()[0])
                command["value"] = freq_ghz * 1e9
             except ValueError:
                 logger.error(f"Geçersiz frekans formatı: {freq_text}")
                 return
        elif "Tarama Modu" in freq_text:
             scan_mode_command["value"] = "multi_frequency"
             # Frekans komutu yerine tarama modu komutu gönder
             response_scan = self.zmq_comm.send_command("sdr", scan_mode_command)
             if response_scan and response_scan.get("status") == "ok":
                 self.statusBar().showMessage("Tarama moduna geçildi.")
             else:
                 self.statusBar().showMessage("Tarama moduna geçilemedi!")
                 QMessageBox.warning(self, "Hata", f"Tarama moduna geçilemedi: {response_scan.get('message') if response_scan else 'İletişim hatası'}")
             return # Frekans ayarlama komutu gönderme
        else:
             logger.error(f"Bilinmeyen frekans seçeneği: {freq_text}")
             return

        # Önce tarama modunu tek frekansa ayarla
        response_scan = self.zmq_comm.send_command("sdr", scan_mode_command)
        if not response_scan or response_scan.get("status") != "ok":
             logger.warning("Tek frekans moduna geçilemedi, yine de frekans ayarlanmaya çalışılıyor.")
             
        # Frekans ayarlama komutunu gönder
        response_freq = self.zmq_comm.send_command("sdr", command)
        if response_freq and response_freq.get("status") == "ok":
            self.statusBar().showMessage(f"Frekans {freq_text} olarak ayarlandı.")
        else:
            self.statusBar().showMessage("Frekans ayarlanamadı!")
            QMessageBox.warning(self, "Hata", f"Frekans ayarlanamadı: {response_freq.get('message') if response_freq else 'İletişim hatası'}")

        # --- ZMQ Mesaj İsleyicileri
        def _handle_preprocessing_message(self, message: dict):
            # logger.debug("Preprocessing mesajı işleniyor...")
            logger.debug("Preprocessing mesajı işleniyor...")

            # Data lock ile veri güncellemesi yapılıyor
            with self.data_lock:
                self.latest_preprocessing_data = message

                # 3D görünümü güncellemek için Depth3DView'a veriyi ilet
                if self.depth_view:
                    # Eğer veriyi işleyip bir işleme eklemek istiyorsanız:
                    if hasattr(self.depth_view, 'process_data'):
                        self.depth_view.process_data(self.latest_preprocessing_data)
                    else:
                        print("Depth3DView nesnesi gerekli 'process_data' metoduna sahip değil.")


def _handle_calibration_message(self, message: dict):
    # logger.debug("Kalibrasyon mesajı işleniyor...")
    with self.data_lock:
        self.latest_calibration_data = message

def _handle_ai_message(self, message: dict):
    # logger.debug("AI mesajı işleniyor...")
    with self.data_lock:
        self.latest_ai_data = message
        # AI sonuçlarını işle ve hedefleri güncelle
        # Preprocessing zaten hedefleri içeriyor olabilir
        pass

# --- UI Güncelleme ---

    def update_ui_elements(self):
        """Arayüzdeki durum göstergelerini ve listeleri günceller."""
        with self.data_lock:
            cal_data = self.latest_calibration_data
            pre_data = self.latest_preprocessing_data
            ai_data = self.latest_ai_data # Henüz kullanılmıyor

        if cal_data:
            ground_type_str = cal_data.get("ground_type", "UNKNOWN")
            try:
                 ground_type_enum = GroundType[ground_type_str]
                 self.lbl_ground_type.setText(f"Zemin Tipi: {ground_type_enum.name}")
            except KeyError:
                 self.lbl_ground_type.setText(f"Zemin Tipi: {ground_type_str} (Bilinmiyor)")
                 
            noise_db = cal_data.get("background_noise_db", -100.0)
            self.lbl_noise_level.setText(f"Gürültü: {noise_db:.1f} dB")
            
            cal_progress = cal_data.get("calibration_progress", 0.0)
            self.progress_calibration.setValue(int(cal_progress * 100))
            if cal_data.get("calibration_complete", False):
                 if self.lbl_system_status.text() == "Kalibrasyon Yapılıyor...":
                      self.lbl_system_status.setText("Beklemede")
            elif self.lbl_system_status.text() != "Tarama Yapılıyor...":
                 self.lbl_system_status.setText("Kalibrasyon Yapılıyor...")
                 
        if pre_data:
            targets = pre_data.get("targets", {})
            target_count = len(targets)
            self.lbl_target_count.setText(f"Hedef Sayısı: {target_count}")
            
            # Hedef listesini güncelle
            self.target_list_widget.clear()
            for index, props in targets.items():
                depth = props.get("depth_m", 0)
                amp = props.get("amplitude", 0)
                type_guess = props.get("type_guess", "unknown")
                item_text = f"Hedef @ {depth:.2f}m | Tip: {type_guess} | Genlik: {amp:.3f}"
                self.target_list_widget.addItem(QListWidgetItem(item_text))

            import numpy as np
            from scipy import signal  # signal modülünü buraya ekliyoruz

            # 2D Haritayı güncelle (placeholder veri)
            if self.signal_map:
                # Preprocessing verisinden 2D harita oluşturulmalı
                # Örnek: Filtrelenmiş verinin zarfını alıp derinliğe göre haritala
                samples_iq = pre_data.get("filtered_samples_iq", [])
                if samples_iq:
                    samples = np.array([complex(s[0], s[1]) for s in samples_iq])
                    envelope = np.abs(signal.hilbert(samples.real))  # Basit zarf
                    # Bunu 2D bir görüntüye dönüştür (zaman/derinlik vs yatay konum)
                    # Şimdilik sadece tek bir dikey çizgi gösterelim
                    depth_axis = np.linspace(0, self.config.get("max_depth_m", 5.0), len(envelope))
                    map_data = envelope.reshape(-1, 1)
                    extent = [-0.5, 0.5, self.config.get("max_depth_m", 5.0), 0]  # xmin, xmax, ymin, ymax
                    self.signal_map.update_map(map_data, extent)

            # 3D Görünümü güncelle (Depth3DView kendi timer\ı ile güncellenmeli)
            # Veya buradan tetikle
            if self.depth_view:
                 # Depth3DView\a yeni hedefleri ver
                 self.depth_view.latest_targets = list(targets.values()) # Veya daha yapılandırılmış veri
                 # Güncellemeyi tetikle (eğer Depth3DView kendi kendine yapmıyorsa)
                 # self.depth_view._update_pyvista_view() # Doğrudan çağırmak yerine sinyal kullanılabilir
                 pass

    # --- Pencere Kapatma ve Temizlik --- 

    def closeEvent(self, event):
        """Pencere kapatıldığında temizlik yapar."""
        logger.info("Arayüz kapatılıyor...")
        self.stop_listeners_and_comm()
        if self.depth_view:
             self.depth_view.stop() # 3D görünümü de durdur
        event.accept()

    def stop_listeners_and_comm(self):
        """Tüm ZMQ dinleyicilerini ve iletişimcisini durdurur."""
        for name, listener in self.listeners.items():
            listener.stop()
        self.zmq_comm.close_all()
        logger.info("Tüm ZMQ bağlantıları kapatıldı.")

# --- Ana Uygulama Başlatma --- 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Radar UI Module")
    parser.add_argument("--config", type=str, default="system_config.json", help="Path to JSON configuration file")
    # Diğer UI özel argümanlar eklenebilir (örn. --fullscreen)
    parser.add_argument("--fullscreen", action="store_true", help="Run in fullscreen mode")
    parser.add_argument("--dark-mode", action="store_true", default=True, help="Use dark theme")

    args = parser.parse_args()

    # Load config from file
    config = {}
    try:
        with open(args.config, "r") as f:
            config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
            # UI özel ayarları config\e ekle
            ui_config = config.setdefault("module_configs", {}).setdefault("radar_ui", {})
            ui_config["fullscreen"] = args.fullscreen
            ui_config["dark_mode"] = args.dark_mode
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}. Using defaults.")
        config = {"module_configs": {"radar_ui": {"fullscreen": args.fullscreen, "dark_mode": args.dark_mode}}}
    except json.JSONDecodeError:
         logger.error(f"Error decoding JSON from {args.config}. Using defaults.")
         config = {"module_configs": {"radar_ui": {"fullscreen": args.fullscreen, "dark_mode": args.dark_mode}}}

    if not PYQT_AVAILABLE:
        logger.critical("PyQt5 bulunamadığı için arayüz başlatılamıyor.")
        sys.exit(1)

    app = QApplication(sys.argv)
    
    # Karanlık tema
    if config.get("module_configs", {}).get("radar_ui", {}).get("dark_mode", True):
        # Tema uygulama kodu (opsiyonel, stillerle daha iyi yapılabilir)
        pass 

    main_window = RadarMainWindow(config)

    # 3D Görünümü başlat (eğer varsa ve PyQt kullanıyorsa)
    if main_window.depth_view and PYVISTA_AVAILABLE and PYQT_AVAILABLE:
        # BackgroundPlotter'ı alıp UI'a ekle
        main_window.depth_view._start_pyvista_interface_headless()  # Plotter'ı oluştur ama gösterme
        if main_window.depth_view.plotter:
            plotter_container = main_window.findChild(QWidget, "plotter_widget_container")  # İsme göre bul
            if plotter_container:
                layout = QVBoxLayout(plotter_container)
                layout.addWidget(main_window.depth_view.plotter)
                logger.info("PyVista plotter arayüze eklendi.")
            else:
                logger.error("Plotter container widget bulunamadı!")
        else:
            logger.error("Depth3DView plotter oluşturulamadı!")

    if config.get("module_configs", {}).get("radar_ui", {}).get("fullscreen", False):
        main_window.showFullScreen()
    else:
        main_window.show()

    # Sinyal handler (Ctrl+C)
    signal.signal(signal.SIGINT, lambda sig, frame: app.quit())
    # Timer ile Python interpreter\ını meşgul et (sinyal yakalamak için)
    timer = QTimer()
    timer.start(500) 
    timer.timeout.connect(lambda: None) 

    exit_code = app.exec_()
    main_window.stop_listeners_and_comm() # Temizlik
    sys.exit(exit_code)

