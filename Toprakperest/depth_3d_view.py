#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
depth_3d_view.py

Bu modül, SDR tabanlı yeraltı tespit sistemi için 3D görselleştirme arayüzünü gerçekleştirir.
Şüpheli obje varsa aktifleşir ve 3D katmanlı yeraltı modeli oluşturur.

Kullanım:
    python3 depth_3d_view.py [options]

Örnek:
    python3 depth_3d_view.py --backend pyvista --max-depth 5.0 --resolution 100
"""

import os
import sys
import time
import argparse
import numpy as np
import zmq
import json
import logging
from datetime import datetime
from threading import Thread, Event
from queue import Queue
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

# Plotly kütüphanesi için koşullu import
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly kütüphanesi bulunamadı. PyVista kullanılacak.")

# PyVista kütüphanesi için koşullu import
try:
    import pyvista as pv
    import vtk

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("PyVista kütüphanesi bulunamadı. Basit 3D görselleştirme kullanılacak.")

# PyQt5 kütüphaneleri için koşullu import
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QLabel, QPushButton, QComboBox,
                                 QSlider, QCheckBox, QGroupBox, QGridLayout,
                                 QTabWidget, QSplitter, QFrame, QFileDialog,
                                 QMessageBox, QDockWidget, QToolBar, QAction,
                                 QStatusBar, QSizePolicy)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize, QThread
    from PyQt5.QtGui import QIcon, QFont, QColor, QPalette, QPixmap

    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt5 kütüphanesi bulunamadı. Basit arayüz kullanılacak.")

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("depth_3d_view.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("depth_3d_view")


class Depth3DView:
    """3D yeraltı görselleştirmesi için ana sınıf."""

    # Nesne renk tanımları (RGB 0-255)
    OBJECT_COLORS = {
        "metal": {
            "default": (255, 165, 0),  # Turuncu
            "small": (255, 200, 0),
            "large": (255, 120, 0)
        },
        "plastic": {
            "default": (0, 255, 0),  # Yeşil
            "small": (100, 255, 100),
            "large": (0, 200, 0)
        },
        "organic": {
            "default": (139, 69, 19),  # Kahverengi
            "small": (160, 82, 45),
            "large": (101, 67, 33)
        },
        "unknown": {
            "default": (255, 0, 0)  # Kırmızı
        }
    }

    # Zemin katman renk tanımları (RGB 0-255)
    GROUND_COLORS = {
        "soil": (194, 178, 128),
        "sand": (194, 178, 128),
        "clay": (182, 150, 108),
        "rock": (136, 140, 141),
        "mixed": (158, 134, 100),
        "default": (160, 160, 160)
    }

    def __init__(self, args):
        self.args = args
        self.backend = args.backend
        self.fullscreen = args.fullscreen
        self.dark_mode = args.dark_mode
        self.update_interval = args.update_interval
        self.max_depth = args.max_depth
        self.resolution = args.resolution
        self.grid_size = (args.resolution, args.resolution, args.resolution)
        self.confidence_threshold = args.confidence_threshold
        self.save_path = args.save_path

        # Veri yapıları için başlangıç değerleri
        self.detections = []
        self.ground_type = "unknown"

        # Backend kontrolü
        if self.backend == "plotly" and not PLOTLY_AVAILABLE:
            logger.warning("Plotly kütüphanesi bulunamadı, PyVista kullanılacak")
            self.backend = "pyvista"

        if self.backend == "pyvista" and not PYVISTA_AVAILABLE:
            logger.warning("PyVista kütüphanesi bulunamadı, basit 3D görselleştirme kullanılacak")
            self.backend = "simple"

        # Veri akışı kontrolü
        self.running = Event()
        self.data_queue = Queue(maxsize=100)

        # ZMQ iletişim kanalları
        self.context = zmq.Context()

        # AI veri soketi
        self.ai_socket = self.context.socket(zmq.SUB)
        self.ai_socket.connect(f"tcp://localhost:{args.ai_port}")
        self.ai_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.ai_socket.setsockopt(zmq.CONFLATE, 1)  # Sadece en son mesajı al

        # Kalibrasyon soketi
        self.calibration_socket = self.context.socket(zmq.SUB)
        self.calibration_socket.connect(f"tcp://localhost:{args.calibration_port}")
        self.calibration_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.calibration_socket.setsockopt(zmq.CONFLATE, 1)

        # Kontrol soketi
        self.control_socket = self.context.socket(zmq.REP)
        self.control_socket.bind(f"tcp://*:{args.control_port}")

        # İş parçacıkları
        self.ai_thread = None
        self.calibration_thread = None
        self.control_thread = None

        # Veri yapılarını başlat
        self._initialize_data_structures()

    def _initialize_data_structures(self):
        """Veri yapılarını başlatır."""
        self.voxel_data = np.zeros(self.grid_size, dtype=np.float32)
        self.x_coords = np.linspace(-self.max_depth / 2, self.max_depth / 2, self.grid_size[0])
        self.y_coords = np.linspace(-self.max_depth / 2, self.max_depth / 2, self.grid_size[1])
        self.z_coords = np.linspace(0, self.max_depth, self.grid_size[2])
        self.ground_layers = []
        self.detected_objects = []

        # Varsayılan zemin katmanını oluştur
        self._update_ground_layers()

    def start(self):
        """3D görselleştirmeyi başlatır."""
        if not self.running.is_set():
            self.running.set()

            # İş parçacıklarını başlat
            self.ai_thread = Thread(target=self._ai_worker, daemon=True)
            self.calibration_thread = Thread(target=self._calibration_worker, daemon=True)
            self.control_thread = Thread(target=self._control_worker, daemon=True)

            self.ai_thread.start()
            self.calibration_thread.start()
            self.control_thread.start()

            logger.info("3D Görselleştirme başlatıldı")

            # Arayüzü başlat
            if self.backend == "plotly":
                if PYQT_AVAILABLE:
                    self._start_plotly_qt_interface()
                else:
                    self._start_plotly_interface()
            elif self.backend == "pyvista":
                self._start_pyvista_interface()
            else:
                self._start_simple_interface()
        else:
            logger.warning("Görselleştirme zaten çalışıyor")

    def stop(self):
        """3D görselleştirmeyi durdurur."""
        if self.running.is_set():
            self.running.clear()

            # Soketleri kapat
            self.ai_socket.close()
            self.calibration_socket.close()
            self.control_socket.close()
            self.context.term()

            # İş parçacıklarını durdur
            if self.ai_thread:
                self.ai_thread.join(timeout=2.0)
            if self.calibration_thread:
                self.calibration_thread.join(timeout=2.0)
            if self.control_thread:
                self.control_thread.join(timeout=2.0)

            logger.info("3D Görselleştirme durduruldu")
        else:
            logger.warning("Görselleştirme zaten durdurulmuş")

    def _ai_worker(self):
        """AI verilerini işler."""
        logger.info("AI iş parçacığı başlatıldı")
        while self.running.is_set():
            try:
                try:
                    message = self.ai_socket.recv_json(flags=zmq.NOBLOCK)
                    if "detections" in message:
                        self.detections = message["detections"]
                        self._update_3d_model()
                except zmq.Again:
                    pass
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"AI veri hatası: {str(e)}")
                time.sleep(0.1)

    def _calibration_worker(self):
        """Kalibrasyon verilerini işler."""
        logger.info("Kalibrasyon iş parçacığı başlatıldı")
        while self.running.is_set():
            try:
                try:
                    message = self.calibration_socket.recv_json(flags=zmq.NOBLOCK)
                    if "ground_type" in message:
                        self.ground_type = message["ground_type"]
                        self._update_ground_layers()
                except zmq.Again:
                    pass
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Kalibrasyon veri hatası: {str(e)}")
                time.sleep(0.1)

    def _control_worker(self):
        """Kontrol komutlarını işler."""
        logger.info("Kontrol iş parçacığı başlatıldı")
        while self.running.is_set():
            try:
                try:
                    message = self.control_socket.recv_json(flags=zmq.NOBLOCK)
                    response = self._handle_control_command(message)
                    self.control_socket.send_json(response)
                except zmq.Again:
                    pass
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Kontrol hatası: {str(e)}")
                time.sleep(0.1)

    def _handle_control_command(self, message):
        """Kontrol komutlarını işler."""
        command = message.get("command", "")
        response = {"status": "error", "message": "Bilinmeyen komut"}

        if command == "status":
            response = {
                "status": "ok",
                "running": self.running.is_set(),
                "detections_count": len(self.detections),
                "ground_type": self.ground_type,
                "voxel_grid_size": self.grid_size,
                "confidence_threshold": self.confidence_threshold
            }
        elif command == "set_confidence_threshold":
            threshold = message.get("value", 0)
            if 0 < threshold <= 1.0:
                self.confidence_threshold = threshold
                response = {"status": "ok", "message": f"Eşik {threshold} olarak ayarlandı"}
            else:
                response = {"status": "error", "message": "Geçersiz eşik değeri (0-1 arası olmalı)"}
        elif command == "get_config":
            response = {
                "status": "ok",
                "config": {
                    "backend": self.backend,
                    "max_depth": self.max_depth,
                    "resolution": self.resolution,
                    "update_interval": self.update_interval,
                    "dark_mode": self.dark_mode
                }
            }
        elif command == "save_screenshot":
            filename = message.get("filename", "")
            result = self.save_screenshot(filename)
            if result:
                response = {"status": "ok", "path": result}
            else:
                response = {"status": "error", "message": "Ekran görüntüsü kaydedilemedi"}

        return response

    def _update_3d_model(self):
        """3D modeli günceller."""
        if not self.detections:
            return

        self.voxel_data.fill(0)
        self.detected_objects = []

        for detection in self.detections:
            if detection['confidence'] >= self.confidence_threshold:
                self._add_object_to_voxel(detection)

        # Görselleştirmeyi güncelle
        if self.backend == "plotly":
            self._update_plotly_view()
        elif self.backend == "pyvista":
            self._update_pyvista_view()
        else:
            self._update_simple_view()

    def _add_object_to_voxel(self, detection):
        """Nesneyi voxel verisine ekler."""
        try:
            obj_type = detection['type']
            obj_subtype = detection.get('subtype', 'default')
            x, y, z = detection['position']
            size = detection['size']

            # Pozisyonu grid indekslerine dönüştür
            xi = np.abs(self.x_coords - x).argmin()
            yi = np.abs(self.y_coords - y).argmin()
            zi = np.abs(self.z_coords - z).argmin()

            # Boyuta göre voxel sayısını hesapla
            size_in_voxels = max(1, int(size * self.resolution / self.max_depth))

            # Voxel verisini güncelle
            half_size = size_in_voxels // 2
            x_start = max(0, xi - half_size)
            x_end = min(self.grid_size[0], xi + half_size + 1)
            y_start = max(0, yi - half_size)
            y_end = min(self.grid_size[1], yi + half_size + 1)
            z_start = max(0, zi - half_size)
            z_end = min(self.grid_size[2], zi + half_size + 1)

            self.voxel_data[x_start:x_end, y_start:y_end, z_start:z_end] = detection['confidence']

            color = self.OBJECT_COLORS.get(obj_type, {}).get(
                obj_subtype,
                self.OBJECT_COLORS.get(obj_type, {}).get('default', (255, 0, 0)))

            self.detected_objects.append({
                'position': (x, y, z),
                'size': size,
                'type': obj_type,
                'subtype': obj_subtype,
                'color': color,
                'confidence': detection['confidence']
            })
        except Exception as e:
            logger.error(f"Nesne voxel verisine eklenirken hata: {str(e)}")

    def _update_ground_layers(self):
        """Zemin katmanlarını günceller."""
        if not self.ground_type:
            return

        ground_color = self.GROUND_COLORS.get(
            self.ground_type,
            self.GROUND_COLORS["default"])

        # Basit bir zemin modeli oluştur (gerçek uygulamada daha karmaşık olabilir)
        self.ground_layers = []
        layer_height = self.max_depth / 5  # 5 katmanlı bir yapı

        for i in range(5):
            depth = i * layer_height
            self.ground_layers.append({
                'depth': depth,
                'thickness': layer_height * 0.8,  # Katman kalınlığı
                'color': ground_color,
                'type': self.ground_type
            })

    def _start_plotly_qt_interface(self):
        """Plotly QT arayüzünü başlatır."""
        if not PYQT_AVAILABLE:
            self._start_plotly_interface()
            return

        self.app = QApplication(sys.argv)

        # Karanlık mod ayarı
        if self.dark_mode:
            self.app.setStyle('Fusion')
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
            self.app.setPalette(dark_palette)

        self.main_window = QMainWindow()
        self.main_window.setWindowTitle("3D Yeraltı Görüntüleyici")
        self.main_window.resize(1200, 800)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.main_window.setCentralWidget(central_widget)

        # Görselleştirme alanı
        self.visualization_widget = QLabel("3D Görselleştirme Yükleniyor...")
        self.visualization_widget.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.visualization_widget)

        # Kontrol paneli
        control_panel = QGroupBox("Kontrol Panel")
        control_layout = QGridLayout()

        # Eşik değeri kontrolü
        threshold_label = QLabel("Güven Eşiği:")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(int(self.confidence_threshold * 100))
        self.threshold_slider.valueChanged.connect(self._update_threshold)

        control_layout.addWidget(threshold_label, 0, 0)
        control_layout.addWidget(self.threshold_slider, 0, 1)

        # Görüntüleme seçenekleri
        self.show_objects_check = QCheckBox("Nesneleri Göster")
        self.show_objects_check.setChecked(True)
        self.show_ground_check = QCheckBox("Zemin Katmanlarını Göster")
        self.show_ground_check.setChecked(True)

        control_layout.addWidget(self.show_objects_check, 1, 0)
        control_layout.addWidget(self.show_ground_check, 1, 1)

        # Ekran görüntüsü butonu
        screenshot_btn = QPushButton("Ekran Görüntüsü Al")
        screenshot_btn.clicked.connect(lambda: self.save_screenshot())
        control_layout.addWidget(screenshot_btn, 2, 0, 1, 2)

        control_panel.setLayout(control_layout)
        layout.addWidget(control_panel)

        if self.fullscreen:
            self.main_window.showFullScreen()
        else:
            self.main_window.show()

        # Görselleştirme güncelleme timer'ı
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_plotly_view)
        self.update_timer.start(self.update_interval)

        sys.exit(self.app.exec_())

    def _update_plotly_view(self):
        """Plotly görünümünü günceller."""
        if not PLOTLY_AVAILABLE:
            return

        try:
            fig = go.Figure()

            # Zemin katmanlarını ekle
            if self.show_ground_check.isChecked():
                for layer in self.ground_layers:
                    depth = layer['depth']
                    thickness = layer['thickness']
                    color = f'rgb{layer["color"]}'

                    fig.add_trace(go.Mesh3d(
                        x=[-self.max_depth / 2, -self.max_depth / 2, self.max_depth / 2, self.max_depth / 2],
                        y=[-self.max_depth / 2, self.max_depth / 2, self.max_depth / 2, -self.max_depth / 2],
                        z=[depth, depth, depth, depth],
                        i=[0, 0],
                        j=[1, 2],
                        k=[2, 3],
                        opacity=0.5,
                        color=color,
                        name=f'{layer["type"]} layer at {depth:.1f}m'
                    ))

            # Tespit edilen nesneleri ekle
            if self.show_objects_check.isChecked():
                for obj in self.detected_objects:
                    x, y, z = obj['position']
                    size = obj['size']
                    color = f'rgb{obj["color"]}'

                    fig.add_trace(go.Scatter3d(
                        x=[x],
                        y=[y],
                        z=[z],
                        mode='markers',
                        marker=dict(
                            size=size * 10,
                            color=color,
                            opacity=0.8
                        ),
                        name=f'{obj["type"]} ({obj["subtype"]})'
                    ))

            # Kamera ve sahne ayarları
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title='X (m)', range=[-self.max_depth / 2, self.max_depth / 2]),
                    yaxis=dict(title='Y (m)', range=[-self.max_depth / 2, self.max_depth / 2]),
                    zaxis=dict(title='Derinlik (m)', range=[0, self.max_depth]),
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=1)
                ),
                title="3D Yeraltı Görüntüleme",
                margin=dict(l=0, r=0, b=0, t=30)
            )

            # Görüntüyü güncelle
            if PYQT_AVAILABLE:
                # Plotly görüntüsünü Qt'ye gömme kodu buraya gelecek
                pass
            else:
                fig.show()

        except Exception as e:
            logger.error(f"Plotly görüntüleme hatası: {str(e)}")

    def _update_threshold(self, value):
        """Güven eşiğini günceller."""
        self.confidence_threshold = value / 100.0
        self._update_3d_model()

    def _start_plotly_interface(self):
        """Standalone Plotly arayüzünü başlatır."""
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly kütüphanesi kullanılamıyor")
            return

        self.fig = go.Figure()
        self._update_plotly_view()
        self.fig.show()

    def _start_pyvista_interface(self):
        """PyVista arayüzünü başlatır."""
        if not PYVISTA_AVAILABLE:
            logger.error("PyVista kütüphanesi kullanılamıyor")
            return

        self.plotter = pv.Plotter()
        self.plotter.add_axes()
        self.plotter.show_grid()
        self.plotter.set_background('black' if self.dark_mode else 'white')
        self._update_pyvista_view()
        self.plotter.show()

    def _update_pyvista_view(self):
        """PyVista görünümünü günceller."""
        if not PYVISTA_AVAILABLE or not hasattr(self, 'plotter'):
            return

        try:
            self.plotter.clear()

            # Zemin katmanlarını ekle
            for layer in self.ground_layers:
                depth = layer['depth']
                thickness = layer['thickness']
                color = np.array(layer['color']) / 255.0

                # Dikdörtgen prizma oluştur
                box = pv.Box(
                    bounds=[
                        -self.max_depth / 2, self.max_depth / 2,
                        -self.max_depth / 2, self.max_depth / 2,
                        depth, depth + thickness
                    ]
                )
                self.plotter.add_mesh(
                    box,
                    color=color,
                    opacity=0.5,
                    name=f'{layer["type"]}_layer'
                )

            # Nesneleri ekle
            for obj in self.detected_objects:
                x, y, z = obj['position']
                size = obj['size']
                color = np.array(obj['color']) / 255.0

                sphere = pv.Sphere(radius=size / 2, center=(x, y, z))
                self.plotter.add_mesh(
                    sphere,
                    color=color,
                    opacity=0.8,
                    name=f'{obj["type"]}_{obj["subtype"]}'
                )

            self.plotter.add_text(
                f"3D Yeraltı Görüntüleme\nMax Derinlik: {self.max_depth}m\nÇözünürlük: {self.resolution}",
                position='upper_left',
                font_size=10,
                color='white' if self.dark_mode else 'black'
            )

        except Exception as e:
            logger.error(f"PyVista görüntüleme hatası: {str(e)}")

    def _start_simple_interface(self):
        """Basit matplotlib arayüzünü başlatır."""
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        if self.dark_mode:
            self.fig.patch.set_facecolor('black')
            self.ax.set_facecolor('black')
            self.ax.xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
            self.ax.yaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
            self.ax.zaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
            self.ax.tick_params(colors='white')
            self.ax.xaxis.label.set_color('white')
            self.ax.yaxis.label.set_color('white')
            self.ax.zaxis.label.set_color('white')
            self.ax.title.set_color('white')

        self._update_simple_view()
        plt.show()

    def _update_simple_view(self):
        """Basit görünümü günceller."""
        if not hasattr(self, 'ax'):
            return

        try:
            self.ax.clear()

            # Zemin katmanlarını çiz
            for layer in self.ground_layers:
                depth = layer['depth']
                thickness = layer['thickness']
                color = np.array(layer['color']) / 255.0

                # Katman için dikdörtgen prizma çiz
                xx, yy = np.meshgrid(
                    [-self.max_depth / 2, self.max_depth / 2],
                    [-self.max_depth / 2, self.max_depth / 2]
                )
                zz = np.ones_like(xx) * depth

                self.ax.plot_surface(xx, yy, zz, color=color, alpha=0.5)
                self.ax.plot_surface(xx, yy, zz + thickness, color=color, alpha=0.5)

            # Nesneleri çiz
            for obj in self.detected_objects:
                x, y, z = obj['position']
                size = obj['size']
                color = np.array(obj['color']) / 255.0

                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                xx = x + size / 2 * np.outer(np.cos(u), np.sin(v))
                yy = y + size / 2 * np.outer(np.sin(u), np.sin(v))
                zz = z + size / 2 * np.outer(np.ones(np.size(u)), np.cos(v))

                self.ax.plot_surface(xx, yy, zz, color=color, alpha=0.8)

            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Derinlik (m)')
            self.ax.set_title('3D Yeraltı Görüntüleme')
            self.ax.set_xlim(-self.max_depth / 2, self.max_depth / 2)
            self.ax.set_ylim(-self.max_depth / 2, self.max_depth / 2)
            self.ax.set_zlim(0, self.max_depth)

            plt.draw()
            plt.pause(0.01)

        except Exception as e:
            logger.error(f"Matplotlib görüntüleme hatası: {str(e)}")

    def save_screenshot(self, filename=None):
        """Ekran görüntüsünü kaydeder."""
        if not filename:
            filename = f"3d_view_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        try:
            save_path = os.path.join(self.save_path, filename)

            if self.backend == "plotly" and hasattr(self, 'fig'):
                self.fig.write_image(save_path)
            elif self.backend == "pyvista" and hasattr(self, 'plotter'):
                self.plotter.screenshot(save_path)
            elif self.backend == "simple" and hasattr(self, 'fig'):
                self.fig.savefig(save_path, dpi=300, facecolor=self.fig.get_facecolor())
            else:
                return None

            logger.info(f"Ekran görüntüsü kaydedildi: {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Ekran görüntüsü kaydedilemedi: {str(e)}")
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='3D Yeraltı Görselleştirici',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--backend', choices=['plotly', 'pyvista', 'simple'],
                        default='plotly', help='Görselleştirme backend seçeneği')
    parser.add_argument('--fullscreen', action='store_true',
                        help='Tam ekran modu')
    parser.add_argument('--dark-mode', action='store_true',
                        help='Karanlık mod')
    parser.add_argument('--update-interval', type=int, default=100,
                        help='Güncelleme aralığı (ms)')
    parser.add_argument('--max-depth', type=float, default=10.0,
                        help='Maksimum görüntüleme derinliği (metre)')
    parser.add_argument('--resolution', type=int, default=50,
                        help='Voxel grid çözünürlüğü')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                        help='Güven eşik değeri (0-1 arası)')
    parser.add_argument('--save-path', type=str, default="screenshots",
                        help='Ekran görüntülerinin kaydedileceği dizin')
    parser.add_argument('--ai-port', type=int, default=5555,
                        help='AI modülü ZMQ portu')
    parser.add_argument('--calibration-port', type=int, default=5556,
                        help='Kalibrasyon modülü ZMQ portu')
    parser.add_argument('--control-port', type=int, default=5557,
                        help='Kontrol ZMQ portu')

    args = parser.parse_args()

    # Dizin oluştur
    os.makedirs(args.save_path, exist_ok=True)

    viewer = Depth3DView(args)
    try:
        viewer.start()
    except KeyboardInterrupt:
        viewer.stop()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Kritik hata: {str(e)}")
        viewer.stop()
        sys.exit(1)