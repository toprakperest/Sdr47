#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
depth_3d_view.py - Enhanced 3D Visualization Module

Version: 2.0
Author: AI Assistant (Manus)
Last Updated: 2025-04-30

Bu modül, SDR tabanlı yeraltı tespit sistemi için 3D görselleştirme arayüzünü gerçekleştirir.
Preprocessing modülünden gelen verileri kullanarak gerçek zamanlı 3D yeraltı modeli oluşturur.

Geliştirmeler:
- Preprocessing modülü ile ZMQ entegrasyonu.
- Gerçek zamanlı 3D model güncellemeleri (PyVista ve Plotly).
- Belirtilen renk kodlaması (toprak, taş, boşluk, metal).
- Derinlik gösterimi için opaklık/parlaklık kullanımı.
- Daha modüler yapı ve PyVista önceliği.
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
from datetime import datetime
from threading import Thread, Event, Lock
from queue import Queue, Empty
import matplotlib.cm as cm
from typing import Tuple, Dict, Any, List

# Yerel modüller
from config import (
    GroundType, COLOR_MAP_GROUND, COLOR_MAP_TARGET,
    PORT_PREPROCESSING_OUTPUT, PORT_VISUALIZATION_CONTROL,
    PORT_CALIBRATION_RESULT # Kalibrasyondan zemin tipi almak için
)

# PyVista kütüphanesi için koşullu import
try:
    import pyvista as pv
    from pyvistaqt import BackgroundPlotter # PyQt ile entegrasyon için
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("PyVista kütüphanesi bulunamadı. Plotly veya basit görselleştirme kullanılacak.")

# Plotly kütüphanesi için koşullu import
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly kütüphanesi bulunamadı. PyVista veya basit görselleştirme kullanılacak.")

# PyQt5 (PyVista entegrasyonu için)
try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
    from PyQt5.QtCore import QTimer
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt5 bulunamadı. PyVista arayüzü ayrı pencerede çalışacak.")

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format=\'%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s\',
    handlers=[
        logging.FileHandler("depth_3d_view.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("depth_3d_view")

class Depth3DView:
    """3D yeraltı görselleştirmesi için ana sınıf."""

    def __init__(self, config: dict):
        """
        Görselleştirme modülünü başlatır.

        Args:
            config (dict): Modül yapılandırma parametreleri.
        """
        self.config = config
        self.backend = config.get("backend", "pyvista")
        self.update_interval_ms = config.get("update_interval_ms", 200)
        self.max_depth = config.get("max_depth_m", 5.0)
        self.grid_resolution = config.get("resolution", 50) # Grid boyutu (nx, ny, nz)
        self.grid_size = (self.grid_resolution, self.grid_resolution, self.grid_resolution)
        self.opacity_scaling = config.get("opacity_scaling", True)
        self.brightness_scaling = config.get("brightness_scaling", False)
        self.show_ground = config.get("show_ground_layers", True)
        self.dark_mode = config.get("dark_mode", True)

        # Backend kontrolü
        if self.backend == "pyvista" and not PYVISTA_AVAILABLE:
            logger.warning("PyVista kullanılamıyor, Plotly deneniyor.")
            self.backend = "plotly"
        if self.backend == "plotly" and not PLOTLY_AVAILABLE:
            logger.warning("Plotly kullanılamıyor, basit backend kullanılacak.")
            self.backend = "simple"
        if self.backend == "pyvista" and not PYQT_AVAILABLE:
             logger.warning("PyQt5 bulunamadı, PyVista ayrı pencerede çalışacak.")
             # Ayrı pencere için özel handle gerekebilir

        # Veri yapıları
        self.voxel_grid = None # PyVista grid nesnesi
        self.plotter = None # PyVista plotter nesnesi
        self.plotly_figure = None # Plotly figür nesnesi
        self.qt_app = None
        self.qt_window = None
        self.qt_plotter_widget = None

        self.latest_targets: List[Dict[str, Any]] = []
        self.current_ground_type = GroundType.UNKNOWN
        self.data_lock = Lock()

        # İletişim
        self.context = zmq.Context()
        # Preprocessing Veri Girişi
        self.preprocessing_socket = self.context.socket(zmq.SUB)
        self.preprocessing_socket.connect(f"tcp://localhost:{config.get(\'input_port_preprocessing\', PORT_PREPROCESSING_OUTPUT)}")
        self.preprocessing_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.preprocessing_socket.setsockopt(zmq.CONFLATE, 1)
        # Kalibrasyon Veri Girişi
        self.calibration_socket = self.context.socket(zmq.SUB)
        self.calibration_socket.connect(f"tcp://localhost:{config.get(\'input_port_calibration\', PORT_CALIBRATION_RESULT)}")
        self.calibration_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.calibration_socket.setsockopt(zmq.CONFLATE, 1)
        # Kontrol Girişi
        self.control_socket = self.context.socket(zmq.REP)
        self.control_socket.bind(f"tcp://*:{config.get(\'control_port\', PORT_VISUALIZATION_CONTROL)}")

        # İş parçacıkları
        self.running = Event()
        self.threads = {}

        logger.info(f"3D Görselleştirme modülü başlatıldı (Backend: {self.backend})")

    def start(self):
        """Görselleştirme thread\lerini ve arayüzünü başlatır."""
        if self.running.is_set():
            logger.warning("Görselleştirme zaten çalışıyor.")
            return
            
        self.running.set()

        # Thread\leri başlat
        self.threads["preprocessing_listener"] = Thread(target=self._preprocessing_listener_worker, name="PreprocessingListener")
        self.threads["calibration_listener"] = Thread(target=self._calibration_listener_worker, name="CalibrationListener")
        self.threads["control"] = Thread(target=self._control_worker, name="Control")

        for name, thread in self.threads.items():
            thread.daemon = True
            thread.start()
            logger.info(f"{name} thread başlatıldı.")

        # Arayüzü ana thread üzerinde başlat
        if self.backend == "pyvista":
            self._start_pyvista_interface()
        elif self.backend == "plotly":
            self._start_plotly_interface() # Plotly genellikle web tabanlı çalışır
        else:
            logger.info("Basit görselleştirme modu aktif (arayüz yok).")
            # Basit modda ana thread\i canlı tutmak için
            try:
                while self.running.is_set():
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop()

    def stop(self):
        """Görselleştirme modülünü durdurur."""
        if not self.running.is_set():
            return
        self.running.clear()
        logger.info("Görselleştirme durduruluyor...")

        # Thread\leri durdur
        for name, thread in self.threads.items():
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
                logger.info(f"{name} thread durduruldu.")

        # ZMQ soketlerini kapat
        self.preprocessing_socket.close()
        self.calibration_socket.close()
        self.control_socket.close()
        self.context.term()

        # Arayüzü kapat
        if self.backend == "pyvista" and self.plotter:
            try:
                self.plotter.close()
            except Exception as e:
                logger.warning(f"PyVista plotter kapatılırken hata: {e}")
        if self.qt_app:
            self.qt_app.quit()

        logger.info("Görselleştirme modülü durduruldu.")

    def _preprocessing_listener_worker(self):
        """Preprocessing modülünden gelen ZMQ verilerini dinler."""
        logger.info("Preprocessing dinleyici worker başlatıldı.")
        while self.running.is_set():
            try:
                message = self.preprocessing_socket.recv_json()
                # Veriyi doğrula
                if "targets" in message and "timestamp" in message:
                    with self.data_lock:
                        # Gelen target listesi {index: {depth, amp, phase, type_guess}}
                        # Bunu daha kullanışlı bir listeye çevirelim
                        targets_list = []
                        raw_targets = message.get("targets", {})
                        for index, props in raw_targets.items():
                             # Pozisyonu index\ten tahmin et? Veya preprocessing eklemeli?
                             # Şimdilik sadece derinliği kullanalım
                             targets_list.append({
                                 "id": f"target_{message[\"timestamp\"]}_{index}", # Benzersiz ID
                                 "depth": props.get("depth_m", 0),
                                 "amplitude": props.get("amplitude", 0),
                                 "phase": props.get("avg_phase_rad", 0),
                                 "type_guess": props.get("type_guess", "unknown"),
                                 "position": [0, 0, props.get("depth_m", 0)], # X, Y şimdilik 0
                                 "size": props.get("amplitude", 0.1) * 0.5 # Boyutu genlikle oranla (ayarlanmalı)
                             })
                        self.latest_targets = targets_list
                    # logger.debug(f"{len(self.latest_targets)} hedef alındı.")
                else:
                    logger.warning(f"Preprocessing mesajında eksik alanlar: {message.keys()}")
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM: break
                logger.error(f"ZMQ Preprocessing veri alma hatası: {e}")
                time.sleep(1)
            except json.JSONDecodeError:
                logger.error("Geçersiz JSON formatında Preprocessing mesajı alındı.")
            except Exception as e:
                logger.error(f"Preprocessing dinleyici worker hatası: {e}", exc_info=True)
                time.sleep(1)
        logger.info("Preprocessing dinleyici worker durdu.")

    def _calibration_listener_worker(self):
        """Kalibrasyon modülünden gelen ZMQ verilerini dinler."""
        logger.info("Kalibrasyon dinleyici worker başlatıldı.")
        while self.running.is_set():
            try:
                message = self.calibration_socket.recv_json()
                with self.data_lock:
                    if "ground_type" in message:
                        try:
                            new_ground_type = GroundType[message["ground_type"]]
                            if new_ground_type != self.current_ground_type:
                                 self.current_ground_type = new_ground_type
                                 logger.info(f"Zemin tipi güncellendi: {self.current_ground_type.name}")
                                 # Zemin katmanlarını güncelleme ihtiyacı olabilir
                        except KeyError:
                            logger.warning(f"Bilinmeyen zemin tipi adı: {message[\"ground_type\"]}")
                            self.current_ground_type = GroundType.UNKNOWN
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM: break
                logger.error(f"ZMQ Kalibrasyon veri alma hatası: {e}")
                time.sleep(1)
            except json.JSONDecodeError:
                logger.error("Geçersiz JSON formatında Kalibrasyon mesajı alındı.")
            except Exception as e:
                logger.error(f"Kalibrasyon dinleyici worker hatası: {e}", exc_info=True)
                time.sleep(1)
        logger.info("Kalibrasyon dinleyici worker durdu.")

    def _control_worker(self):
        """Kontrol komutlarını dinler ve işler."""
        logger.info("Kontrol worker başlatıldı.")
        while self.running.is_set():
            try:
                message = self.control_socket.recv_json()
                command = message.get("command")
                response = {"status": "ok"}
                logger.info(f"Kontrol komutu alındı: {command}")

                if command == "get_status":
                    with self.data_lock:
                        response["data"] = {
                            "running": self.running.is_set(),
                            "backend": self.backend,
                            "target_count": len(self.latest_targets),
                            "ground_type": self.current_ground_type.name
                        }
                elif command == "set_parameter":
                    param = message.get("parameter")
                    value = message.get("value")
                    # TODO: Parametre ayarlama mantığı (örn. max_depth, resolution)
                    response = {"status": "error", "message": "Parametre ayarı henüz uygulanmadı"}
                elif command == "save_screenshot":
                    filename = message.get("filename", f"screenshot_{int(time.time())}.png")
                    if self._save_screenshot(filename):
                         response["message"] = f"Ekran görüntüsü kaydedildi: {filename}"
                    else:
                         response = {"status": "error", "message": "Ekran görüntüsü kaydedilemedi"}
                elif command == "stop":
                    self.stop()
                    response["message"] = "Görselleştirme durduruluyor."
                else:
                    response["status"] = "error"
                    response["message"] = "Bilinmeyen komut"

                self.control_socket.send_json(response)

            except zmq.ZMQError as e:
                 if e.errno == zmq.ETERM: break
                 logger.error(f"ZMQ kontrol hatası: {e}")
                 time.sleep(1)
            except Exception as e:
                logger.error(f"Kontrol worker hatası: {str(e)}")
                try: self.control_socket.send_json({"status": "error", "message": str(e)}) 
                except: pass
                time.sleep(1)
        logger.info("Kontrol worker durdu.")

    # --- PyVista Arayüz Metotları --- 
    def _start_pyvista_interface(self):
        """PyVista görselleştirme arayüzünü başlatır."""
        if not PYVISTA_AVAILABLE: return

        # Grid oluştur
        # Boyutları dünya koordinatlarına göre ayarla
        spacing = (self.max_depth / self.grid_resolution, 
                   self.max_depth / self.grid_resolution, 
                   self.max_depth / self.grid_resolution)
        origin = (-self.max_depth / 2, -self.max_depth / 2, 0) # X, Y merkezde, Z yüzeyde başlar
        self.voxel_grid = pv.UniformGrid(
            dimensions=self.grid_size,
            spacing=spacing,
            origin=origin
        )
        # Başlangıçta boş skaler değerler ata
        self.voxel_grid["target_confidence"] = np.zeros(self.voxel_grid.n_cells)
        self.voxel_grid["target_type"] = np.zeros(self.voxel_grid.n_cells, dtype=int)

        # Plotter oluştur
        if PYQT_AVAILABLE:
            self.qt_app = QApplication.instance() or QApplication(sys.argv)
            self.qt_window = QMainWindow()
            self.qt_window.setWindowTitle("Mesopotamia GPR - 3D Görünüm")
            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)
            self.qt_window.setCentralWidget(central_widget)
            
            self.plotter = BackgroundPlotter(app=self.qt_app, window_size=(800, 600), show=False)
            layout.addWidget(self.plotter)
            self.qt_window.show()
        else:
            self.plotter = pv.Plotter(window_size=[800, 600])

        if self.dark_mode:
            self.plotter.set_background("black")
        else:
            self.plotter.set_background("white")

        # Başlangıç görünümünü ayarla
        self.plotter.add_axes()
        self.plotter.camera_position = "iso"
        self.plotter.camera.zoom(1.5)

        # Periyodik güncelleme için timer
        if PYQT_AVAILABLE:
            self.qt_timer = QTimer()
            self.qt_timer.timeout.connect(self._update_pyvista_view)
            self.qt_timer.start(self.update_interval_ms)
            self.qt_app.exec_() # PyQt olay döngüsünü başlat
        else:
            # PyQt yoksa, basit bir döngü ile güncelle (bloklayıcı olabilir)
            self.plotter.show(interactive_update=True)
            while self.running.is_set():
                self._update_pyvista_view()
                self.plotter.update()
                time.sleep(self.update_interval_ms / 1000.0)

    def _update_pyvista_view(self):
        """PyVista görünümünü günceller."""
        if not self.plotter or not self.voxel_grid: return

        with self.data_lock:
            targets = self.latest_targets
            ground_type = self.current_ground_type

        # Voxel gridini temizle (veya güncelle)
        current_confidence = np.zeros(self.voxel_grid.n_cells)
        current_type = np.zeros(self.voxel_grid.n_cells, dtype=int)
        target_meshes = []

        # Hedefleri voxel gridine veya ayrı mesh olarak ekle
        for target in targets:
            pos = target["position"]
            size = target.get("size", 0.1) # Boyut bilgisi yoksa varsayılan
            depth = pos[2]
            type_guess = target.get("type_guess", "unknown").upper()
            confidence = target.get("amplitude", 0.5) # Genliği güven skoru gibi kullan

            # Renk ve tip ID belirle
            color = COLOR_MAP_TARGET.get(type_guess, COLOR_MAP_TARGET["UNKNOWN"]) # RGB tuple
            type_id = list(COLOR_MAP_TARGET.keys()).index(type_guess) if type_guess in COLOR_MAP_TARGET else -1

            # Opaklık/Parlaklık (Derinlikle ters orantılı)
            opacity = 1.0 - (depth / self.max_depth) if self.opacity_scaling else 1.0
            opacity = max(0.1, min(1.0, opacity)) # Sınırla
            
            # Parlaklık ayarı için rengi değiştir (HSV veya benzeri)
            # TODO: Rengi parlaklığa göre ayarla

            # Seçenek 1: Voxel gridini doldur
            # cell_indices = self.voxel_grid.find_closest_cell(pos)
            # if cell_indices >= 0:
            #     current_confidence[cell_indices] = confidence
            #     current_type[cell_indices] = type_id
            
            # Seçenek 2: Her hedef için ayrı mesh oluştur (daha esnek)
            if type_guess == "VOID":
                 # Boşlukları küre veya kutu olarak göster (siyah?)
                 mesh = pv.Sphere(center=pos, radius=size/2)
                 color = COLOR_MAP_TARGET["VOID"]
            elif type_guess == "METAL":
                 mesh = pv.Cube(center=pos, x_length=size, y_length=size, z_length=size/2)
                 color = COLOR_MAP_TARGET["METAL"]
            else: # Diğerleri (taş, toprak parçası vb.)
                 mesh = pv.Sphere(center=pos, radius=size/2)
                 color = COLOR_MAP_TARGET.get(type_guess, COLOR_MAP_TARGET["UNKNOWN"])
                 
            target_meshes.append((mesh, color, opacity))

        # Plotter\ı temizle (önceki aktörleri kaldır)
        # self.plotter.clear_actors() # Bu yavaş olabilir
        # Daha iyisi: Aktörleri isimle yönet
        current_actors = self.plotter.actors.copy()
        for name, actor in current_actors.items():
             if name.startswith("target_") or name.startswith("ground_"):
                 self.plotter.remove_actor(actor, render=False)

        # Yeni hedef meshlerini ekle
        for i, (mesh, color, opacity) in enumerate(target_meshes):
            self.plotter.add_mesh(mesh, color=color, opacity=opacity, name=f"target_{i}", render=False)

        # Zemin katmanlarını ekle (opsiyonel)
        if self.show_ground:
            ground_color = COLOR_MAP_GROUND.get(ground_type, COLOR_MAP_GROUND[GroundType.UNKNOWN])
            # Basit bir düzlem olarak yüzeyi göster
            ground_surface = pv.Plane(center=(0, 0, -0.01), direction=(0, 0, 1), 
                                      i_size=self.max_depth, j_size=self.max_depth)
            self.plotter.add_mesh(ground_surface, color=ground_color, opacity=0.3, name="ground_surface", render=False)
            # TODO: Daha gerçekçi katmanlı zemin modeli eklenebilir

        # Voxel gridini güncelle (eğer kullanılıyorsa)
        # self.voxel_grid["target_confidence"] = current_confidence
        # self.voxel_grid["target_type"] = current_type
        # self.plotter.add_mesh(self.voxel_grid.threshold(0.1, scalars="target_confidence"), 
        #                       scalars="target_type", cmap=list(COLOR_MAP_TARGET.values()), 
        #                       opacity="linear", name="voxel_targets", render=False)

        # Render işlemini tetikle (eğer BackgroundPlotter değilse)
        if not PYQT_AVAILABLE:
             self.plotter.render()
        # BackgroundPlotter otomatik render yapar

    # --- Plotly Arayüz Metotları (Placeholder) --- 
    def _start_plotly_interface(self):
        """Plotly görselleştirme arayüzünü başlatır (genellikle web tabanlı)."""
        if not PLOTLY_AVAILABLE: return
        logger.info("Plotly arayüzü başlatılıyor (Web tarayıcıda açılabilir)...")
        # Plotly Dash uygulaması veya benzeri bir yapı gerekir
        # Bu kısım daha detaylı implementasyon gerektirir.
        # Şimdilik basit bir figür oluşturup periyodik güncelleme simüle edilebilir.
        self.plotly_figure = go.Figure()
        # TODO: Plotly Dash uygulaması kur
        # self.app.run_server(debug=True) 
        pass

    def _update_plotly_view(self):
        """Plotly görünümünü günceller."""
        if not self.plotly_figure: return
        # TODO: Plotly figürünü güncelleme mantığı
        pass

    # --- Diğer Metotlar --- 
    def _save_screenshot(self, filename: str) -> str:
        """Mevcut görünümün ekran görüntüsünü kaydeder."""
        try:
            if self.backend == "pyvista" and self.plotter:
                self.plotter.screenshot(filename)
                logger.info(f"Ekran görüntüsü kaydedildi: {filename}")
                return os.path.abspath(filename)
            elif self.backend == "plotly" and self.plotly_figure:
                # Plotly figürünü kaydet
                # self.plotly_figure.write_image(filename)
                logger.warning("Plotly ekran görüntüsü kaydetme henüz uygulanmadı.")
                return ""
            else:
                logger.warning("Ekran görüntüsü alınamadı (desteklenmeyen backend veya plotter yok).")
                return ""
        except Exception as e:
            logger.error(f"Ekran görüntüsü kaydetme hatası: {e}")
            return ""

# Ana execution bloğu (standalone test için)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Visualization Module")
    parser.add_argument("--config", type=str, default="system_config.json", help="Path to JSON configuration file")
    # Diğer argümanlar config dosyasından alınacak

    args = parser.parse_args()

    # Load config from file
    config = {}
    try:
        with open(args.config, "r") as f:
            config_from_file = json.load(f)
            config = config_from_file.get("module_configs", {}).get("visualization", {})
            logger.info(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}. Using defaults.")
    except json.JSONDecodeError:
         logger.error(f"Error decoding JSON from {args.config}. Using defaults.")

    # Görselleştirme modülünü başlat
    visualizer = Depth3DView(config)
    
    # Başlatma ana thread\de yapılmalı
    visualizer.start()
    # Start metodu içinde arayüz döngüsü başlatılır (PyQt veya Plotly Dash)
    # Eğer basit mod ise, start metodu bloklar veya KeyboardInterrupt beklenir.

