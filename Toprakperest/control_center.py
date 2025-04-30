#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
control_center.py

SDR tabanlı yeraltı tespit sistemi için ana kontrol merkezi.
Tüm modülleri başlatır, koordine eder ve kullanıcı arayüzü sağlar.

Geliştirmeler:
- Hata yönetimi iyileştirildi
- Yapılandırma doğrulama eklendi
- ZMQ timeout ve reconnect özellikleri
- Süreç izleme ve otomatik restart
- UI fallback mekanizması
"""

import os
import sys
import time
import json
import argparse
import logging
import signal
import subprocess
import threading
import queue
import zmq
import platform
from datetime import datetime
from enum import Enum, auto
import traceback

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("control_center.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("control_center")


class ModuleStatus(Enum):
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    ERROR = auto()
    RESTARTING = auto()


class ControlCenter:
    """
    SDR tabanlı yeraltı tespit sistemi için ana kontrol merkezi.
    """

    DEFAULT_PORTS = {
        "sdr_pub": 5555,
        "sdr_control": 5556,
        "calibration_pub": 5557,
        "calibration_control": 5558,
        "ai_pub": 5559,
        "ai_control": 5560,
        "radar_control": 5561,
        "3d_control": 5562,
        "control_pub": 5563,
        "control_rep": 5564
    }

    def __init__(self, args):
        self.args = args
        self.config_file = args.config
        self.debug_mode = args.debug
        self.test_mode = args.test_mode
        self.auto_start = args.auto_start
        self.headless = args.headless

        # Sistem durumu
        self.system_status = {
            "running": False,
            "start_time": None,
            "scan_count": 0,
            "detection_count": 0,
            "ground_type": None,
            "last_calibration": None,
            "error_count": 0,
            "last_error": None
        }

        # Modül yönetimi
        self.modules = {
            "sdr_receiver": {
                "status": ModuleStatus.STOPPED,
                "process": None,
                "restart_count": 0,
                "last_restart": None
            },
            "calibration": {
                "status": ModuleStatus.STOPPED,
                "process": None,
                "restart_count": 0,
                "last_restart": None
            },
            "ai_classifier": {
                "status": ModuleStatus.STOPPED,
                "process": None,
                "restart_count": 0,
                "last_restart": None
            },
            "radar_ui": {
                "status": ModuleStatus.STOPPED,
                "process": None,
                "restart_count": 0,
                "last_restart": None
            },
            "depth_3d_view": {
                "status": ModuleStatus.STOPPED,
                "process": None,
                "restart_count": 0,
                "last_restart": None
            },
            "logger": {
                "status": ModuleStatus.STOPPED,
                "process": None,
                "restart_count": 0,
                "last_restart": None
            }
        }

        # Veri yapıları
        self.detections = []
        self.calibration_data = {}
        self.sdr_data = {}
        self.ai_data = {}

        # İletişim
        self.context = None
        self.sockets = {}
        self.data_queue = queue.Queue(maxsize=100)
        self.command_queue = queue.Queue()

        # Thread yönetimi
        self.threads = {}
        self.running = threading.Event()

        # Yapılandırma
        self.config = self._load_config()
        self._validate_config()

        # UI
        self.ui = None

        logger.info("Kontrol merkezi örneği oluşturuldu")

    def _load_config(self):
        """Yapılandırma dosyasını yükler ve doğrular."""
        default_config = {
            "sdr": {
                "center_freq": 500e6,
                "sample_rate": 2e6,
                "gain": 20,
                "freq_range": [300e6, 3e9]
            },
            "calibration": {
                "auto_calibrate": True,
                "interval": 300
            },
            "ai": {
                "model_path": "models",
                "confidence_threshold": 0.7
            },
            "ui": {
                "fullscreen": False,
                "dark_mode": True,
                "update_interval": 100,
                "max_range": 5,
                "max_depth": 3.0,
                "resolution": 32,
                "3d_backend": "plotly"
            },
            "logger": {
                "log_level": "INFO",
                "log_dir": "logs"
            },
            "ports": self.DEFAULT_PORTS,
            "module_restart_limit": 3,
            "module_restart_delay": 5
        }

        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                    self._merge_dicts(default_config, user_config)
                    logger.info(f"Yapılandırma dosyası yüklendi: {self.config_file}")
            else:
                os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
                with open(self.config_file, 'w') as f:
                    json.dump(default_config, f, indent=4)
                    logger.info(f"Varsayılan yapılandırma oluşturuldu: {self.config_file}")

            return default_config

        except Exception as e:
            logger.error(f"Yapılandırma yükleme hatası: {str(e)}")
            return default_config

    def _validate_config(self):
        """Yapılandırma ayarlarını doğrular."""
        try:
            # Frekans kontrolü
            if not (300e6 <= self.config['sdr']['center_freq'] <= 3e9):
                raise ValueError("Geçersiz merkez frekansı")

            # Port kontrolü
            for port in self.config['ports'].values():
                if not (1024 <= port <= 65535):
                    raise ValueError("Geçersiz port numarası")

            logger.info("Yapılandırma doğrulaması başarılı")

        except Exception as e:
            logger.error(f"Yapılandırma doğrulama hatası: {str(e)}")
            raise

    def _merge_dicts(self, target, source):
        """İç içe sözlükleri birleştirir."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_dicts(target[key], value)
            else:
                target[key] = value

    def start(self):
        """Sistemi başlatır."""
        try:
            self.running.set()
            self.system_status['running'] = True
            self.system_status['start_time'] = datetime.now()

            # ZMQ bağlamını oluştur
            self.context = zmq.Context()
            self._setup_zmq_sockets()

            # İzleme thread'lerini başlat
            self._start_monitoring_threads()

            # UI başlat
            if not self.headless:
                self._start_ui()

            # Otomatik başlatma
            if self.auto_start:
                self.start_all_modules()

            # Ana döngü
            self._main_loop()

            return True

        except Exception as e:
            logger.error(f"Başlatma hatası: {str(e)}")
            self.stop()
            return False

    def stop(self):
        """Sistemi durdurur."""
        try:
            self.running.clear()
            self.system_status['running'] = False

            # UI durdur
            if self.ui:
                self.ui.stop()

            # Modülleri durdur
            self.stop_all_modules()

            # Thread'leri durdur
            for name, thread in self.threads.items():
                if thread.is_alive():
                    thread.join(timeout=2.0)

            # ZMQ soketlerini kapat
            for socket in self.sockets.values():
                socket.close()

            if self.context:
                self.context.term()

            logger.info("Sistem durduruldu")
            return True

        except Exception as e:
            logger.error(f"Durdurma hatası: {str(e)}")
            return False

    def _setup_zmq_sockets(self):
        """ZMQ soketlerini oluşturur ve yapılandırır."""
        try:
            # Publisher soketleri
            self.sockets['control_pub'] = self.context.socket(zmq.PUB)
            self.sockets['control_pub'].bind(f"tcp://*:{self.config['ports']['control_pub']}")

            # REP soketi (komutlar için)
            self.sockets['control_rep'] = self.context.socket(zmq.REP)
            self.sockets['control_rep'].bind(f"tcp://*:{self.config['ports']['control_rep']}")
            self.sockets['control_rep'].setsockopt(zmq.RCVTIMEO, 100)
            self.sockets['control_rep'].setsockopt(zmq.LINGER, 0)

            # SUB soketleri (modül verileri için)
            self.sockets['sdr_sub'] = self.context.socket(zmq.SUB)
            self.sockets['sdr_sub'].connect(f"tcp://localhost:{self.config['ports']['sdr_pub']}")
            self.sockets['sdr_sub'].setsockopt_string(zmq.SUBSCRIBE, "")
            self.sockets['sdr_sub'].setsockopt(zmq.RCVTIMEO, 100)

            self.sockets['calibration_sub'] = self.context.socket(zmq.SUB)
            self.sockets['calibration_sub'].connect(f"tcp://localhost:{self.config['ports']['calibration_pub']}")
            self.sockets['calibration_sub'].setsockopt_string(zmq.SUBSCRIBE, "")
            self.sockets['calibration_sub'].setsockopt(zmq.RCVTIMEO, 100)

            self.sockets['ai_sub'] = self.context.socket(zmq.SUB)
            self.sockets['ai_sub'].connect(f"tcp://localhost:{self.config['ports']['ai_pub']}")
            self.sockets['ai_sub'].setsockopt_string(zmq.SUBSCRIBE, "")
            self.sockets['ai_sub'].setsockopt(zmq.RCVTIMEO, 100)

            # REQ soketleri (modül kontrolü için)
            self.sockets['sdr_req'] = self.context.socket(zmq.REQ)
            self.sockets['sdr_req'].connect(f"tcp://localhost:{self.config['ports']['sdr_control']}")
            self.sockets['sdr_req'].setsockopt(zmq.RCVTIMEO, 1000)
            self.sockets['sdr_req'].setsockopt(zmq.LINGER, 0)

            self.sockets['calibration_req'] = self.context.socket(zmq.REQ)
            self.sockets['calibration_req'].connect(f"tcp://localhost:{self.config['ports']['calibration_control']}")
            self.sockets['calibration_req'].setsockopt(zmq.RCVTIMEO, 1000)
            self.sockets['calibration_req'].setsockopt(zmq.LINGER, 0)

            self.sockets['ai_req'] = self.context.socket(zmq.REQ)
            self.sockets['ai_req'].connect(f"tcp://localhost:{self.config['ports']['ai_control']}")
            self.sockets['ai_req'].setsockopt(zmq.RCVTIMEO, 1000)
            self.sockets['ai_req'].setsockopt(zmq.LINGER, 0)

            self.sockets['radar_req'] = self.context.socket(zmq.REQ)
            self.sockets['radar_req'].connect(f"tcp://localhost:{self.config['ports']['radar_control']}")
            self.sockets['radar_req'].setsockopt(zmq.RCVTIMEO, 1000)
            self.sockets['radar_req'].setsockopt(zmq.LINGER, 0)

            self.sockets['3d_req'] = self.context.socket(zmq.REQ)
            self.sockets['3d_req'].connect(f"tcp://localhost:{self.config['ports']['3d_control']}")
            self.sockets['3d_req'].setsockopt(zmq.RCVTIMEO, 1000)
            self.sockets['3d_req'].setsockopt(zmq.LINGER, 0)

            logger.info("ZMQ soketleri oluşturuldu ve yapılandırıldı")

        except Exception as e:
            logger.error(f"ZMQ soket hatası: {str(e)}")
            raise

    def _start_monitoring_threads(self):
        """İzleme thread'lerini başlatır."""
        threads = {
            'sdr_monitor': self._sdr_monitor_worker,
            'calibration_monitor': self._calibration_monitor_worker,
            'ai_monitor': self._ai_monitor_worker,
            'control_monitor': self._control_monitor_worker,
            'module_monitor': self._module_monitor_worker,
            'command_processor': self._command_processor_worker
        }

        for name, target in threads.items():
            self.threads[name] = threading.Thread(
                target=target,
                name=name,
                daemon=True
            )
            self.threads[name].start()

        logger.info("İzleme thread'leri başlatıldı")

    def _sdr_monitor_worker(self):
        """SDR verilerini izler."""
        while self.running.is_set():
            try:
                message = self.sockets['sdr_sub'].recv_json()
                self.sdr_data = message
                self.data_queue.put(('sdr_data', message))

                if self.debug_mode:
                    logger.debug(f"SDR verisi alındı: {message.keys()}")

            except zmq.Again:
                pass
            except Exception as e:
                logger.error(f"SDR izleme hatası: {str(e)}")
                time.sleep(1)

    def _calibration_monitor_worker(self):
        """Kalibrasyon verilerini izler."""
        while self.running.is_set():
            try:
                message = self.sockets['calibration_sub'].recv_json()
                self.calibration_data = message
                self.data_queue.put(('calibration_data', message))

                if 'ground_type' in message:
                    self.system_status['ground_type'] = message['ground_type']
                    self.system_status['last_calibration'] = datetime.now()

                if self.debug_mode:
                    logger.debug(f"Kalibrasyon verisi alındı: {message}")

            except zmq.Again:
                pass
            except Exception as e:
                logger.error(f"Kalibrasyon izleme hatası: {str(e)}")
                time.sleep(1)

    def _ai_monitor_worker(self):
        """AI verilerini izler."""
        while self.running.is_set():
            try:
                message = self.sockets['ai_sub'].recv_json()
                self.ai_data = message
                self.detections = message.get('detections', [])
                self.data_queue.put(('ai_data', message))

                self.system_status['detection_count'] += len(self.detections)

                if self.debug_mode:
                    logger.debug(f"AI verisi alındı: {len(self.detections)} tespit")

            except zmq.Again:
                pass
            except Exception as e:
                logger.error(f"AI izleme hatası: {str(e)}")
                time.sleep(1)

    def _control_monitor_worker(self):
        """Kontrol komutlarını izler."""
        while self.running.is_set():
            try:
                message = self.sockets['control_rep'].recv_json()
                response = self._handle_command(message)
                self.sockets['control_rep'].send_json(response)

                if self.debug_mode:
                    logger.debug(f"Kontrol komutu işlendi: {message} -> {response}")

            except zmq.Again:
                pass
            except Exception as e:
                logger.error(f"Kontrol izleme hatası: {str(e)}")
                self.sockets['control_rep'].send_json({"status": "error", "message": str(e)})
                time.sleep(1)

    def _module_monitor_worker(self):
        """Modül durumlarını izler ve yönetir."""
        while self.running.is_set():
            try:
                for module_name, module_data in self.modules.items():
                    process = module_data['process']

                    # Süreç durumunu kontrol et
                    if process and process.poll() is None:
                        if module_data['status'] != ModuleStatus.RUNNING:
                            module_data['status'] = ModuleStatus.RUNNING
                            logger.info(f"{module_name} modülü çalışıyor")
                    else:
                        if module_data['status'] == ModuleStatus.RUNNING:
                            module_data['status'] = ModuleStatus.ERROR
                            logger.warning(f"{module_name} modülü beklenmedik şekilde durdu")

                            # Otomatik yeniden başlatma
                            if module_data['restart_count'] < self.config.get('module_restart_limit', 3):
                                delay = self.config.get('module_restart_delay', 5)
                                logger.info(f"{module_name} modülü {delay} saniye sonra yeniden başlatılacak")
                                module_data['status'] = ModuleStatus.RESTARTING
                                module_data['last_restart'] = datetime.now()
                                threading.Timer(delay, self.start_module, args=(module_name,)).start()
                                module_data['restart_count'] += 1

                time.sleep(1)

            except Exception as e:
                logger.error(f"Modül izleme hatası: {str(e)}")
                time.sleep(5)

    def _command_processor_worker(self):
        """Komutları işler."""
        while self.running.is_set():
            try:
                command, args, kwargs = self.command_queue.get(timeout=1)

                if command == 'start_module':
                    self.start_module(*args, **kwargs)
                elif command == 'stop_module':
                    self.stop_module(*args, **kwargs)
                elif command == 'restart_module':
                    self.stop_module(*args, **kwargs)
                    self.start_module(*args, **kwargs)

                self.command_queue.task_done()

            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Komut işleme hatası: {str(e)}")

    def _handle_command(self, message):
        """Gelen komutları işler."""
        try:
            command = message.get('command', '')
            module = message.get('module', '')
            params = message.get('params', {})

            if command == 'status':
                return {
                    'status': 'ok',
                    'system': self.system_status,
                    'modules': {name: data['status'].name for name, data in self.modules.items()},
                    'detections': self.detections
                }

            elif command == 'start_module':
                if module in self.modules:
                    self.command_queue.put(('start_module', [module], params))
                    return {'status': 'ok', 'message': f'{module} başlatma isteği alındı'}

            elif command == 'stop_module':
                if module in self.modules:
                    self.command_queue.put(('stop_module', [module], params))
                    return {'status': 'ok', 'message': f'{module} durdurma isteği alındı'}

            elif command == 'restart_module':
                if module in self.modules:
                    self.command_queue.put(('restart_module', [module], params))
                    return {'status': 'ok', 'message': f'{module} yeniden başlatma isteği alındı'}

            return {'status': 'error', 'message': 'Geçersiz komut'}

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def _start_ui(self):
        """Kullanıcı arayüzünü başlatır."""
        try:
            # Önce ana UI modülünü dene
            from ui.main_window import MainWindow
            self.ui = MainWindow(self)
            self.ui.start()
            logger.info("Ana UI başlatıldı")

        except ImportError:
            # Fallback olarak basit Tkinter UI
            logger.warning("Ana UI modülü bulunamadı, basit arayüz kullanılacak")
            self._start_fallback_ui()

    def _start_fallback_ui(self):
        """Basit bir fallback arayüzü başlatır."""
        try:
            import tkinter as tk
            from tkinter import ttk, scrolledtext

            class FallbackUI:
                def __init__(self, control_center):
                    self.control_center = control_center
                    self.root = tk.Tk()
                    self.root.title("SDR Yeraltı Tespit Sistemi - Basit Arayüz")

                    # Durum bilgisi
                    self.status_frame = ttk.LabelFrame(self.root, text="Sistem Durumu")
                    self.status_frame.pack(padx=10, pady=5, fill=tk.X)

                    self.status_text = scrolledtext.ScrolledText(
                        self.status_frame,
                        height=10,
                        state='disabled'
                    )
                    self.status_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

                    # Kontrol butonları
                    self.control_frame = ttk.LabelFrame(self.root, text="Kontrol")
                    self.control_frame.pack(padx=10, pady=5, fill=tk.X)

                    ttk.Button(
                        self.control_frame,
                        text="Tüm Modülleri Başlat",
                        command=self.start_all
                    ).pack(side=tk.LEFT, padx=5, pady=5)

                    ttk.Button(
                        self.control_frame,
                        text="Tüm Modülleri Durdur",
                        command=self.stop_all
                    ).pack(side=tk.LEFT, padx=5, pady=5)

                    # Modül listesi
                    self.module_frame = ttk.LabelFrame(self.root, text="Modüller")
                    self.module_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

                    self.module_buttons = {}
                    for module in self.control_center.modules.keys():
                        frame = ttk.Frame(self.module_frame)
                        frame.pack(fill=tk.X, padx=5, pady=2)

                        ttk.Label(frame, text=module, width=15).pack(side=tk.LEFT)

                        self.module_buttons[module] = {
                            'status': ttk.Label(frame, text="DURDURULDU", width=15),
                            'start': ttk.Button(frame, text="Başlat",
                                                command=lambda m=module: self.start_module(m)),
                            'stop': ttk.Button(frame, text="Durdur",
                                               command=lambda m=module: self.stop_module(m))
                        }

                        self.module_buttons[module]['status'].pack(side=tk.LEFT, padx=5)
                        self.module_buttons[module]['start'].pack(side=tk.LEFT, padx=2)
                        self.module_buttons[module]['stop'].pack(side=tk.LEFT, padx=2)

                    # Güncelleme timer'ı
                    self.update_ui()

                def start_all(self):
                    self.control_center.start_all_modules()

                def stop_all(self):
                    self.control_center.stop_all_modules()

                def start_module(self, module):
                    self.control_center.start_module(module)

                def stop_module(self, module):
                    self.control_center.stop_module(module)

                def update_ui(self):
                    try:
                        # Durum metnini güncelle
                        self.status_text.config(state='normal')
                        self.status_text.delete(1.0, tk.END)

                        status = {
                            'running': 'ÇALIŞIYOR' if self.control_center.system_status['running'] else 'DURDURULDU',
                            'uptime': str(datetime.now() - self.control_center.system_status['start_time'])
                            if self.control_center.system_status['start_time'] else 'N/A',
                            'detections': self.control_center.system_status['detection_count'],
                            'ground_type': self.control_center.system_status['ground_type'] or 'Bilinmiyor'
                        }

                        status_text = "\n".join(f"{k}: {v}" for k, v in status.items())
                        self.status_text.insert(tk.END, status_text)
                        self.status_text.config(state='disabled')

                        # Modül durumlarını güncelle
                        for module, data in self.control_center.modules.items():
                            status = data['status'].name
                            color = 'green' if data['status'] == ModuleStatus.RUNNING else 'red'
                            self.module_buttons[module]['status'].config(
                                text=status,
                                foreground=color
                            )

                    except Exception as e:
                        logger.error(f"UI güncelleme hatası: {str(e)}")

                    # Her 1 saniyede bir güncelle
                    self.root.after(1000, self.update_ui)

                def start(self):
                    self.root.mainloop()

                def stop(self):
                    self.root.quit()

            self.ui = FallbackUI(self)
            self.ui.start()
            logger.info("Basit arayüz başlatıldı")

        except Exception as e:
            logger.error(f"UI başlatma hatası: {str(e)}")
            self.headless = True

    def _main_loop(self):
        """Ana kontrol döngüsü."""
        try:
            while self.running.is_set():
                # Sistem durumunu güncelle
                self.system_status['scan_count'] += 1

                # Veri kuyruğunu işle
                self._process_data_queue()

                # UI güncellemesi
                if self.ui:
                    pass  # UI kendi güncelleme mekanizmasını kullanıyor

                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Kullanıcı tarafından durduruldu")
        except Exception as e:
            logger.error(f"Ana döngü hatası: {str(e)}")
        finally:
            self.stop()

    def _process_data_queue(self):
        """Veri kuyruğundaki mesajları işler."""
        try:
            while not self.data_queue.empty():
                data_type, data = self.data_queue.get_nowait()

                if data_type == 'sdr_data':
                    self._handle_sdr_data(data)
                elif data_type == 'calibration_data':
                    self._handle_calibration_data(data)
                elif data_type == 'ai_data':
                    self._handle_ai_data(data)

                self.data_queue.task_done()

        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Veri işleme hatası: {str(e)}")

    def _handle_sdr_data(self, data):
        """SDR verilerini işler."""
        # Gelen verileri işle
        pass

    def _handle_calibration_data(self, data):
        """Kalibrasyon verilerini işler."""
        # Gelen verileri işle
        pass

    def _handle_ai_data(self, data):
        """AI verilerini işler."""
        # Gelen verileri işle
        pass

    def start_module(self, module_name):
        """Belirtilen modülü başlatır."""
        if module_name not in self.modules:
            logger.error(f"Geçersiz modül adı: {module_name}")
            return False

        if self.modules[module_name]['status'] in [ModuleStatus.RUNNING, ModuleStatus.STARTING]:
            logger.warning(f"{module_name} modülü zaten çalışıyor")
            return True

        try:
            self.modules[module_name]['status'] = ModuleStatus.STARTING
            logger.info(f"{module_name} modülü başlatılıyor...")

            # Modül bağımlılıklarını başlat
            for dep in self._get_module_dependencies(module_name):
                if self.modules[dep]['status'] != ModuleStatus.RUNNING:
                    self.start_module(dep)

            # Modül komutunu oluştur
            cmd = self._build_module_command(module_name)

            # Modülü başlat
            self.modules[module_name]['process'] = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                start_new_session=True
            )

            self.modules[module_name]['status'] = ModuleStatus.RUNNING
            self.modules[module_name]['restart_count'] = 0
            logger.info(f"{module_name} modülü başlatıldı (PID: {self.modules[module_name]['process'].pid})")

            return True

        except Exception as e:
            self.modules[module_name]['status'] = ModuleStatus.ERROR
            logger.error(f"{module_name} modülü başlatma hatası: {str(e)}")
            return False

    def stop_module(self, module_name):
        """Belirtilen modülü durdurur."""
        if module_name not in self.modules:
            logger.error(f"Geçersiz modül adı: {module_name}")
            return False

        if self.modules[module_name]['status'] in [ModuleStatus.STOPPED, ModuleStatus.STOPPING]:
            logger.warning(f"{module_name} modülü zaten durdurulmuş")
            return True

        try:
            self.modules[module_name]['status'] = ModuleStatus.STOPPING
            logger.info(f"{module_name} modülü durduruluyor...")

            process = self.modules[module_name]['process']
            if process:
                # Önce SIGTERM gönder
                process.terminate()

                try:
                    # 2 saniye bekleyerek düzgün kapanmasını sağla
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # Zorla kapat
                    process.kill()
                    process.wait()

                self.modules[module_name]['process'] = None

            self.modules[module_name]['status'] = ModuleStatus.STOPPED
            logger.info(f"{module_name} modülü durduruldu")

            return True

        except Exception as e:
            self.modules[module_name]['status'] = ModuleStatus.ERROR
            logger.error(f"{module_name} modülü durdurma hatası: {str(e)}")
            return False

    def start_all_modules(self):
        """Tüm modülleri başlatır."""
        logger.info("Tüm modüller başlatılıyor...")

        # Bağımlılık sırasına göre başlat
        startup_order = ['sdr_receiver', 'calibration', 'ai_classifier', 'radar_ui', 'depth_3d_view', 'logger']

        for module in startup_order:
            if module in self.modules:
                self.start_module(module)
                time.sleep(1)  # Modüller arasında kısa bekleme

    def stop_all_modules(self):
        """Tüm modülleri durdurur."""
        logger.info("Tüm modüller durduruluyor...")

        # Ters sırada durdur
        shutdown_order = ['depth_3d_view', 'radar_ui', 'ai_classifier', 'calibration', 'sdr_receiver', 'logger']

        for module in shutdown_order:
            if module in self.modules:
                self.stop_module(module)
                time.sleep(0.5)  # Modüller arasında kısa bekleme

    def _get_module_dependencies(self, module_name):
        """Modül bağımlılıklarını döndürür."""
        dependencies = {
            'sdr_receiver': [],
            'calibration': ['sdr_receiver'],
            'ai_classifier': ['sdr_receiver', 'calibration'],
            'radar_ui': ['sdr_receiver', 'ai_classifier'],
            'depth_3d_view': ['ai_classifier', 'calibration'],
            'logger': []
        }

        return dependencies.get(module_name, [])

    def _build_module_command(self, module_name):
        """Modül komut satırı argümanlarını oluşturur."""
        base_cmd = [sys.executable, f"modules/{module_name}.py"]

        # Modüle özel parametreler
        params = {
            'sdr_receiver': [
                f"--center_freq={self.config['sdr']['center_freq']}",
                f"--sample_rate={self.config['sdr']['sample_rate']}",
                f"--gain={self.config['sdr']['gain']}",
                f"--pub_port={self.config['ports']['sdr_pub']}",
                f"--control_port={self.config['ports']['sdr_control']}"
            ],
            'calibration': [
                f"--pub_port={self.config['ports']['calibration_pub']}",
                f"--control_port={self.config['ports']['calibration_control']}",
                f"--sub_port={self.config['ports']['sdr_pub']}",
                f"--interval={self.config['calibration']['interval']}"
            ],
            'ai_classifier': [
                f"--pub_port={self.config['ports']['ai_pub']}",
                f"--control_port={self.config['ports']['ai_control']}",
                f"--sdr_sub_port={self.config['ports']['sdr_pub']}",
                f"--calibration_sub_port={self.config['ports']['calibration_pub']}",
                f"--model_path={self.config['ai']['model_path']}",
                f"--confidence_threshold={self.config['ai']['confidence_threshold']}"
            ],
            'radar_ui': [
                f"--control_port={self.config['ports']['radar_control']}",
                f"--ai_sub_port={self.config['ports']['ai_pub']}",
                f"--fullscreen={self.config['ui']['fullscreen']}",
                f"--dark_mode={self.config['ui']['dark_mode']}",
                f"--update_interval={self.config['ui']['update_interval']}",
                f"--max_range={self.config['ui']['max_range']}"
            ],
            'depth_3d_view': [
                f"--control_port={self.config['ports']['3d_control']}",
                f"--ai_port={self.config['ports']['ai_pub']}",
                f"--calibration_port={self.config['ports']['calibration_pub']}",
                f"--backend={self.config['ui']['3d_backend']}",
                f"--max_depth={self.config['ui']['max_depth']}",
                f"--resolution={self.config['ui']['resolution']}"
            ],
            'logger': [
                f"--log_level={self.config['logger']['log_level']}",
                f"--log_dir={self.config['logger']['log_dir']}"
            ]
        }

        # Test modu parametresi
        if self.test_mode:
            params[module_name].append("--test_mode")

        return base_cmd + params.get(module_name, [])


def main():
    """Ana uygulama giriş noktası."""
    parser = argparse.ArgumentParser(
        description='SDR Tabanlı Yeraltı Tespit Sistemi Kontrol Merkezi',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        default="config.json",
        help='Yapılandırma dosyası yolu'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Hata ayıklama modu'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Test modu'
    )
    parser.add_argument(
        '--auto-start',
        action='store_true',
        help='Modülleri otomatik başlat'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='UI olmadan çalıştır'
    )

    args = parser.parse_args()

    try:
        # Kontrol merkezini oluştur ve başlat
        control_center = ControlCenter(args)

        # Çıkış sinyallerini yakala
        signal.signal(signal.SIGINT, lambda s, f: control_center.stop())
        signal.signal(signal.SIGTERM, lambda s, f: control_center.stop())

        # Başlat
        if control_center.start():
            logger.info("Kontrol merkezi başarıyla başlatıldı")

            # Ana döngüyü çalıştır
            while control_center.running.is_set():
                time.sleep(1)

        else:
            logger.error("Kontrol merkezi başlatılamadı")
            return 1

    except Exception as e:
        logger.error(f"Kritik hata: {str(e)}\n{traceback.format_exc()}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())