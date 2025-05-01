#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
monitoring.py - System Performance and Resource Monitoring Module

Version: 2.0
Author: AI Assistant (Manus)
Last Updated: 2025-04-30

Mesopotamia GPR sisteminin performansını ve kaynak kullanımını izler.
CPU, bellek gibi genel sistem metriklerinin yanı sıra modül bazlı
işlem süreleri ve ZMQ kuyruk durumları gibi özel metrikleri de takip eder.
Sonuçları loglar ve ZMQ üzerinden yayınlar.
"""

import time
import psutil
import numpy as np
import zmq
import json
import logging
import logging.handlers
from threading import Thread, Event, Lock
from collections import deque
from typing import Dict, Any, Optional, List
import argparse

# Yerel modüller
from config import PORT_MONITORING_DATA, PORT_MONITORING_CONTROL

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s",
    handlers=[
        logging.FileHandler("monitoring.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("monitoring")

class PerformanceMonitor:
    """Detaylı performans ve kaynak metriklerini toplar ve yönetir."""

    def __init__(self, history_size: int = 100):
        """
        Args:
            history_size (int): Her metrik için tutulacak geçmiş veri sayısı.
        """
        self.history_size = history_size
        self.metrics: Dict[str, deque] = {
            "timestamp": deque(maxlen=history_size),
            "cpu_percent": deque(maxlen=history_size), # İşlemin CPU kullanımı
            "memory_percent": deque(maxlen=history_size), # İşlemin Bellek kullanımı
            # Modül bazlı metrikler (dinamik olarak eklenecek)
            # Örn: "preprocessing_latency_ms": deque(maxlen=history_size),
            # Örn: "ai_queue_size": deque(maxlen=history_size),
        }
        self.start_time = time.time()
        self.lock = Lock()
        try:
            self.process = psutil.Process() # Mevcut işlem için
        except psutil.NoSuchProcess:
            logger.error("Mevcut işlem bulunamadı! Metrikler toplanamayabilir.")
            self.process = None

    def update_system_metrics(self):
        """Genel sistem metriklerini günceller."""
        if not self.process:
            return
            
        with self.lock:
            timestamp = time.time()
            self.metrics["timestamp"].append(timestamp)
            try:
                # Sadece mevcut işlemin CPU kullanımı (tüm çekirdeklere göre normalize edilmiş)
                cpu_usage = self.process.cpu_percent() / psutil.cpu_count()
                self.metrics["cpu_percent"].append(cpu_usage)
            except Exception as e:
                 logger.warning(f"CPU metriği alınamadı: {e}")
                 self.metrics["cpu_percent"].append(0.0)
                 
            try:
                # Sadece mevcut işlemin Bellek kullanımı yüzdesi
                mem_usage = self.process.memory_percent()
                self.metrics["memory_percent"].append(mem_usage)
            except Exception as e:
                 logger.warning(f"Bellek metriği alınamadı: {e}")
                 self.metrics["memory_percent"].append(0.0)

    def record_metric(self, name: str, value: float, timestamp: Optional[float] = None):
        """Özel bir metrik kaydeder (örn. işlem süresi, kuyruk boyutu)."""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=self.history_size)
                logger.info(f"Yeni metrik eklendi: {name}")
            self.metrics[name].append(value)
            # Zaman damgası yönetimi şimdilik basitleştirildi, sadece değer saklanıyor.

    def get_summary(self) -> Dict[str, Any]:
        """Son metrik durumunun özetini döndürür."""
        summary = {"uptime_s": time.time() - self.start_time}
        with self.lock:
            for name, values in self.metrics.items():
                if values:
                    if name == "timestamp":
                        summary["last_update_ts"] = values[-1]
                    else:
                        try:
                            last_val = values[-1]
                            summary[f"{name}_last"] = last_val
                            if len(values) > 1 and isinstance(last_val, (int, float)):
                                summary[f"{name}_avg"] = np.mean(values)
                                summary[f"{name}_max"] = np.max(values)
                                summary[f"{name}_min"] = np.min(values)
                        except (TypeError, ValueError, IndexError):
                            summary[f"{name}_last"] = values[-1] # Sadece son değeri ekle
        return summary

    def get_history(self, metric_names: Optional[List[str]] = None) -> Dict[str, list]:
         """Belirtilen veya tüm metriklerin geçmişini döndürür."""
         with self.lock:
             if metric_names:
                 return {name: list(self.metrics.get(name, [])) for name in metric_names if name in self.metrics}
             else:
                 # Deque\leri listeye çevir
                 return {name: list(values) for name, values in self.metrics.items()}


class MonitoringService:
    """Periyodik olarak metrikleri toplayan ve yayınlayan servis."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.interval_s = config.get("update_interval_s", 5.0)
        self.history_size = config.get("history_size", 120)  # 10 dakika (5sn aralıkla)
        self.monitor = PerformanceMonitor(self.history_size)

        self.context = zmq.Context()
        # Metrik Yayın Soketi
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind(f'tcp://*:{config.get("publish_port", PORT_MONITORING_DATA)}')

        # Kontrol Soketi
        self.control_socket = self.context.socket(zmq.REP)
        self.control_socket.bind(f'tcp://*:{config.get("control_port", PORT_MONITORING_CONTROL)}')

        self.running = Event()
        self.worker_thread = None
        self.control_thread = None
        logger.info("Monitoring Service başlatıldı.")

    def start(self):
        """İzleme servisini başlatır."""
        if not self.running.is_set():
            self.running.set()
            self.worker_thread = Thread(target=self._worker, daemon=True, name="MonitorWorker")
            self.control_thread = Thread(target=self._control_worker, daemon=True, name="MonitorControl")
            self.worker_thread.start()
            self.control_thread.start()
            logger.info("İzleme ve kontrol thread\leri başlatıldı.")
        else:
            logger.warning("Monitoring Service zaten çalışıyor.")

    def stop(self):
        """İzleme servisini durdurur."""
        if self.running.is_set():
            self.running.clear()
            logger.info("Monitoring Service durduruluyor...")

            # Soketleri kapat
            self.pub_socket.close()
            self.control_socket.close()
            # self.context.term() # Context paylaşımlı olabilir

            # Thread\leri bekle
            if self.worker_thread:
                self.worker_thread.join(timeout=2.0)
            if self.control_thread:
                self.control_thread.join(timeout=2.0)

            logger.info("Monitoring Service durduruldu.")
        else:
            logger.warning("Monitoring Service zaten durdurulmuş.")

    def _worker(self):
        """Periyodik olarak metrikleri toplayan ve yayınlayan worker."""
        logger.info("İzleme worker\ı çalışmaya başladı.")
        while self.running.is_set():
            start_time = time.time()
            try:
                # Sistem metriklerini güncelle
                self.monitor.update_system_metrics()
                
                # TODO: Diğer modüllerden özel metrikleri al (ZMQ REQ/REP ile?)
                # self._collect_module_metrics()
                
                # Metrik özetini yayınla
                summary = self.monitor.get_summary()
                self.pub_socket.send_json(summary)
                # logger.debug(f"Metrik özeti yayınlandı: {summary}")
                
            except Exception as e:
                logger.error(f"İzleme worker hatası: {e}", exc_info=True)

            # Döngü süresini hesaba katarak bekle
            elapsed_time = time.time() - start_time
            sleep_time = max(0, self.interval_s - elapsed_time)
            if self.running.is_set():
                 time.sleep(sleep_time)
                 
        logger.info("İzleme worker\ı durdu.")

    # def _collect_module_metrics(self):
    #     """Diğer modüllerden ZMQ üzerinden metrikleri toplar (opsiyonel)."""
    #     # Örnek: Preprocessing modülünden kuyruk boyutunu al
    #     # try:
    #     #     req_socket = self.context.socket(zmq.REQ)
    #     #     req_socket.connect(f"tcp://localhost:{PORT_PREPROCESSING_CONTROL}")
    #     #     req_socket.setsockopt(zmq.LINGER, 0)
    #     #     req_socket.setsockopt(zmq.RCVTIMEO, 500)
    #     #     req_socket.send_json({"command": "get_status"})
    #     #     response = req_socket.recv_json()
    #     #     if response.get("status") == "ok":
    #     #         queue_size = response.get("data", {}).get("queue_size", 0)
    #     #         self.monitor.record_metric("preprocessing_queue_size", queue_size)
    #     #     req_socket.close()
    #     # except Exception as e:
    #     #     logger.warning(f"Preprocessing metrikleri alınamadı: {e}")
    #     pass

    def _control_worker(self):
        """Kontrol komutlarını dinler ve işler."""
        logger.info("İzleme kontrol thread\i başladı.")
        while self.running.is_set():
            try:
                message = self.control_socket.recv_json()
                response = self._handle_control_command(message)
                self.control_socket.send_json(response)
            except zmq.ZMQError as e:
                 if e.errno == zmq.ETERM: break
                 logger.error(f"ZMQ İzleme kontrol hatası: {e}")
                 time.sleep(1)
            except Exception as e:
                logger.error(f"İzleme kontrol hatası: {str(e)}")
                try:
                    self.control_socket.send_json({"status": "error", "message": str(e)})
                except zmq.ZMQError:
                    pass
                time.sleep(1)
        logger.info("İzleme kontrol thread\i durdu.")

    def _handle_control_command(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Gelen kontrol komutlarını işler."""
        command = message.get("command")
        logger.info(f"Kontrol komutu alındı: {command}")

        if command == "get_summary":
            summary = self.monitor.get_summary()
            return {"status": "ok", "data": summary}
        elif command == "get_history":
            metric_names = message.get("metrics") # İstenen metrikler (opsiyonel)
            history = self.monitor.get_history(metric_names)
            return {"status": "ok", "data": history}
        elif command == "set_interval":
            new_interval = message.get("value")
            if new_interval is not None and new_interval > 0:
                self.interval_s = float(new_interval)
                logger.info(f"İzleme aralığı {self.interval_s} saniye olarak ayarlandı.")
                return {"status": "ok"}
            else:
                return {"status": "error", "message": "Geçersiz aralık değeri."}
        else:
            return {"status": "error", "message": "Bilinmeyen komut"}

# Komut satırı argümanları
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="System Monitoring Service")
    parser.add_argument("--config", type=str, default="system_config.json", help="Path to JSON configuration file")
    # Diğer argümanlar config dosyasından alınacak

    args = parser.parse_args()

    # Load config from file
    config = {}
    try:
        with open(args.config, "r") as f:
            config_from_file = json.load(f)
            config = config_from_file.get("module_configs", {}).get("monitoring", {})
            logger.info(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}. Using defaults.")
    except json.JSONDecodeError:
         logger.error(f"Error decoding JSON from {args.config}. Using defaults.")

    # Monitoring servisini başlat
    service = MonitoringService(config)
    service.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt alındı, sistem durduruluyor.")
    finally:
        service.stop()

