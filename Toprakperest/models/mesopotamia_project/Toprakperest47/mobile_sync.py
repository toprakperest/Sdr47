#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mobile_sync.py - Mobil Cihaz Senkronizasyon Modülü (Placeholder)

Bu modül, yeraltı tespit sisteminin verilerini (tespitler, durum, harita vb.)
mobil bir uygulama ile senkronize etmek için bir yapı sağlar.

Gerçek implementasyon, seçilen iletişim protokolüne (Bluetooth, WiFi, USB)
ve mobil uygulamanın API'sine bağlı olacaktır.
"""

import time
import zmq
import json
import logging
from threading import Thread, Event
from typing import Dict, Any

# Yerel modüller
from config import PORT_MOBILE_SYNC, PORT_AI_OUTPUT, PORT_CALIBRATION_RESULT, PORT_UI_CONTROL

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format=\'%(asctime)s - %(name)s - %(levelname)s - %(message)s\',
    handlers=[
        logging.FileHandler("mobile_sync.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mobile_sync")


class MobileSync:
    """Mobil cihazlarla veri senkronizasyonu sağlar (Placeholder)."""

    def __init__(self, config: Dict[str, Any]):
        """
        MobileSync modülünü başlatır.

        Args:
            config (Dict[str, Any]): Modül yapılandırma parametreleri.
        """
        self.config = config
        self.enabled = config.get("enabled", False)
        self.sync_interval = config.get("sync_interval_sec", 1.0)
        self.sync_port = config.get("sync_port", PORT_MOBILE_SYNC)
        self.communication_method = config.get("method", "wifi") # wifi, bluetooth, usb

        # ZMQ İletişim (Diğer modüllerden veri almak için)
        self.context = zmq.Context()

        # AI Tespitleri
        self.ai_socket = self.context.socket(zmq.SUB)
        self.ai_socket.connect(f"tcp://localhost:{config.get('ai_input_port', PORT_AI_OUTPUT)}")
        self.ai_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.ai_socket.setsockopt(zmq.CONFLATE, 1)

        # Kalibrasyon/Durum Bilgisi
        self.calibration_socket = self.context.socket(zmq.SUB)
        self.calibration_socket.connect(f"tcp://localhost:{config.get('calibration_input_port', PORT_CALIBRATION_RESULT)}")
        self.calibration_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.calibration_socket.setsockopt(zmq.CONFLATE, 1)

        # TODO: UI/Control Center'dan genel sistem durumu almak için ek soket?

        # Mobil Cihaza Veri Gönderme Soketi (Placeholder)
        # Gerçek implementasyon ZMQ olmayabilir (örn. BluetoothSocket, TCPSocket)
        self.mobile_socket = None
        if self.enabled:
            # Örnek: WiFi üzerinden TCP soketi
            if self.communication_method == "wifi":
                try:
                    # Bu kısım, mobil uygulamanın beklediği protokole göre değişir
                    self.mobile_socket = self.context.socket(zmq.PUB) # Veya PUSH/PAIR
                    # Mobil cihazın IP'si ve portu bilinmeli veya keşfedilmeli
                    # self.mobile_socket.connect("tcp://<mobile_ip>:<mobile_port>")
                    logger.info(f"Mobil senkronizasyon (WiFi) için soket hazırlanıyor (Port: {self.sync_port}) - Bağlantı bekleniyor...")
                    # Şimdilik sadece bind yapalım, bağlantı dışarıdan kurulabilir
                    self.mobile_socket.bind(f"tcp://*:{self.sync_port}")
                except Exception as e:
                    logger.error(f"Mobil WiFi soketi oluşturulamadı: {e}")
                    self.enabled = False
            elif self.communication_method == "bluetooth":
                logger.warning("Bluetooth senkronizasyonu henüz implemente edilmedi.")
                # Bluetooth soket kütüphanesi (örn. PyBluez) gerekir
                self.enabled = False
            else:
                logger.warning(f"Desteklenmeyen iletişim metodu: {self.communication_method}")
                self.enabled = False

        # Dahili durum
        self.running = Event()
        self.worker_thread = None
        self.last_sync_data = {}

        if not self.enabled:
            logger.info("Mobil senkronizasyon devre dışı.")
        else:
            logger.info(f"Mobil senkronizasyon aktif ({self.communication_method}). Interval: {self.sync_interval}s")

    def start(self):
        """Senkronizasyon işlemini başlatır."""
        if self.enabled and not self.running.is_set():
            self.running.set()
            self.worker_thread = Thread(target=self._sync_worker, daemon=True, name="MobileSyncWorker")
            self.worker_thread.start()
            logger.info("Mobil senkronizasyon thread'i başlatıldı.")
        elif not self.enabled:
            logger.debug("Mobil senkronizasyon başlatılmadı (devre dışı).")
        else:
            logger.warning("Mobil senkronizasyon zaten çalışıyor.")

    def stop(self):
        """Senkronizasyon işlemini durdurur."""
        if self.running.is_set():
            self.running.clear()
            logger.info("Mobil senkronizasyon durduruluyor...")

            # Soketleri kapat
            self.ai_socket.close()
            self.calibration_socket.close()
            if self.mobile_socket:
                self.mobile_socket.close()
            self.context.term()

            if self.worker_thread:
                self.worker_thread.join(timeout=2.0)

            logger.info("Mobil senkronizasyon durduruldu.")
        elif self.enabled:
            logger.warning("Mobil senkronizasyon zaten durdurulmuş.")

    def _sync_worker(self):
        """Periyodik olarak veri toplayıp mobil cihaza gönderen ana döngü."""
        logger.info("Mobil senkronizasyon worker başladı.")
        while self.running.is_set():
            start_time = time.time()
            try:
                # Verileri topla
                current_data = self._collect_data()

                # Veri değiştiyse veya periyodik olarak gönder
                # if current_data != self.last_sync_data: # Değişiklik kontrolü
                self._send_data_to_mobile(current_data)
                self.last_sync_data = current_data

            except Exception as e:
                logger.error(f"Mobil senkronizasyon hatası: {str(e)}", exc_info=True)

            # Senkronizasyon aralığını bekle
            elapsed_time = time.time() - start_time
            sleep_time = max(0, self.sync_interval - elapsed_time)
            time.sleep(sleep_time)

        logger.info("Mobil senkronizasyon worker durdu.")

    def _collect_data(self) -> Dict[str, Any]:
        """Diğer modüllerden senkronize edilecek verileri toplar."""
        data = {
            "timestamp": time.time(),
            "system_status": "SCANNING", # TODO: Get actual status
            "detections": [],
            "ground_info": {},
            "calibration_status": "UNKNOWN" # TODO: Get actual status
        }

        # AI Tespitleri
        try:
            ai_msg = self.ai_socket.recv_json(flags=zmq.NOBLOCK)
            data["detections"] = ai_msg.get("detections", [])
        except zmq.Again:
            pass # Veri yok
        except json.JSONDecodeError:
            logger.warning("AI soketinden geçersiz JSON alındı.")

        # Kalibrasyon/Zemin Bilgisi
        try:
            cal_msg = self.calibration_socket.recv_json(flags=zmq.NOBLOCK)
            data["ground_info"] = cal_msg.get("ground_params", {})
            data["calibration_status"] = "COMPLETED" if cal_msg.get("calibration_complete") else "IN_PROGRESS"
        except zmq.Again:
            pass
        except json.JSONDecodeError:
            logger.warning("Kalibrasyon soketinden geçersiz JSON alındı.")

        # TODO: Diğer durum bilgilerini ekle (batarya, GPS vb.)

        return data

    def _send_data_to_mobile(self, data: Dict[str, Any]):
        """Toplanan veriyi mobil cihaza gönderir (Placeholder)."""
        if not self.mobile_socket:
            return

        try:
            # Veriyi JSON formatına çevir
            message_str = json.dumps(data)
            # Gönder (Protokole göre değişir)
            self.mobile_socket.send_string(message_str)
            logger.debug(f"Mobil cihaza veri gönderildi ({len(message_str)} bytes).")
        except zmq.ZMQError as e:
            # Bağlantı kopmuş olabilir
            logger.error(f"Mobil cihaza veri gönderilemedi (ZMQ Hatası): {e}")
            # Yeniden bağlanmayı dene?
        except Exception as e:
            logger.error(f"Mobil cihaza veri gönderme hatası: {e}")

# Doğrudan çalıştırma için (test amaçlı)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mobile Sync Module (Placeholder)")
    parser.add_argument("--enable", action="store_true", help="Enable mobile sync")
    parser.add_argument("--port", type=int, default=PORT_MOBILE_SYNC, help="Port for mobile communication")
    parser.add_argument("--interval", type=float, default=1.0, help="Sync interval in seconds")
    parser.add_argument("--ai-port", type=int, default=PORT_AI_OUTPUT, help="Port for AI detections")
    parser.add_argument("--cal-port", type=int, default=PORT_CALIBRATION_RESULT, help="Port for calibration results")
    parser.add_argument("--method", type=str, default="wifi", choices=["wifi", "bluetooth"], help="Communication method")

    args = parser.parse_args()

    config = {
        "enabled": args.enable,
        "sync_port": args.port,
        "sync_interval_sec": args.interval,
        "ai_input_port": args.ai_port,
        "calibration_input_port": args.cal_port,
        "method": args.method
    }

    mobile_sync = MobileSync(config)
    mobile_sync.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Durduruluyor...")
    finally:
        mobile_sync.stop()

