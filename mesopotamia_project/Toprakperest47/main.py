#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py - SDR Tabanlı Yeraltı Tespit Sistemi Ana Orkestrasyon Modülü

Version: 2.0
Author: AI Assistant (Manus)
Last Updated: 2025-04-30

Sistemin ana giriş noktası. Tüm alt modüllerin (SDR, Kalibrasyon, Ön İşleme,
AI Sınıflandırma, Görselleştirme, İzleme) başlatılmasından, yönetilmesinden
ve koordinasyonundan sorumludur. Yapılandırmayı yükler, modül yaşam döngüsünü
yönetir ve sistemin düzgün bir şekilde kapatılmasını sağlar.
"""

import os
import sys
import time
import json
import signal
import logging
import logging.handlers
import argparse
import threading
from typing import Dict, List, Optional, Any, Type
from pathlib import Path
import structlog

# Sistem yollarını ayarla (eğer modüller farklı dizinlerdeyse)
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Yerel modüller
from config import (
    DEFAULT_CONFIG_FILENAME, MAX_MODULE_START_TIME_SEC, HEARTBEAT_INTERVAL_SEC,
    MAX_RESTART_ATTEMPTS, RESOURCE_CHECK_INTERVAL_SEC, ModuleStatus,
    PORT_SDR_CONTROL, PORT_CALIBRATION_CONTROL, PORT_PREPROCESSING_CONTROL,
    PORT_AI_CONTROL, PORT_MONITORING_CONTROL, PORT_UI_CONTROL # UI kontrol portu eklendi
)
from sdr_receiver import SDRReceiver
from calibration import CalibrationModule # Orijinal isim kullanıldı
from preprocessing import SignalPreprocessor
from ai_classifier import AIClassifier
from depth_3d_view import Depth3DView # Görselleştirme modülü
from monitoring import MonitoringService # İzleme modülü
from radar_ui import RadarMainWindow # Arayüz modülü
from mobile_sync import MobileSync # Mobil senkronizasyon (opsiyonel)

# --- Logging Kurulumu --- 

def setup_logging(log_level_str: str = "INFO", log_directory: str = "logs", max_log_size_mb: int = 10, backup_count: int = 5):
    """Yapılandırılmış loglamayı ayarlar."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    log_dir = Path(log_directory)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "system.log"

    file_handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=max_log_size_mb * 1024 * 1024,
    backupCount=backup_count, encoding="utf-8"
)

    console_handler = logging.StreamHandler()

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[file_handler, console_handler]
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # structlog.dev.ConsoleRenderer(), # Renkli konsol çıktısı için
            structlog.processors.JSONRenderer(), # Dosya için JSON
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

# Global logger (setup_logging çağrıldıktan sonra kullanılacak)
logger = structlog.get_logger("main")

# --- Modül Yönetimi --- 

class ModuleInfo:
    """Modül bilgilerini ve durumunu tutar."""
    def __init__(self, name: str, module_class: Type, config: Dict[str, Any], dependencies: List[str]):
        self.name = name
        self.module_class = module_class
        self.config = config
        self.dependencies = dependencies
        self.instance: Optional[Any] = None
        self.thread: Optional[threading.Thread] = None
        self.status: ModuleStatus = ModuleStatus.STOPPED
        self.last_heartbeat: float = 0.0
        self.restart_count: int = 0
        self.lock = threading.Lock()

class ModuleManager:
    """Modül yaşam döngüsünü yönetir."""
    def __init__(self, system_config: Dict[str, Any]):
        self.system_config = system_config
        self.modules: Dict[str, ModuleInfo] = {}
        self.module_order: List[str] = [] # Başlatma sırası
        self.lock = threading.RLock() # Reentrant lock
        self.running = threading.Event()
        self.monitor_thread: Optional[threading.Thread] = None

    def register_module(self, name: str, module_class: Type, config_key: str, dependencies: List[str]):
        """Bir modülü yöneticiye kaydeder."""
        with self.lock:
            if name in self.modules:
                logger.warning("Modül zaten kayıtlı", module_name=name)
                return
            module_config = self.system_config.get("module_configs", {}).get(config_key, {})
            self.modules[name] = ModuleInfo(name, module_class, module_config, dependencies)
            self.module_order.append(name)
            logger.info("Modül kaydedildi", module_name=name, config_key=config_key)

    def start_all_modules(self) -> bool:
        """Tüm kayıtlı modülleri bağımlılık sırasına göre başlatır."""
        logger.info("Tüm modüller başlatılıyor...")
        self.running.set()
        
        # Sağlık izleme thread\ini başlat
        if not self.monitor_thread or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(target=self._monitor_modules, daemon=True, name="ModuleManagerMonitor")
            self.monitor_thread.start()
            
        success = True
        with self.lock:
            for name in self.module_order:
                if not self._start_module(name):
                    success = False
                    logger.error("Modül başlatılamadığı için sistem başlatma durduruldu", failed_module=name)
                    # Başarısızlık durumunda başlatılanları geri al?
                    self.stop_all_modules()
                    break
        
        if success:
            logger.info("Tüm modüller başarıyla başlatıldı.")
        return success

    def stop_all_modules(self):
        """Tüm modülleri ters sırada durdurur."""
        logger.info("Tüm modüller durduruluyor...")
        self.running.clear() # İzleme döngüsünü durdur
        
        with self.lock:
            # Modülleri başlatma sırasının tersine göre durdur
            for name in reversed(self.module_order):
                self._stop_module(name)
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=HEARTBEAT_INTERVAL_SEC * 1.5)
            
        logger.info("Tüm modüller durduruldu.")

    def _start_module(self, name: str) -> bool:
        """Belirtilen modülü başlatır."""
        module_info = self.modules.get(name)
        if not module_info:
            logger.error("Başlatılacak modül bulunamadı", module_name=name)
            return False

        with module_info.lock:
            if module_info.status != ModuleStatus.STOPPED and module_info.status != ModuleStatus.ERROR:
                logger.warning("Modül zaten çalışıyor veya başlatılıyor", module_name=name, status=module_info.status.name)
                return True # Zaten çalışıyorsa başarılı say

            logger.info("Modül başlatılıyor", module_name=name)
            module_info.status = ModuleStatus.INITIALIZING
            module_info.restart_count = 0 # Başarılı başlatmada sıfırla

            # Bağımlılıkları kontrol et
            for dep_name in module_info.dependencies:
                dep_info = self.modules.get(dep_name)
                if not dep_info or dep_info.status != ModuleStatus.RUNNING:
                    logger.error("Bağımlılık karşılanmadı", module_name=name, dependency=dep_name, dep_status=dep_info.status.name if dep_info else "NOT_FOUND")
                    module_info.status = ModuleStatus.ERROR
                    return False

            try:
                # Modül örneğini oluştur
                module_info.instance = module_info.module_class(module_info.config)

                # Modülün kendi start metodunu çağır (varsa ve thread başlatıyorsa)
                if hasattr(module_info.instance, "start"):
                    start_result = module_info.instance.start()
                    if start_result is False:  # Başlatma başarısız olursa
                        raise RuntimeError("Modülün start() metodu False döndürdü")
                    # Eğer start metodu thread başlatmıyorsa, burada manuel başlatılabilir
                    # module_info.thread = threading.Thread(...)
                    # module_info.thread.start()
                else:
                    # Eğer start metodu yoksa, run metodunu thread içinde çalıştır
                    module_info.thread = threading.Thread(
                        target=self._run_module_instance,
                        args=(module_info,),
                        daemon=True,
                        name=f"{name}_thread"
                    )
                    module_info.thread.start()

                # Başlatma zaman aşımı kontrolü (modülün kendi içinde yapılmalı veya heartbeat beklenmeli)
                # Şimdilik basit bir bekleme
                time.sleep(1.0) # Modülün başlaması için kısa bir süre tanı
                
                # Durumu RUNNING olarak ayarla (heartbeat ile doğrulanacak)
                module_info.status = ModuleStatus.RUNNING 
                module_info.last_heartbeat = time.time()
                logger.info("Modül başlatıldı (çalışıyor durumu heartbeat ile doğrulanacak)", module_name=name)
                return True

            except Exception as e:
                logger.error("Modül başlatma hatası", module_name=name, error=str(e), exc_info=True)
                module_info.status = ModuleStatus.ERROR
                # Başlatma sırasında hata olursa instance\ı temizle
                if module_info.instance and hasattr(module_info.instance,"stop"):
                    try: module_info.instance.stop()
                    except: pass
                module_info.instance = None
                module_info.thread = None
                return False

    def _run_module_instance(self, module_info: ModuleInfo):
        """Modülün run() metodunu çağıran thread hedefi."""
        try:
            if hasattr(module_info.instance, "run"):
                 module_info.instance.run() # run metodu bloklayıcı olmalı
            else:
                 logger.warning("Modülün run() metodu bulunamadı", module_name=module_info.name)
                 # Eğer run yoksa ve start thread başlatmadıysa, burada bir sorun var
                 # Şimdilik thread sonlanacak
        except Exception as e:
            logger.error("Modül çalışma zamanı hatası", module_name=module_info.name, error=str(e), exc_info=True)
            with module_info.lock:
                 module_info.status = ModuleStatus.ERROR
        finally:
            logger.info("Modül thread\i sonlandı", module_name=module_info.name)
            # Durumu STOPPED yapma, monitor thread yönetecek

    def _stop_module(self, name: str, timeout: float = 5.0) -> bool:
        """Belirtilen modülü durdurur."""
        module_info = self.modules.get(name)
        if not module_info:
            return True # Modül yoksa durdurulmuş say

        with module_info.lock:
            if module_info.status == ModuleStatus.STOPPED:
                return True
            
            current_status = module_info.status
            module_info.status = ModuleStatus.TERMINATING
            logger.info("Modül durduruluyor", module_name=name, previous_status=current_status.name)

            try:
                # Modülün kendi stop metodunu çağır
                if module_info.instance and hasattr(module_info.instance, "stop"):
                    module_info.instance.stop()
                
                # Eğer thread varsa ve bizim tarafımızdan başlatıldıysa bekle
                if module_info.thread and module_info.thread.is_alive():
                    module_info.thread.join(timeout=timeout)
                    if module_info.thread.is_alive():
                        logger.warning("Modül thread\i zamanında durmadı", module_name=name)
                        # Burada daha agresif durdurma yöntemleri denenebilir (örn. process kill)
                
                module_info.status = ModuleStatus.STOPPED
                module_info.instance = None
                module_info.thread = None
                logger.info("Modül başarıyla durduruldu", module_name=name)
                return True

            except Exception as e:
                logger.error("Modül durdurma hatası", module_name=name, error=str(e), exc_info=True)
                module_info.status = ModuleStatus.ERROR # Durdurma hatası da bir hatadır
                return False

    def _monitor_modules(self):
        """Modüllerin sağlık durumunu ve heartbeat\lerini izler."""
        logger.info("Modül izleyici başlatıldı.")
        while self.running.is_set():
            start_time = time.time()
            modules_to_restart = []
            with self.lock:
                for name, module_info in self.modules.items():
                    with module_info.lock:
                        # Heartbeat kontrolü (modülün kendisi güncellemiyorsa, thread durumuna bak)
                        if module_info.status == ModuleStatus.RUNNING:
                             is_alive = False
                             if module_info.instance and hasattr(module_info.instance, "is_alive"):
                                 is_alive = module_info.instance.is_alive() # Modül kendi durumunu bildiriyorsa
                             elif module_info.thread and module_info.thread.is_alive():
                                 is_alive = True # Thread çalışıyorsa
                             
                             if not is_alive:
                                 logger.warning("Modül yanıt vermiyor veya thread durmuş", module_name=name)
                                 module_info.status = ModuleStatus.ERROR
                                 modules_to_restart.append(name)
                             else:
                                 # Yanıt veriyorsa heartbeat\i güncelle (opsiyonel, modül kendi yapabilir)
                                 module_info.last_heartbeat = time.time()
                                 
                        # Hata durumundaki modülleri yeniden başlatma listesine ekle
                        elif module_info.status == ModuleStatus.ERROR:
                             modules_to_restart.append(name)
            
            # Kilidi bıraktıktan sonra yeniden başlatma işlemlerini yap
            for name in modules_to_restart:
                self._attempt_restart(name)

            # Döngü süresini hesaba katarak bekle
            elapsed = time.time() - start_time
            sleep_time = max(0, HEARTBEAT_INTERVAL_SEC - elapsed)
            if self.running.is_set():
                time.sleep(sleep_time)
                
        logger.info("Modül izleyici durduruldu.")

    def _attempt_restart(self, name: str):
        """Hata veren bir modülü yeniden başlatmayı dener."""
        with self.lock:
            module_info = self.modules.get(name)
            if not module_info or module_info.status == ModuleStatus.RECOVERING: # Zaten deneniyorsa atla
                return

            if module_info.restart_count < MAX_RESTART_ATTEMPTS:
                logger.warning("Modül yeniden başlatılıyor", module_name=name, attempt=module_info.restart_count + 1)
                module_info.status = ModuleStatus.RECOVERING
                module_info.restart_count += 1
                
                # Önce durdurmayı dene
                self._stop_module(name, timeout=2.0)
                
                # Biraz bekle
                time.sleep(1.0)
                
                # Yeniden başlatmayı dene
                if self._start_module(name):
                    logger.info("Modül başarıyla yeniden başlatıldı", module_name=name)
                    # Başarılı olunca restart sayacını sıfırla?
                    # module_info.restart_count = 0 
                else:
                    logger.error("Modül yeniden başlatılamadı", module_name=name)
                    # Hata durumu devam ediyor
                    module_info.status = ModuleStatus.ERROR 
            else:
                logger.critical("Modül maksimum yeniden başlatma denemesini aştı, kalıcı hata!", module_name=name)
                # Kalıcı hata durumunda modülü durdur ve tekrar deneme
                self._stop_module(name)
                # Burada sistem yöneticisine bildirim gönderilebilir

# --- Ana Sistem Orkestrasyonu --- 

class SystemOrchestrator:
    """Sistemin genelini yönetir."""
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.system_config: Optional[Dict[str, Any]] = None
        self.module_manager: Optional[ModuleManager] = None
        self.shutdown_event = threading.Event()

        # Sinyal işleyicileri
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)

    def _handle_shutdown_signal(self, signum, frame):
        """Kapatma sinyallerini (SIGINT, SIGTERM) yakalar."""
        logger.warning("Kapatma sinyali alındı", signal=signal.Signals(signum).name)
        self.shutdown_event.set()

    def load_configuration(self) -> bool:
        """Sistem yapılandırmasını JSON dosyasından yükler."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.system_config = json.load(f)
            # TODO: Yapılandırma doğrulama (Pydantic veya jsonschema ile)
            log_config = self.system_config.get("logging", {})
            setup_logging(
                log_level_str=log_config.get("level", "INFO"),
                log_directory=log_config.get("directory", "logs"),
                max_log_size_mb=log_config.get("max_size_mb", 10),
                backup_count=log_config.get("backup_count", 5)
            )
            logger.info("Sistem yapılandırması başarıyla yüklendi", path=self.config_path)
            return True
        except FileNotFoundError:
            logging.error(f"Yapılandırma dosyası bulunamadı: {self.config_path}")
            return False
        except json.JSONDecodeError as e:
            logging.error(f"Yapılandırma dosyası okunamadı (JSON Hatası): {e}")
            return False
        except Exception as e:
            logging.error(f"Yapılandırma yüklenirken beklenmedik hata: {e}", exc_info=True)
            return False

    def initialize_system(self) -> bool:
        """Modül yöneticisini başlatır ve modülleri kaydeder."""
        if not self.system_config:
            logger.error("Sistem yapılandırması yüklenmedi, başlatılamıyor.")
            return False
            
        self.module_manager = ModuleManager(self.system_config)
        
        # Modülleri kaydet (bağımlılıklara dikkat et)
        # Sıra önemli olabilir
        self.module_manager.register_module("monitoring", MonitoringService, "monitoring", [])
        self.module_manager.register_module("sdr_receiver", SDRReceiver, "sdr_receiver", ["monitoring"])
        self.module_manager.register_module("calibration", CalibrationModule, "calibration", ["sdr_receiver", "monitoring"])
        self.module_manager.register_module("preprocessing", SignalPreprocessor, "preprocessing", ["sdr_receiver", "calibration", "monitoring"])
        self.module_manager.register_module("ai_classifier", AIClassifier, "ai_classifier", ["preprocessing", "monitoring"])
        self.module_manager.register_module("depth_3d_view", Depth3DView, "visualization", ["ai_classifier", "monitoring"])
        self.module_manager.register_module("radar_ui", RadarMainWindow, "visualization", ["ai_classifier", "monitoring"])
        self.module_manager.register_module("mobile_sync", MobileSync, "mobile_sync", ["ai_classifier", "monitoring"]) # Opsiyonel
        
        logger.info("Sistem başlatılmaya hazır.")
        return True

    def run(self):
        """Sistemi çalıştırır ve kapatma sinyalini bekler."""
        if not self.module_manager:
            logger.error("Sistem başlatılmadı, çalıştırılamıyor.")
            return

        if not self.module_manager.start_all_modules():
            logger.critical("Sistem başlatılamadı, çıkılıyor.")
            return

        logger.info("Sistem çalışıyor. Kapatmak için CTRL+C basın.")

        # Ana thread\in kapatma sinyalini beklemesini sağla
        try:
            self.shutdown_event.wait()
        except KeyboardInterrupt:
            # CTRL+C durumunda da shutdown event set edilebilir
            logger.warning("KeyboardInterrupt alındı, sistem kapatılıyor...")
            self.shutdown_event.set()
        
        logger.info("Kapatma işlemi başlatıldı...")
        self.module_manager.stop_all_modules()
        logger.info("Sistem başarıyla kapatıldı.")

# --- Ana Giriş Noktası --- 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesopotamia SDR Yeraltı Tespit Sistemi")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_FILENAME,
        help=f"Sistem yapılandırma dosyasının yolu (varsayılan: {DEFAULT_CONFIG_FILENAME})"
    )
    args = parser.parse_args()

    # Geçici logger (yapılandırma yüklenene kadar)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    orchestrator = SystemOrchestrator(args.config)

    if orchestrator.load_configuration():
        if orchestrator.initialize_system():
            orchestrator.run()
        else:
            logger.critical("Sistem başlatılamadı.")
    else:
        logger.critical("Yapılandırma yüklenemedi, sistem başlatılamıyor.")

    logging.info("Program sonlandı.")

