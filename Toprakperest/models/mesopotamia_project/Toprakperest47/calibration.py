#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibration.py - Zemin Kalibrasyon Modülü

Version: 3.1
Author: AI Assistant (Manus)
Last Updated: 2025-04-30

SDR tabanlı yeraltı tespit sistemi için zemin kalibrasyonunu yönetir.
SDR alıcısından gelen verileri kullanarak zemin özelliklerini (nem, iletkenlik,
dielektrik sabiti, mineral içeriği) tahmin eder, arka plan gürültüsünü izler
ve kalibrasyon sonuçlarını diğer modüllere yayınlar.

Geliştirmeler:
- ZMQ üzerinden veri alımı ve SDR kontrolü.
- Çoklu frekans verilerine dayalı özellik çıkarımı.
- Geliştirilmiş zemin tipi tanımlama mantığı (placeholder).
- Sürekli arka plan gürültüsü izleme.
- Daha modüler ve okunabilir yapı.
- Türkçe dokümantasyon.
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
import pickle
from datetime import datetime
from threading import Thread, Event, Lock
from queue import Queue, Empty, Full
import scipy.signal as signal
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Tuple

# Yerel modüller
from config import (
    GroundType, COLOR_MAP_GROUND, PORT_SDR_DATA, PORT_SDR_CONTROL,
    PORT_CALIBRATION_RESULT, PORT_CALIBRATION_CONTROL,
    DEFAULT_SCAN_FREQ_START_MHZ, DEFAULT_SCAN_FREQ_END_MHZ,
    DEFAULT_CALIBRATION_DIR, DEFAULT_PERMITTIVITY
)

# Logging yapılandırması
logger = logging.getLogger("CALIBRATION")
# Ana logger yapılandırması main.py tarafından yapılır.
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format=\"%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s\",
        handlers=[logging.StreamHandler()] # Sadece konsola yaz
    )

# Sabitler
FEATURE_VECTOR_SIZE = 10 # Örnek özellik vektörü boyutu
MIN_SAMPLES_FOR_CALIBRATION = 5 # Bir frekans için gereken min örnek sayısı

class CalibrationModule:
    """
    Zemin kalibrasyonunu ve çevresel veri toplamayı yönetir.
    """

    # Örnek Zemin İmzaları (Deneysel olarak doldurulmalı)
    # Özellikler: [Ort. Genlik, Genlik Std, Faz Değişim Ort, Faz Değişim Std, Spektral Tepe Frekansı Norm., ...]
    GROUND_TYPE_SIGNATURES = {
        GroundType.DRY_SOIL: np.array([0.8, 0.1, 0.5, 0.2, 0.7] + [0]*(FEATURE_VECTOR_SIZE-5)),
        GroundType.WET_SOIL: np.array([0.6, 0.2, 1.5, 0.5, 0.3] + [0]*(FEATURE_VECTOR_SIZE-5)),
        GroundType.LIMESTONE: np.array([0.7, 0.1, 0.8, 0.3, 0.8] + [0]*(FEATURE_VECTOR_SIZE-5)),
        GroundType.IRON_OXIDE: np.array([0.5, 0.3, 1.2, 0.6, 0.2] + [0]*(FEATURE_VECTOR_SIZE-5)),
        GroundType.ROCKY: np.array([0.75, 0.15, 0.7, 0.3, 0.75] + [0]*(FEATURE_VECTOR_SIZE-5)),
        # ... diğer zemin türleri ...
        GroundType.UNKNOWN: np.array([0.5] * FEATURE_VECTOR_SIZE) # Varsayılan
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Kalibrasyon modülünü başlatır.

        Args:
            config (Dict[str, Any]): Modül yapılandırma parametreleri.
        """
        self.config = config
        self.calibration_time_sec = float(config.get("calibration_time_sec", 60))
        self.calibration_sweeps = int(config.get("calibration_sweeps", 20))
        self.reference_data_path = config.get("reference_data_path", os.path.join(DEFAULT_CALIBRATION_DIR, "reference.pkl"))
        self.noise_antenna = config.get("noise_antenna", "Dipole_RX2") # Gürültü ölçümü için anten
        self.scan_antenna = config.get("scan_antenna", "UWB_RX1") # Kalibrasyon taraması için anten
        self.min_freq_mhz = float(config.get("min_scan_freq_mhz", DEFAULT_SCAN_FREQ_START_MHZ))
        self.max_freq_mhz = float(config.get("max_scan_freq_mhz", DEFAULT_SCAN_FREQ_END_MHZ))
        self.noise_monitor_interval_sec = float(config.get("noise_monitor_interval_sec", 300))
        self.sample_rate = float(config.get("sample_rate", DEFAULT_SDR_SAMPLE_RATE)) # SDR\den alınmalı ama varsayılan

        # Kalibrasyon Durumu
        self.calibration_complete = False
        self.ground_type = GroundType.UNKNOWN
        self.estimated_permittivity = DEFAULT_PERMITTIVITY
        self.ground_params: Dict[str, Any] = {} # Detaylı parametreler (örn. nem, iletkenlik tahmini)
        self.calibration_progress = 0.0
        self.background_noise_db = -100.0 # dBFS veya dBm
        self.interference_freqs: List[float] = [] # Tespit edilen parazit frekansları (Hz)

        # Veri Yapıları
        self.calibration_data_buffer = defaultdict(lambda: deque(maxlen=50)) # {freq_hz: deque([samples1, samples2,...])}
        self.feature_vectors = defaultdict(list) # {freq_hz: [features1, features2, ...]}
        self.reference_signatures = self._load_reference_signatures()
        self.scaler = StandardScaler() # Özellik ölçeklendirme için

        # İletişim (ZMQ)
        self.context = zmq.Context()
        self.sdr_data_socket = self.context.socket(zmq.SUB)
        self.sdr_data_socket.connect(f"tcp://localhost:{config.get(\"input_port\", PORT_SDR_DATA)}")
        self.sdr_data_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sdr_data_socket.setsockopt(zmq.CONFLATE, 1)

        self.result_socket = self.context.socket(zmq.PUB)
        self.result_socket.bind(f"tcp://*:{config.get(\"result_port\", PORT_CALIBRATION_RESULT)}")

        self.control_socket = self.context.socket(zmq.REP)
        self.control_socket.bind(f"tcp://*:{config.get(\"control_port\", PORT_CALIBRATION_CONTROL)}")
        self.control_socket.setsockopt(zmq.RCVTIMEO, 1000)

        self.sdr_control_socket = self.context.socket(zmq.REQ)
        self.sdr_control_socket.connect(f"tcp://localhost:{PORT_SDR_CONTROL}")

        # Thread Yönetimi
        self.running = Event()
        self.shutdown_event = Event()
        self.lock = Lock()
        self.data_queue = Queue(maxsize=500) # Alınan veriler için kuyruk
        self.threads: Dict[str, Optional[Thread]] = {}

        logger.info("Kalibrasyon modülü başlatıldı.")

    def _load_reference_signatures(self) -> Dict[GroundType, np.ndarray]:
        """Referans zemin imzalarını yükler (varsa dosyadan, yoksa varsayılan)."""
        # TODO: Referans imzaları dosyadan yükleme eklenebilir
        logger.info("Varsayılan zemin imzaları kullanılıyor.")
        # Ölçeklendirme için imzaları hazırla
        signatures = list(self.GROUND_TYPE_SIGNATURES.values())
        try:
            self.scaler.fit(signatures) # Ölçekleyiciyi imzalarla eğit
            scaled_signatures = {gt: self.scaler.transform([sig])[0] for gt, sig in self.GROUND_TYPE_SIGNATURES.items()}
            logger.info("Referans zemin imzaları ölçeklendi.")
            return scaled_signatures
        except Exception as e:
            logger.error(f"Referans imzalar ölçeklenirken hata: {e}. Ölçeklenmemiş imzalar kullanılıyor.")
            return self.GROUND_TYPE_SIGNATURES

    def _save_calibration_result(self, filename: str = "last_calibration.pkl") -> bool:
        """Mevcut kalibrasyon sonucunu dosyaya kaydeder."""
        if not self.calibration_complete:
            logger.warning("Kaydedilecek tamamlanmış kalibrasyon yok.")
            return False

        result = {
            "timestamp": datetime.now().isoformat(),
            "ground_type": self.ground_type.name,
            "estimated_permittivity": self.estimated_permittivity,
            "ground_params": self.ground_params,
            "background_noise_db": self.background_noise_db,
            "interference_freqs_hz": self.interference_freqs,
            "feature_vectors": {f: np.mean(v, axis=0).tolist() for f, v in self.feature_vectors.items() if v}
        }
        save_path = os.path.join(DEFAULT_CALIBRATION_DIR, filename)
        try:
            os.makedirs(DEFAULT_CALIBRATION_DIR, exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(result, f)
            logger.info(f"Kalibrasyon sonucu kaydedildi: {save_path}")
            return True
        except Exception as e:
            logger.error(f"Kalibrasyon sonucu kaydedilemedi: {e}", exc_info=True)
            return False

    def _send_sdr_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """SDR Alıcı modülüne ZMQ üzerinden kontrol komutu gönderir ve yanıt bekler."""
        try:
            self.sdr_control_socket.send_json(command)
            poller = zmq.Poller()
            poller.register(self.sdr_control_socket, zmq.POLLIN)
            if poller.poll(3000): # 3 saniye timeout
                response = self.sdr_control_socket.recv_json()
                logger.debug(f"SDR Komutu: {command.get(\"command\")}, Yanıt: {response.get(\"status\")}")
                return response
            else:
                logger.error(f"SDR kontrol komutuna yanıt alınamadı (Timeout): {command.get(\"command\")}")
                # Soketi yeniden bağlamayı dene
                self.sdr_control_socket.close(linger=0)
                self.sdr_control_socket = self.context.socket(zmq.REQ)
                self.sdr_control_socket.connect(f"tcp://localhost:{PORT_SDR_CONTROL}")
                return {"status": "error", "message": "Timeout"}
        except zmq.ZMQError as e:
            logger.error(f"SDR kontrol ZMQ hatası: {e}")
            return {"status": "error", "message": f"ZMQ Error: {e}"}
        except Exception as e:
            logger.error(f"SDR kontrol komutu gönderilemedi: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def start(self) -> bool:
        """Kalibrasyon modülünü ve ilgili thread\leri başlatır."""
        if self.running.is_set():
            logger.warning("Kalibrasyon modülü zaten çalışıyor.")
            return True
            
        self.running.set()
        self.shutdown_event.clear()

        self.threads["data_receiver"] = Thread(target=self._data_receiver_worker, name="DataReceiver")
        self.threads["processing"] = Thread(target=self._processing_worker, name="Processing")
        self.threads["control"] = Thread(target=self._control_worker, name="Control")
        self.threads["noise_monitor"] = Thread(target=self._noise_monitoring_worker, name="NoiseMonitor")

        for name, thread in self.threads.items():
            if thread:
                thread.daemon = True
                thread.start()
                logger.info(f"{name} thread\i başlatıldı.")

        logger.info("Kalibrasyon modülü başlatıldı ve ZMQ verisi bekleniyor.")
        return True

    def stop(self):
        """Kalibrasyon modülünü ve thread\leri güvenli bir şekilde durdurur."""
        if not self.running.is_set() and self.shutdown_event.is_set():
            return
            
        logger.info("Kalibrasyon modülü durduruluyor...")
        self.running.clear()
        self.shutdown_event.set()

        for name, thread in self.threads.items():
            if thread and thread.is_alive():
                try:
                    thread.join(timeout=2.0)
                    if not thread.is_alive():
                        logger.info(f"{name} thread\i durduruldu.")
                    else:
                        logger.warning(f"{name} thread\i zamanında durmadı.")
                except Exception as e:
                     logger.error(f"{name} thread\ini durdururken hata: {e}")
            self.threads[name] = None

        try:
            self.sdr_data_socket.close(linger=0)
            self.result_socket.close(linger=0)
            self.control_socket.close(linger=0)
            self.sdr_control_socket.close(linger=0)
            self.context.term()
            logger.info("ZMQ soketleri kapatıldı.")
        except Exception as e:
            logger.error(f"ZMQ kapatılırken hata: {e}")

        logger.info("Kalibrasyon modülü durduruldu.")

    def _data_receiver_worker(self):
        """SDR Alıcıdan gelen ZMQ verilerini alır ve işleme kuyruğuna ekler."""
        logger.info("Veri alıcı worker başlatıldı.")
        while self.running.is_set():
            try:
                message = self.sdr_data_socket.recv_json()
                if "samples_iq" in message and "center_freq" in message:
                    # JSON listesini numpy array\e çevir (Optimize edilmiş)
                    samples_list = message["samples_iq"]
                    try:
                        # Doğrudan view ile daha hızlı
                        samples = np.array(samples_list, dtype=np.float64).view(np.complex128).squeeze()
                    except (ValueError, TypeError):
                         # Eski yöntem (güvenli fallback)
                         samples = np.array([complex(s[0], s[1]) for s in samples_list], dtype=np.complex64)
                    
                    data_packet = {
                        "samples": samples,
                        "freq_hz": message["center_freq"],
                        "sample_rate": message.get("sample_rate", self.sample_rate),
                        "gain": message.get("gain"),
                        "snr_db": message.get("snr_db"),
                        "timestamp": message.get("timestamp", time.time())
                    }
                    try:
                        self.data_queue.put_nowait(data_packet)
                    except Full:
                        logger.warning("Kalibrasyon veri kuyruğu dolu, veri atlandı!")
                else:
                    logger.warning(f"Alınan SDR mesajında eksik alanlar: {list(message.keys())}")
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM: break
                logger.error(f"ZMQ veri alma hatası: {e}")
                time.sleep(1)
            except json.JSONDecodeError:
                logger.error("Geçersiz JSON formatında SDR mesajı alındı.")
            except Exception as e:
                logger.error(f"Veri alıcı worker hatası: {e}", exc_info=True)
                time.sleep(1)
        logger.info("Veri alıcı worker durdu.")

    def _run_calibration_sequence(self):
        """Tam kalibrasyon dizisini (frekans taraması) yönetir."""
        if not self.running.is_set(): return
        logger.info("Kalibrasyon dizisi başlatılıyor...")
        with self.lock:
            self.calibration_complete = False
            self.calibration_progress = 0.0
            self.calibration_data_buffer.clear()
            self.feature_vectors.clear()
        
        freq_hz_list = np.linspace(self.min_freq_mhz * 1e6, self.max_freq_mhz * 1e6, self.calibration_sweeps)
        sweep_duration_sec = self.calibration_time_sec / self.calibration_sweeps

        # SDR\yi tarama antenine ayarla
        # response = self._send_sdr_command({"command": "set_antenna", "params": {"antenna": self.scan_antenna, "channel": 0}})
        # if response.get("status") != "ok":
        #     logger.error(f"Tarama antenine ({self.scan_antenna}) geçilemedi.")
            # Hata durumunda ne yapılmalı? Belki devam et?

        start_calibration_time = time.time()
        for i, freq_hz in enumerate(freq_hz_list):
            if not self.running.is_set(): break # Durdurma sinyali geldiyse çık
            
            logger.info(f"Kalibrasyon Tarama {i+1}/{self.calibration_sweeps}: {freq_hz / 1e6:.1f} MHz")
            response = self._send_sdr_command({"command": "set_frequency", "params": {"frequency_hz": freq_hz}})
            if response.get("status") != "ok":
                logger.error(f"Frekans {freq_hz / 1e6:.1f} MHz ayarlanamadı.")
                continue

            # Belirtilen süre boyunca veri topla (işleme thread\i bu verileri alacak)
            sweep_start_time = time.time()
            while time.time() - sweep_start_time < sweep_duration_sec:
                if not self.running.is_set(): break
                current_progress = (time.time() - start_calibration_time) / self.calibration_time_sec
                with self.lock:
                    self.calibration_progress = min(1.0, current_progress)
                time.sleep(0.1)
            if not self.running.is_set(): break

        if self.running.is_set():
            logger.info("Kalibrasyon veri toplama tamamlandı. Sonuçlar işleniyor...")
            # İşleme thread\inin kuyruktaki verileri bitirmesini bekle
            processing_finished = False
            wait_start_time = time.time()
            while time.time() - wait_start_time < 10.0: # Max 10 sn bekle
                 if self.data_queue.empty():
                     processing_finished = True
                     break
                 time.sleep(0.5)
            
            if processing_finished:
                 logger.info("Veri işleme tamamlandı.")
                 self._finalize_calibration()
            else:
                 logger.warning("Veri işleme zaman aşımına uğradı, kalibrasyon tamamlanamadı.")
                 with self.lock:
                     self.calibration_progress = 1.0 # Yine de bitti say
                     self.calibration_complete = False # Ama başarısız
        else:
            logger.info("Kalibrasyon dizisi yarıda kesildi.")
            with self.lock:
                self.calibration_progress = 1.0 # Bitti say
                self.calibration_complete = False

    def _processing_worker(self):
        """Kuyruktaki verileri işler, özellikleri çıkarır ve kalibrasyonu günceller."""
        logger.info("İşleme worker başlatıldı.")
        while self.running.is_set() or not self.data_queue.empty():
            try:
                data_packet = self.data_queue.get(timeout=0.1)
                samples = data_packet["samples"]
                freq_hz = data_packet["freq_hz"]
                sample_rate = data_packet["sample_rate"]
                
                # Veriyi kalibrasyon buffer\ına ekle
                with self.lock:
                    self.calibration_data_buffer[freq_hz].append(samples)
                
                # Özellikleri çıkar
                features = self._extract_features(samples, sample_rate, freq_hz)
                if features is not None:
                    with self.lock:
                        self.feature_vectors[freq_hz].append(features)
                        # Belirli sayıda örnek birikince zemin tipini güncelle?
                        # if len(self.feature_vectors[freq_hz]) > MIN_SAMPLES_FOR_CALIBRATION:
                        #     self._update_ground_type_estimate()
                            
                self.data_queue.task_done()

            except Empty:
                if not self.running.is_set(): break
                continue
            except Exception as e:
                logger.error(f"Veri işleme hatası: {e}", exc_info=True)
                try: self.data_queue.task_done() # Hata olsa bile görevi tamamla
                except ValueError: pass
                time.sleep(0.1)
        logger.info("İşleme worker durdu.")

    def _extract_features(self, samples: np.ndarray, sample_rate: float, freq_hz: float) -> Optional[np.ndarray]:
        """Verilen sinyal bloğundan kalibrasyon için özellik vektörü çıkarır."""
        try:
            n_samples = len(samples)
            if n_samples < 10: return None # Çok kısa sinyal

            # Zaman domeni özellikleri
            amplitude = np.abs(samples)
            mean_amp = np.mean(amplitude)
            std_amp = np.std(amplitude)
            skewness_amp = skew(amplitude)
            kurtosis_amp = kurtosis(amplitude)

            # Faz özellikleri
            phase = np.unwrap(np.angle(samples))
            phase_diff = np.diff(phase)
            mean_phase_diff = np.mean(phase_diff)
            std_phase_diff = np.std(phase_diff)

            # Frekans domeni özellikleri (FFT)
            fft_result = np.fft.fftshift(np.fft.fft(samples))
            fft_freqs = np.fft.fftshift(np.fft.fftfreq(n_samples, 1.0 / sample_rate))
            spectrum = np.abs(fft_result)**2
            
            # Spektral tepe noktası (merkez frekansa göre normalize edilmiş)
            peak_index = np.argmax(spectrum)
            spectral_peak_freq_norm = (fft_freqs[peak_index] - freq_hz) / (sample_rate / 2.0) if sample_rate > 0 else 0
            
            # Spektral yayılım (basitçe std dev)
            spectral_centroid = np.sum(fft_freqs * spectrum) / np.sum(spectrum) if np.sum(spectrum) > 1e-9 else 0
            spectral_spread = np.sqrt(np.sum(((fft_freqs - spectral_centroid)**2) * spectrum) / np.sum(spectrum)) if np.sum(spectrum) > 1e-9 else 0
            spectral_spread_norm = spectral_spread / (sample_rate / 2.0) if sample_rate > 0 else 0

            # Özellik vektörünü oluştur (FEATURE_VECTOR_SIZE boyutunda)
            feature_vector = np.array([
                mean_amp,
                std_amp,
                skewness_amp,
                kurtosis_amp,
                mean_phase_diff,
                std_phase_diff,
                spectral_peak_freq_norm,
                spectral_spread_norm,
                # Diğer özellikler eklenebilir...
                0, # Placeholder
                0  # Placeholder
            ])[:FEATURE_VECTOR_SIZE] # Boyutu garantile
            
            # NaN veya Inf değerleri kontrol et/temizle
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return feature_vector

        except Exception as e:
            logger.error(f"Özellik çıkarımı hatası: {e}", exc_info=True)
            return None

    def _finalize_calibration(self):
        """Toplanan verilere göre nihai kalibrasyon sonuçlarını hesaplar."""
        logger.info("Nihai kalibrasyon sonuçları hesaplanıyor...")
        with self.lock:
            if not self.feature_vectors:
                logger.warning("Kalibrasyon için yeterli özellik verisi toplanamadı.")
                self.calibration_complete = False
                return

            # Tüm frekanslardaki ortalama özellikleri birleştir
            all_freq_features = []
            valid_freqs = []
            for freq_hz, vectors in self.feature_vectors.items():
                if len(vectors) >= MIN_SAMPLES_FOR_CALIBRATION:
                    all_freq_features.append(np.mean(vectors, axis=0))
                    valid_freqs.append(freq_hz)
                else:
                    logger.debug(f"{freq_hz/1e6:.1f} MHz için yeterli örnek yok ({len(vectors)}/{MIN_SAMPLES_FOR_CALIBRATION})")
            
            if not all_freq_features:
                logger.warning("Ortalaması alınacak yeterli frekans verisi bulunamadı.")
                self.calibration_complete = False
                return

            # Ortalama özellik vektörünü hesapla
            overall_feature_vector = np.mean(all_freq_features, axis=0)
            
            # Zemin tipini tahmin et (en yakın imzayı bularak)
            self.ground_type = self._classify_ground_type(overall_feature_vector)
            
            # Zemine özgü parametreleri tahmin et (Placeholder)
            self.estimated_permittivity = self._estimate_permittivity(self.ground_type, overall_feature_vector)
            self.ground_params = {
                "conductivity_estimate_S/m": self._estimate_conductivity(self.ground_type, overall_feature_vector),
                "moisture_estimate_percent": self._estimate_moisture(self.ground_type, overall_feature_vector)
            }

            self.calibration_complete = True
            self.calibration_progress = 1.0
            logger.info(f"Kalibrasyon tamamlandı: Zemin Tipi={self.ground_type.name}, İzinirlik≈{self.estimated_permittivity:.2f}")
            
            # Sonuçları yayınla
            self._publish_calibration_result()
            # Sonucu kaydet
            self._save_calibration_result()

    def _classify_ground_type(self, feature_vector: np.ndarray) -> GroundType:
        """Verilen özellik vektörüne en yakın referans zemin tipini bulur."""
        if feature_vector is None or len(feature_vector) != FEATURE_VECTOR_SIZE:
            return GroundType.UNKNOWN
            
        try:
            # Gelen vektörü ölçekle
            scaled_vector = self.scaler.transform([feature_vector])[0]
            
            min_distance = float(\"inf\")
            best_match = GroundType.UNKNOWN

            # Ölçeklenmiş imzalarla mesafeyi hesapla
            for gt, signature in self.reference_signatures.items():
                if gt == GroundType.UNKNOWN: continue # Bilinmeyen ile karşılaştırma
                distance = euclidean_distances([scaled_vector], [signature])[0][0]
                logger.debug(f"Zemin Tipi Karşılaştırma: {gt.name}, Mesafe: {distance:.4f}")
                if distance < min_distance:
                    min_distance = distance
                    best_match = gt
            
            # TODO: Mesafe eşiği ekle? Çok uzaksa UNKNOWN dönebilir.
            logger.info(f"Tahmin edilen zemin tipi: {best_match.name} (Mesafe: {min_distance:.4f})")
            return best_match
            
        except Exception as e:
            logger.error(f"Zemin tipi sınıflandırma hatası: {e}")
            return GroundType.UNKNOWN

    def _estimate_permittivity(self, ground_type: GroundType, features: np.ndarray) -> float:
        """Zemin tipine ve özelliklere göre dielektrik sabitini tahmin eder (Placeholder)."""
        # Basit tahmin: Zemin tipine göre varsayılan değer + özelliklere göre küçük ayarlama
        base_permit = {
            GroundType.DRY_SOIL: 3, GroundType.WET_SOIL: 20, GroundType.LIMESTONE: 6,
            GroundType.IRON_OXIDE: 10, GroundType.MINERAL_RICH: 15, GroundType.ROCKY: 7,
            GroundType.SANDY: 4, GroundType.CLAY: 12, GroundType.MIXED: 8,
            GroundType.UNKNOWN: DEFAULT_PERMITTIVITY
        }.get(ground_type, DEFAULT_PERMITTIVITY)
        
        # Örnek ayarlama: Faz değişimi yüksekse izinirlik artar (nemli gibi)
        phase_feature_index = 4 # mean_phase_diff
        adjustment = (features[phase_feature_index] - 0.5) * 5 # Kaba bir ölçekleme
        
        estimated = base_permit + adjustment
        return max(1.0, min(80.0, estimated)) # Fiziksel sınırlar içinde tut

    def _estimate_conductivity(self, ground_type: GroundType, features: np.ndarray) -> float:
        """İletkenlik tahmini (Placeholder)."""
        # Örnek: Genlik std dev yüksekse iletkenlik artar (kayıp fazla)
        amp_std_index = 1
        conductivity = features[amp_std_index] * 0.1 # Kaba ölçekleme
        return max(0.001, conductivity)

    def _estimate_moisture(self, ground_type: GroundType, features: np.ndarray) -> float:
        """Nem tahmini (Placeholder)."""
        # Örnek: Faz değişimi yüksekse nem artar
        phase_feature_index = 4
        moisture = features[phase_feature_index] * 20 # Kaba ölçekleme
        return max(0.0, min(100.0, moisture))

    def _publish_calibration_result(self):
        """Hesaplanan kalibrasyon sonuçlarını ZMQ üzerinden yayınlar."""
        if not self.calibration_complete:
            return
            
        with self.lock:
            result_message = {
                "timestamp": time.time(),
                "calibration_complete": self.calibration_complete,
                "ground_type": self.ground_type.name,
                "estimated_permittivity": self.estimated_permittivity,
                "background_noise_db": self.background_noise_db,
                "interference_freqs_hz": self.interference_freqs,
                "ground_params": self.ground_params
            }
        try:
            self.result_socket.send_json(result_message)
            logger.info("Kalibrasyon sonuçları yayınlandı.", ground_type=self.ground_type.name)
        except Exception as e:
            logger.error(f"Kalibrasyon sonuçları yayınlanamadı: {e}")

    def _noise_monitoring_worker(self):
        """Periyodik olarak arka plan gürültüsünü ölçer."""
        logger.info("Arka plan gürültü izleyici başlatıldı.")
        while self.running.is_set():
            start_time = time.time()
            try:
                self._measure_background_noise()
            except Exception as e:
                logger.error(f"Arka plan gürültü ölçüm hatası: {e}", exc_info=True)
            
            # Bir sonraki ölçüme kadar bekle
            elapsed = time.time() - start_time
            wait_time = max(0, self.noise_monitor_interval_sec - elapsed)
            # shutdown_event ile bölünebilir bekleme
            self.shutdown_event.wait(timeout=wait_time)
            if self.shutdown_event.is_set(): break
            
        logger.info("Arka plan gürültü izleyici durduruldu.")

    def _measure_background_noise(self):
        """Gürültü referans antenini kullanarak arka plan gürültüsünü ölçer."""
        logger.debug("Arka plan gürültüsü ölçülüyor...")
        # 1. SDR\yi gürültü antenine ayarla
        # response = self._send_sdr_command({"command": "set_antenna", "params": {"antenna": self.noise_antenna, "channel": 0}}) # Ana RX kanalı kullanılıyor varsayımı
        # if response.get("status") != "ok":
        #     logger.error(f"Gürültü antenine ({self.noise_antenna}) geçilemedi.")
        #     return
        
        # 2. Geniş bir frekans aralığında veya birkaç noktada veri topla
        # Şimdilik sadece mevcut frekanstaki veriyi kullan (daha iyisi SDR\den spektrum istemek)
        noise_samples_list = []
        collect_duration = 5.0 # 5 saniye veri topla
        collect_start_time = time.time()
        temp_queue = Queue()

        # Geçici olarak veri alıcıyı bu kuyruğa yönlendir?
        # Veya doğrudan SDR\den oku?
        # Şimdilik data_queue\dan okuyalım, ama bu kalibrasyon sırasında sorun olabilir.
        # En iyisi SDR Receiver\a gürültü ölçüm komutu eklemek.
        # --- Placeholder --- 
        # logger.warning("Gürültü ölçümü için SDR Receiver\a özel komut gerekiyor. Şimdilik placeholder.")
        # Simüle edilmiş gürültü:
        noise_power_db = -90 + np.random.randn() * 5
        # --- Placeholder Sonu ---
        
        # Gerçek ölçüm için (SDR Receiver desteklerse):
        # response = self._send_sdr_command({"command": "measure_noise_floor", "params": {"duration_sec": collect_duration}})
        # if response.get("status") == "ok":
        #     noise_power_db = response.get("data", {}).get("noise_floor_dbfs", -100.0)
        #     interference = response.get("data", {}).get("interference_peaks", [])
        # else:
        #     logger.error("SDR Receiver\dan gürültü ölçümü alınamadı.")
        #     return

        with self.lock:
            self.background_noise_db = noise_power_db
            # self.interference_freqs = interference
            logger.info(f"Arka plan gürültüsü güncellendi: {self.background_noise_db:.2f} dB")
        
        # Sonucu yayınla (opsiyonel, sadece değiştiğinde yayınlanabilir)
        self._publish_calibration_result() 

        # 3. SDR\yi tekrar tarama antenine ayarla (eğer değiştirildiyse)
        # response = self._send_sdr_command({"command": "set_antenna", "params": {"antenna": self.scan_antenna, "channel": 0}})

    def _control_worker(self):
        """ZMQ üzerinden gelen kontrol komutlarını işler."""
        logger.info("Kontrol thread\i başlatıldı.")
        while not self.shutdown_event.is_set():
            try:
                message = self.control_socket.recv_json()
                response = self._handle_control_command(message)
                self.control_socket.send_json(response)
            except zmq.Again:
                continue # Timeout
            except json.JSONDecodeError:
                logger.error("Geçersiz JSON formatında kontrol mesajı alındı.")
                try: self.control_socket.send_json({"status": "error", "message": "Invalid JSON"})
                except: pass
            except Exception as e:
                logger.error(f"Kontrol thread hatası: {e}", exc_info=True)
                try: self.control_socket.send_json({"status": "error", "message": str(e)})
                except: pass
                time.sleep(0.1)
        logger.info("Kontrol thread\i durduruldu.")

    def _handle_control_command(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Gelen kontrol komutunu işler ve yanıt döndürür."""
        command = message.get("command")
        logger.info(f"Kontrol komutu alındı: {command}", params=message.get("params"))
        
        try:
            if command == "get_status":
                with self.lock:
                    status_data = {
                        "running": self.running.is_set(),
                        "calibration_complete": self.calibration_complete,
                        "calibration_progress": self.calibration_progress,
                        "ground_type": self.ground_type.name,
                        "estimated_permittivity": self.estimated_permittivity,
                        "background_noise_db": self.background_noise_db
                    }
                return {"status": "ok", "data": status_data}
            
            elif command == "start_calibration":
                if self.calibration_progress > 0 and self.calibration_progress < 1.0:
                     return {"status": "error", "message": "Kalibrasyon zaten चल रहा है."}
                # Kalibrasyonu ayrı bir thread\de başlat ki kontrol worker bloklanmasın
                cal_thread = Thread(target=self._run_calibration_sequence, name="CalibrationRunner")
                cal_thread.daemon = True
                cal_thread.start()
                return {"status": "ok", "message": "Kalibrasyon dizisi başlatıldı."}
            
            elif command == "get_result":
                 if not self.calibration_complete:
                     return {"status": "error", "message": "Kalibrasyon henüz tamamlanmadı."}
                 with self.lock:
                    result_data = {
                        "timestamp": time.time(), # Anlık sonuç
                        "ground_type": self.ground_type.name,
                        "estimated_permittivity": self.estimated_permittivity,
                        "background_noise_db": self.background_noise_db,
                        "interference_freqs_hz": self.interference_freqs,
                        "ground_params": self.ground_params
                    }
                 return {"status": "ok", "data": result_data}

            elif command == "save_reference":
                 success = self._save_calibration_result("manual_reference.pkl")
                 return {"status": "ok" if success else "error", "message": "Referans kaydedildi." if success else "Referans kaydedilemedi."}
                 
            else:
                return {"status": "error", "message": f"Bilinmeyen komut: {command}"}
                
        except Exception as e:
             logger.error(f"Kontrol komutu işlenirken hata: {e}", command=command, exc_info=True)
             return {"status": "error", "message": f"Komut işlenirken hata: {e}"}

    def is_alive(self) -> bool:
        """Modülün çalışıp çalışmadığını kontrol eder."""
        # Ana thread\lerin durumunu kontrol et
        proc_thread = self.threads.get("processing")
        return self.running.is_set() and (proc_thread is not None and proc_thread.is_alive())

# --- Komut Satırı Arayüzü (Bağımsız Çalıştırma İçin) --- 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ground Calibration Module")
    parser.add_argument("--config", type=str, default=None, help="Path to system JSON configuration file")
    parser.add_argument("--run-cal", action="store_true", help="Start calibration sequence immediately after launch")

    args = parser.parse_args()

    # Yapılandırmayı yükle
    config = {}
    if args.config:
        try:
            with open(args.config, \"r\") as f:
                file_config = json.load(f)
                config = file_config.get("module_configs", {}).get("calibration", {})
                logger.info(f"Yapılandırma dosyası yüklendi: {args.config}")
        except FileNotFoundError:
            logger.error(f"Yapılandırma dosyası bulunamadı: {args.config}. Varsayılanlar kullanılıyor.")
        except json.JSONDecodeError as e:
            logger.error(f"Yapılandırma dosyası okunamadı (JSON Hatası): {e}. Varsayılanlar kullanılıyor.")
    else:
        logger.warning("Yapılandırma dosyası belirtilmedi, varsayılan ayarlar kullanılıyor.")

    # Kalibrasyon modülünü başlat
    calibration_module = CalibrationModule(config)

    def shutdown_handler(signum, frame):
        print("\nKapatma sinyali alındı, Kalibrasyon Modülü durduruluyor...")
        calibration_module.stop()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    if calibration_module.start():
        logger.info("Kalibrasyon Modülü çalışıyor.")
        if args.run_cal:
            logger.info("Otomatik kalibrasyon başlatılıyor...")
            # Kalibrasyonu ayrı thread\de başlat
            auto_cal_thread = Thread(target=calibration_module._run_calibration_sequence, name="AutoCalRunner")
            auto_cal_thread.daemon = True
            auto_cal_thread.start()
            
        # Ana thread\in çalışmasını bekle
        while calibration_module.running.is_set():
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                break # Sinyal işleyici zaten çağrılacak
    else:
        logger.critical("Kalibrasyon Modülü başlatılamadı.")

    logger.info("Kalibrasyon Modülü programı sonlandı.")

