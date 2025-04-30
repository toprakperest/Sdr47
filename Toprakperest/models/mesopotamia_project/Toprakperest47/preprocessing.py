#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocessing.py - Sinyal Ön İşleme Modülü

Version: 2.1
Author: AI Assistant (Manus)
Last Updated: 2025-04-30

SDR alıcısından gelen ham IQ verilerini alır ve AI sınıflandırması ile
3D görselleştirme için uygun hale getirir. Bu modül, gürültü azaltma,
yankı tespiti, derinlik tahmini ve temel clutter (istenmeyen yansıma)
filtreleme işlemlerini gerçekleştirir.

Geliştirmeler:
- ZMQ üzerinden SDR verisi ve Kalibrasyon sonucu alımı.
- Wavelet dönüşümü ile gürültü azaltma.
- Optimize edilmiş yankı tespiti ve analizi (genlik, faz, derinlik).
- Kalibrasyon modülünden alınan dielektrik sabiti ile derinlik hesabı.
- Basit clutter filtreleme.
- Türkçe dokümantasyon ve iyileştirilmiş kod yapısı.
"""

import numpy as np
import scipy.signal as signal
from scipy.constants import speed_of_light
import pywt # PyWavelets kütüphanesi
import zmq
import json
import time
import logging
import logging.handlers
from threading import Thread, Event, Lock
from queue import Queue, Empty, Full
from typing import Dict, Any, Tuple, List, Optional

# Yerel modüller
from config import (
    GroundType, PORT_SDR_DATA, PORT_CALIBRATION_RESULT,
    PORT_PREPROCESSING_OUTPUT, PORT_PREPROCESSING_CONTROL,
    DEFAULT_PERMITTIVITY
)

# Logging yapılandırması
logger = logging.getLogger("PREPROCESSING")
# Ana logger yapılandırması main.py tarafından yapılır.
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format=\"%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s\",
        handlers=[logging.StreamHandler()] # Sadece konsola yaz
    )

# Varsayılan dielektrik sabitleri (Kalibrasyondan güncellenir)
GROUND_PERMITTIVITY = {
    GroundType.DRY_SOIL: 3,
    GroundType.WET_SOIL: 20,
    GroundType.LIMESTONE: 6,
    GroundType.IRON_OXIDE: 10,
    GroundType.MINERAL_RICH: 15,
    GroundType.ROCKY: 7,
    GroundType.SANDY: 4,
    GroundType.CLAY: 12,
    GroundType.MIXED: 8,
    GroundType.UNKNOWN: DEFAULT_PERMITTIVITY
}

class SignalPreprocessor:
    """Gelen SDR verilerini AI ve görselleştirme için ön işler."""

    def __init__(self, config: Dict[str, Any]):
        """
        Ön işleme modülünü başlatır.

        Args:
            config (Dict[str, Any]): Modül yapılandırma parametreleri.
        """
        self.config = config
        self.wavelet_type = config.get("wavelet_type", "db4")
        self.wavelet_level = int(config.get("wavelet_level", 4))
        self.depth_estimation_method = config.get("depth_method", "peak_time") # "peak_time" veya "correlation"
        self.clutter_filter_threshold = float(config.get("clutter_threshold", 0.5)) # 0-1 arası, göreceli genlik
        self.min_peak_distance_ms = float(config.get("min_peak_distance_ms", 10))
        self.surface_skip_ms = float(config.get("surface_skip_ms", 5))

        # Kalibrasyon Durumu
        self.current_ground_type = GroundType.UNKNOWN
        self.current_permittivity = GROUND_PERMITTIVITY[GroundType.UNKNOWN]
        self.background_noise_db = -100.0
        self.calibration_lock = Lock()

        # İletişim (ZMQ)
        self.context = zmq.Context()
        # SDR Veri Girişi
        self.sdr_data_socket = self.context.socket(zmq.SUB)
        self.sdr_data_socket.connect(f"tcp://localhost:{config.get(\"input_port_sdr\", PORT_SDR_DATA)}")
        self.sdr_data_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sdr_data_socket.setsockopt(zmq.CONFLATE, 1) # Sadece en son mesajı al
        # Kalibrasyon Veri Girişi
        self.calibration_socket = self.context.socket(zmq.SUB)
        self.calibration_socket.connect(f"tcp://localhost:{config.get(\"input_port_cal\", PORT_CALIBRATION_RESULT)}")
        self.calibration_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.calibration_socket.setsockopt(zmq.CONFLATE, 1)
        # İşlenmiş Veri Çıkışı
        self.output_socket = self.context.socket(zmq.PUB)
        self.output_socket.bind(f"tcp://*:{config.get(\"output_port\", PORT_PREPROCESSING_OUTPUT)}")
        # Kontrol Girişi
        self.control_socket = self.context.socket(zmq.REP)
        self.control_socket.bind(f"tcp://*:{config.get(\"control_port\", PORT_PREPROCESSING_CONTROL)}")
        self.control_socket.setsockopt(zmq.RCVTIMEO, 1000)

        # Thread Yönetimi
        self.running = Event()
        self.shutdown_event = Event()
        self.data_queue = Queue(maxsize=100) # Gelen SDR verileri için kuyruk
        self.threads: Dict[str, Optional[Thread]] = {}

        logger.info("Ön işleme modülü başlatıldı.")

    def start(self) -> bool:
        """Ön işleme modülünü ve ilgili thread\leri başlatır."""
        if self.running.is_set():
            logger.warning("Ön işleme modülü zaten çalışıyor.")
            return True
            
        self.running.set()
        self.shutdown_event.clear()

        self.threads["sdr_listener"] = Thread(target=self._sdr_listener_worker, name="SDRListener")
        self.threads["cal_listener"] = Thread(target=self._calibration_listener_worker, name="CalListener")
        self.threads["processing"] = Thread(target=self._processing_worker, name="Processing")
        self.threads["control"] = Thread(target=self._control_worker, name="Control")
        
        for name, thread in self.threads.items():
            if thread:
                thread.daemon = True
                thread.start()
                logger.info(f"{name} thread\i başlatıldı.")
                
        logger.info("Ön işleme modülü başlatıldı ve ZMQ verisi bekleniyor.")
        return True

    def stop(self):
        """Ön işleme modülünü ve thread\leri güvenli bir şekilde durdurur."""
        if not self.running.is_set() and self.shutdown_event.is_set():
            return
            
        logger.info("Ön işleme modülü durduruluyor...")
        self.running.clear()
        self.shutdown_event.set()

        for name, thread in self.threads.items():
            if thread and thread.is_alive():
                try:
                    thread.join(timeout=1.5)
                    if not thread.is_alive():
                        logger.info(f"{name} thread\i durduruldu.")
                    else:
                        logger.warning(f"{name} thread\i zamanında durmadı.")
                except Exception as e:
                     logger.error(f"{name} thread\ini durdururken hata: {e}")
            self.threads[name] = None

        try:
            self.sdr_data_socket.close(linger=0)
            self.calibration_socket.close(linger=0)
            self.output_socket.close(linger=0)
            self.control_socket.close(linger=0)
            # self.context.term() # Context paylaşımlı olabilir, ana uygulama kapatmalı
            logger.info("Ön işleme ZMQ soketleri kapatıldı.")
        except Exception as e:
            logger.error(f"Ön işleme ZMQ kapatılırken hata: {e}")

        logger.info("Ön işleme modülü durduruldu.")

    def _sdr_listener_worker(self):
        """SDR Alıcıdan gelen ZMQ verilerini dinler ve işleme kuyruğuna ekler."""
        logger.info("SDR veri dinleyici worker başlatıldı.")
        while self.running.is_set():
            try:
                message = self.sdr_data_socket.recv_json()
                # Temel doğrulama
                if isinstance(message, dict) and "samples_iq" in message:
                    try:
                        # Kuyruk doluysa en eskiyi at (non-blocking get)
                        while self.data_queue.full():
                            self.data_queue.get_nowait()
                            logger.warning("Ön işleme SDR veri kuyruğu doluydu, en eski veri atıldı.")
                        self.data_queue.put_nowait(message)
                    except Full: # put_nowait nadiren de olsa Full verebilir
                        logger.error("Ön işleme SDR veri kuyruğuna eklenemedi (hala dolu?).")
                    except Empty: # get_nowait boşken çağrılırsa
                        self.data_queue.put_nowait(message) # Direkt ekle
                else:
                    logger.warning("Alınan SDR mesajı geçersiz veya eksik.")
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM: break # Context kapatıldı
                logger.error(f"ZMQ SDR veri alma hatası: {e}")
                time.sleep(1) # Hata durumunda bekle
            except json.JSONDecodeError:
                logger.error("Geçersiz JSON formatında SDR mesajı alındı.")
            except Exception as e:
                logger.error(f"SDR dinleyici worker hatası: {e}", exc_info=True)
                time.sleep(1)
        logger.info("SDR veri dinleyici worker durdu.")

    def _calibration_listener_worker(self):
        """Kalibrasyon modülünden gelen ZMQ verilerini dinler ve durumu günceller."""
        logger.info("Kalibrasyon dinleyici worker başlatıldı.")
        while self.running.is_set():
            try:
                message = self.calibration_socket.recv_json()
                logger.debug(f"Kalibrasyon mesajı alındı: {list(message.keys())}")
                with self.calibration_lock:
                    if "ground_type" in message:
                        try:
                            gt_name = message["ground_type"]
                            self.current_ground_type = GroundType[gt_name]
                            # Dielektrik sabitini de güncelle
                            self.current_permittivity = message.get("estimated_permittivity", 
                                                                  GROUND_PERMITTIVITY.get(self.current_ground_type, DEFAULT_PERMITTIVITY))
                            logger.info(f"Kalibrasyon güncellendi: Zemin={self.current_ground_type.name}, İzinirlik={self.current_permittivity:.2f}")
                        except KeyError:
                            logger.warning(f"Bilinmeyen zemin tipi adı alındı: {gt_name}")
                            self.current_ground_type = GroundType.UNKNOWN
                            self.current_permittivity = DEFAULT_PERMITTIVITY
                    if "background_noise_db" in message:
                        self.background_noise_db = float(message["background_noise_db"])
                        # logger.debug(f"Arka plan gürültüsü güncellendi: {self.background_noise_db:.2f} dB")
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

    def _processing_worker(self):
        """Kuyruktaki SDR verilerini işler ve sonuçları yayınlar."""
        logger.info("Ön işleme worker başlatıldı.")
        while self.running.is_set() or not self.data_queue.empty():
            try:
                sdr_data = self.data_queue.get(timeout=0.5)
                start_proc_time = time.perf_counter()

                # Veriyi NumPy array\e çevir (Optimize edilmiş)
                samples_list = sdr_data.get("samples_iq")
                if samples_list is None: 
                    logger.warning("SDR verisinde 'samples_iq' alanı eksik.")
                    self.data_queue.task_done()
                    continue
                try:
                    samples = np.array(samples_list, dtype=np.float64).view(np.complex128).squeeze()
                except (ValueError, TypeError) as e:
                    logger.error(f"IQ verisi NumPy array\e çevrilemedi: {e}. Veri: {str(samples_list)[:100]}...")
                    self.data_queue.task_done()
                    continue
                    
                sample_rate = float(sdr_data.get("sample_rate", DEFAULT_SDR_SAMPLE_RATE))
                center_freq = float(sdr_data.get("center_freq", 0))
                timestamp = float(sdr_data.get("timestamp", time.time()))

                # --- Ön İşleme Adımları --- 
                
                # 1. Gürültü Azaltma (Wavelet Denoising)
                filtered_samples = self._apply_wavelet_denoising(samples)
                if filtered_samples is None: # Hata oluştuysa orijinali kullan
                    filtered_samples = samples 
                
                # 2. Yankı Tespiti ve Derinlik Tahmini
                targets = self._find_and_analyze_reflections(filtered_samples, sample_rate)
                
                # 3. Clutter Filtreleme
                final_targets = self._filter_clutter(targets)

                # --- Sonuçları Yayınla --- 
                # Optimizasyon: Büyük numpy arrayleri yerine sadece hedefleri gönder
                output_message = {
                    "timestamp": timestamp,
                    "center_freq_hz": center_freq,
                    "sample_rate_hz": sample_rate,
                    "ground_type": self.current_ground_type.name,
                    "permittivity": self.current_permittivity,
                    # "filtered_samples_iq": [[s.real, s.imag] for s in filtered_samples], # Çok büyük olabilir, opsiyonel
                    "targets": final_targets # {time_index: {depth_m, amplitude, phase_change_rad, ...}}
                }
                self.output_socket.send_json(output_message)

                proc_time_ms = (time.perf_counter() - start_proc_time) * 1000
                logger.debug(f"Ön işleme tamamlandı: {len(final_targets)} hedef bulundu, Süre: {proc_time_ms:.2f} ms")
                self.data_queue.task_done()

            except Empty:
                if not self.running.is_set(): break # Kapatılıyorsa ve kuyruk boşsa çık
                continue # Kuyruk boşsa döngüye devam et
            except Exception as e:
                logger.error(f"Ön işleme hatası: {e}", exc_info=True)
                # Hata durumunda görevi tamamlandı say
                try: self.data_queue.task_done() 
                except ValueError: pass 
                time.sleep(0.1)
        logger.info("Ön işleme worker durdu.")

    def _apply_wavelet_denoising(self, samples: np.ndarray) -> Optional[np.ndarray]:
        """Wavelet dönüşümü kullanarak sinyaldeki gürültüyü azaltır."""
        if samples is None or len(samples) == 0:
            return samples
            
        try:
            # Reel ve sanal kısımları ayrı ayrı işle
            real_part = samples.real
            imag_part = samples.imag

            # Reel kısım
            coeffs_real = pywt.wavedec(real_part, self.wavelet_type, level=self.wavelet_level)
            # Gürültü seviyesini tahmin et (MAD)
            sigma_real = np.median(np.abs(coeffs_real[-1] - np.median(coeffs_real[-1]))) / 0.6745
            # Evrensel eşik değeri
            threshold_real = sigma_real * np.sqrt(2 * np.log(len(real_part))) if len(real_part) > 0 else 0
            # Eşikleme uygula (soft thresholding)
            new_coeffs_real = [coeffs_real[0]] + [pywt.threshold(c, threshold_real, mode='soft') for c in coeffs_real[1:]]
            # Sinyali yeniden oluştur
            denoised_real = pywt.waverec(new_coeffs_real, self.wavelet_type)

            # Sanal kısım (aynı işlemler)
            coeffs_imag = pywt.wavedec(imag_part, self.wavelet_type, level=self.wavelet_level)
            sigma_imag = np.median(np.abs(coeffs_imag[-1] - np.median(coeffs_imag[-1]))) / 0.6745
            threshold_imag = sigma_imag * np.sqrt(2 * np.log(len(imag_part))) if len(imag_part) > 0 else 0
            new_coeffs_imag = [coeffs_imag[0]] + [pywt.threshold(c, threshold_imag, mode='soft') for c in coeffs_imag[1:]]
            denoised_imag = pywt.waverec(new_coeffs_imag, self.wavelet_type)
            
            # Boyutları eşitle (waverec bazen orijinalden farklı uzunluk verebilir)
            min_len = min(len(denoised_real), len(denoised_imag), len(samples))
            denoised_samples = denoised_real[:min_len] + 1j * denoised_imag[:min_len]
            
            return denoised_samples
            
        except Exception as e:
            logger.warning(f"Wavelet denoising hatası: {e}. Orijinal sinyal döndürülüyor.")
            return None # Hata durumunda None döndür

    def _find_and_analyze_reflections(self, samples: np.ndarray, sample_rate: float) -> Dict[int, Dict[str, Any]]:
        """Filtrelenmiş sinyaldeki yankıları bulur, derinliği tahmin eder ve özelliklerini çıkarır."""
        targets = {}
        if samples is None or len(samples) == 0 or sample_rate <= 0:
            return targets
            
        n_samples = len(samples)
        time_axis_s = np.arange(n_samples) / sample_rate

        # Kalibrasyondan alınan güncel dielektrik sabiti
        with self.calibration_lock:
            permittivity = self.current_permittivity
        # Hız hesaplama (ışık hızı / sqrt(epsilon_r))
        velocity_mps = speed_of_light / np.sqrt(permittivity) if permittivity > 0 else speed_of_light
        
        # Zarf (envelope) hesaplama
        envelope = np.abs(samples)
        
        # Pik tespiti için parametreler
        # Gürültü seviyesi tahmini (daha sağlam bir yöntem kullanılabilir)
        noise_level_est = np.median(envelope) * 1.5 + np.std(envelope) * 0.5 
        min_peak_height = noise_level_est * 1.5 # Eşik değeri
        min_distance_samples = max(1, int(self.min_peak_distance_ms / 1000 * sample_rate))
        surface_skip_samples = max(0, int(self.surface_skip_ms / 1000 * sample_rate))
        
        try:
            peaks, properties = signal.find_peaks(envelope, height=min_peak_height, distance=min_distance_samples)
        except Exception as e:
            logger.warning(f"Pik tespiti sırasında hata: {e}")
            return {}

        # Yüzey yansımasını ve çok yakın pikleri filtrele
        valid_peaks = peaks[peaks > surface_skip_samples]

        if len(valid_peaks) == 0:
            return {}

        # Faz hesaplama (tüm sinyal için bir kere)
        phases_rad = np.angle(samples)
        # Fazı açma (unwrap) - sadece piklerin etrafında yapılabilir ama şimdilik tümü
        # unwrapped_phases_rad = np.unwrap(phases_rad)

        # Piklerin özelliklerini çıkar
        for peak_index in valid_peaks:
            time_delay_s = time_axis_s[peak_index]
            # Derinlik = (hız * zaman) / 2 (gidiş-dönüş)
            depth_m = (velocity_mps * time_delay_s) / 2.0
            
            amplitude = envelope[peak_index]
            phase_at_peak_rad = phases_rad[peak_index]
            
            # Faz değişimi (önceki pike göre veya başlangıca göre?)
            # Şimdilik sadece pik noktasındaki fazı alalım.
            # Daha gelişmiş: Pikin etrafındaki faz değişimini analiz et.
            phase_change_rad = 0.0 # Placeholder
            
            # Pik genişliği gibi ek özellikler
            peak_width_samples = properties["widths"][np.where(peaks == peak_index)][0] if "widths" in properties else 0
            peak_width_s = peak_width_samples / sample_rate if sample_rate > 0 else 0

            targets[int(peak_index)] = {
                "depth_m": round(depth_m, 3),
                "amplitude": float(amplitude),
                "phase_rad": float(phase_at_peak_rad),
                "phase_change_rad": float(phase_change_rad), # Geliştirilmeli
                "time_delay_s": float(time_delay_s),
                "peak_width_s": float(peak_width_s)
            }
            
        # logger.debug(f"{len(targets)} potansiyel hedef bulundu.")
        return targets

    def _filter_clutter(self, targets: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Tespit edilen hedefler arasından clutter (istenmeyen yansımalar) filtreler."""
        if not targets:
            return {}
            
        filtered_targets = {} 
        # Basit filtreleme kuralları:
        # 1. Çok düşük genlikli hedefleri at
        amplitudes = np.array([t["amplitude"] for t in targets.values()])
        if len(amplitudes) == 0: return {}
        max_amplitude = np.max(amplitudes) if len(amplitudes) > 0 else 1.0
        min_amplitude_threshold = max_amplitude * (1.0 - self.clutter_filter_threshold) * 0.1 # Daha dinamik eşik
        
        # 2. Çok sığ hedefleri at (yüzey skip zaten yapıldı ama ek kontrol)
        min_depth_m = 0.05 # 5 cm

        for index, target_data in targets.items():
            if target_data["amplitude"] < min_amplitude_threshold:
                # logger.debug(f"Clutter (Düşük Genlik): Index={index}, Amp={target_data[\"amplitude\"]:.3f}")
                continue
            if target_data["depth_m"] < min_depth_m:
                # logger.debug(f"Clutter (Çok Sığ): Index={index}, Derinlik={target_data[\"depth_m\"]:.3f}m")
                continue
                
            # TODO: Daha gelişmiş filtreleme eklenebilir:
            # - Beklenen yansıma desenine uymayanlar (örn. çok geniş pikler)
            # - Sürekli tekrarlayan sabit derinlikteki yansımalar (anten yansıması vb.)
            
            filtered_targets[index] = target_data
            
        # logger.debug(f"Clutter filtreleme sonrası {len(filtered_targets)} hedef kaldı.")
        return filtered_targets

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
        params = message.get("params", {})
        logger.info(f"Kontrol komutu alındı: {command}", params=params)
        
        try:
            if command == "get_status":
                with self.calibration_lock:
                    status_data = {
                        "running": self.running.is_set(),
                        "ground_type": self.current_ground_type.name,
                        "permittivity": self.current_permittivity,
                        "noise_db": self.background_noise_db,
                        "queue_size": self.data_queue.qsize()
                    }
                return {"status": "ok", "data": status_data}
            
            elif command == "set_parameter":
                param_name = params.get("name")
                param_value = params.get("value")
                if param_name == "wavelet_type":
                    # TODO: Desteklenen wavelet'leri kontrol et
                    self.wavelet_type = str(param_value)
                    logger.info(f"Wavelet tipi ayarlandı: {self.wavelet_type}")
                    return {"status": "ok"}
                elif param_name == "clutter_threshold":
                    try:
                        threshold = float(param_value)
                        if 0.0 <= threshold <= 1.0:
                            self.clutter_filter_threshold = threshold
                            logger.info(f"Clutter eşiği ayarlandı: {self.clutter_filter_threshold}")
                            return {"status": "ok"}
                        else:
                            return {"status": "error", "message": "Clutter eşiği 0-1 arasında olmalı."}
                    except (ValueError, TypeError):
                         return {"status": "error", "message": "Geçersiz clutter eşik değeri."}
                # Diğer parametreler eklenebilir
                else:
                    return {"status": "error", "message": f"Bilinmeyen veya ayarlanamayan parametre: {param_name}"}
            
            else:
                return {"status": "error", "message": f"Bilinmeyen komut: {command}"}
                
        except Exception as e:
             logger.error(f"Kontrol komutu işlenirken hata: {e}", command=command, exc_info=True)
             return {"status": "error", "message": f"Komut işlenirken hata: {e}"}

    def is_alive(self) -> bool:
        """Modülün çalışıp çalışmadığını kontrol eder."""
        proc_thread = self.threads.get("processing")
        return self.running.is_set() and (proc_thread is not None and proc_thread.is_alive())

# --- Komut Satırı Arayüzü (Bağımsız Çalıştırma İçin) --- 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal Preprocessing Module")
    parser.add_argument("--config", type=str, default=None, help="Path to system JSON configuration file")

    args = parser.parse_args()

    # Yapılandırmayı yükle
    config = {}
    if args.config:
        try:
            with open(args.config, "r") as f:
                file_config = json.load(f)
                config = file_config.get("module_configs", {}).get("preprocessing", {})
                logger.info(f"Yapılandırma dosyası yüklendi: {args.config}")
        except FileNotFoundError:
            logger.error(f"Yapılandırma dosyası bulunamadı: {args.config}. Varsayılanlar kullanılıyor.")
        except json.JSONDecodeError as e:
            logger.error(f"Yapılandırma dosyası okunamadı (JSON Hatası): {e}. Varsayılanlar kullanılıyor.")
    else:
        logger.warning("Yapılandırma dosyası belirtilmedi, varsayılan ayarlar kullanılıyor.")

    # Modülü başlat
    preprocessor = SignalPreprocessor(config)

    def shutdown_handler(signum, frame):
        print("\nKapatma sinyali alındı, Ön İşleme Modülü durduruluyor...")
        preprocessor.stop()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    if preprocessor.start():
        logger.info("Ön İşleme Modülü çalışıyor. Durdurmak için CTRL+C basın.")
        # Ana thread\in çalışmasını bekle
        while preprocessor.running.is_set():
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                break # Sinyal işleyici zaten çağrılacak
    else:
        logger.critical("Ön İşleme Modülü başlatılamadı.")

    logger.info("Ön İşleme Modülü programı sonlandı.")

