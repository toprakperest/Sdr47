#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sdr_receiver.py - SDR Alıcı Modülü

Version: 3.1
Author: AI Assistant (Manus)
Last Updated: 2025-04-30

SDR donanımından (örn. HamGeek Zynq7020+AD9363) veri alımını yönetir.
SoapySDR kütüphanesini kullanarak donanımı yapılandırır, veri akışını başlatır,
temel sinyal işleme (filtreleme, kazanç kontrolü) uygular ve işlenmiş verileri
ZMQ üzerinden diğer modüllere (örn. Preprocessing) yayınlar.

Özellikler:
- SoapySDR ile donanım soyutlama.
- Yapılandırılabilir örnekleme hızı, merkez frekans, bant genişliği, kazanç.
- İsteğe bağlı LNA kontrolü.
- İsteğe bağlı uyarlanabilir kazanç kontrolü (AGC).
- İsteğe bağlı çift antenli gürültü bastırma (referans anten ile).
- Çoklu frekans tarama yeteneği.
- Temel DSP: Bandpass ve Notch filtreleme.
- ZMQ üzerinden veri yayını ve kontrol arayüzü.
- Kapsamlı hata yönetimi ve kurtarma.
- Test modu (SoapySDR olmadan simülasyon).
- Performans metrikleri.
"""

import os
import sys
import time
import argparse
import logging
import logging.handlers
import json
import numpy as np
import zmq
from threading import Thread, Event, Lock
from queue import Queue, Full, Empty
import scipy.signal as signal
from datetime import datetime
from collections import deque
import psutil
from typing import Dict, Any, List, Optional, Tuple
import io
# Yerel modüller
from config import (
    HARDWARE_SPECS, PORT_SDR_DATA, PORT_SDR_CONTROL,
    DEFAULT_SDR_SAMPLE_RATE, DEFAULT_SDR_BANDWIDTH, DEFAULT_SDR_GAIN,
    DEFAULT_SCAN_FREQ_START_MHZ, DEFAULT_SCAN_FREQ_END_MHZ
)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# SoapySDR importu (opsiyonel)
try:
    import SoapySDR
    from soapy import SOAPY_SDR_TIMEOUT, SOAPY_SDR_OVERFLOW, SOAPY_SDR_UNDERFLOW
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_TX, SOAPY_SDR_CF32
    SOAPY_AVAILABLE = True
except ImportError:
    SOAPY_AVAILABLE = False
    print("UYARI: SoapySDR kütüphanesi bulunamadi. SDR Alici test modunda calisacak.")

# Logging yapılandırması
logger = logging.getLogger("SDR_RECEIVER")
# Ana logger yapılandırması main.py tarafından yapılır.
# Eğer bu modül bağımsız çalıştırılacaksa, aşağıdaki gibi temel yapılandırma eklenebilir:
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s",
        handlers=[logging.StreamHandler()]  # Sadece konsola yaz
    )

# Sabitler
RX_BUFFER_SIZE_SAMPLES = 8192 # SDR\den okunacak örnek sayısı
DEFAULT_TIMEOUT_MS = 100 # SDR okuma zaman aşımı (ms)

class SDRReceiver:
    """SoapySDR kullanarak SDR donanımını yönetir ve veri alır."""

    def __init__(self, config: Dict[str, Any]):
        """
        SDR Alıcıyı başlatır.

        Args:
            config (Dict[str, Any]): Sistem yapılandırma dosyasından gelen modül ayarları.
        """
        self.config = config
        self.running = Event()
        self.shutdown_event = Event() # Graceful shutdown için
        self.lock = Lock() # Donanım erişimi için kilit

        # Donanım Ayarları
        self.device_args = config.get("device_args", HARDWARE_SPECS["sdr"].get("default_driver_args", ""))
        self.sample_rate = float(config.get("sample_rate", DEFAULT_SDR_SAMPLE_RATE))
        self.bandwidth = float(config.get("bandwidth", DEFAULT_SDR_BANDWIDTH))
        self.default_gain = float(config.get("default_gain", DEFAULT_SDR_GAIN))
        self.adaptive_gain_enabled = bool(config.get("adaptive_gain", True))
        self.lna_enabled = bool(config.get("lna_enabled", True))
        self.lna_gain = float(config.get("lna_gain", HARDWARE_SPECS["lna"].get("gain_db", 20)))

        # Anten Ayarları
        self.antenna_rx_main = config.get("antenna_rx_main", HARDWARE_SPECS["antennas"]["rx1"].name)
        self.antenna_rx_noise = config.get("antenna_rx_noise", HARDWARE_SPECS["antennas"]["rx2"].name)
        self.antenna_tx = config.get("antenna_tx", HARDWARE_SPECS["antennas"]["tx1"].name)
        self.channel_rx_main = int(config.get("channel_rx_main", 0))
        self.channel_rx_noise = int(config.get("channel_rx_noise", 1))
        self.channel_tx = int(config.get("channel_tx", 0))

        # Frekans ve Tarama Ayarları
        default_freq = float(config.get("default_frequency_mhz", DEFAULT_SCAN_FREQ_START_MHZ)) * 1e6
        self.current_freq = default_freq
        scan_start = float(config.get("scan_freq_start_mhz", DEFAULT_SCAN_FREQ_START_MHZ)) * 1e6
        scan_end = float(config.get("scan_freq_end_mhz", DEFAULT_SCAN_FREQ_END_MHZ)) * 1e6
        scan_step = float(config.get("scan_freq_step_mhz", 100)) * 1e6
        self.scan_frequencies = list(np.arange(scan_start, scan_end + scan_step, scan_step))
        if not self.scan_frequencies:
            self.scan_frequencies = [default_freq]
        self.scan_mode = config.get("scan_mode", "single") # "single" veya "sweep"
        self.current_scan_index = 0
        self.sweep_dwell_time_sec = float(config.get("sweep_dwell_time_sec", 0.5))

        # DSP Ayarları
        dsp_config = config.get("dsp", {})
        self.bandpass_low_hz = float(dsp_config.get("bandpass_low_hz", 300e6))
        self.bandpass_high_hz = float(dsp_config.get("bandpass_high_hz", 3000e6))
        self.notch_freq_hz = float(dsp_config.get("notch_freq_hz", 0))  # 0=disable
        self.notch_width_hz = float(dsp_config.get("notch_width_hz", 100e3))
        self.noise_suppression_enabled = bool(dsp_config.get("noise_suppression_enabled", True))
        self.noise_filter_taps = int(dsp_config.get("noise_filter_taps", 128))
        self.noise_mu = float(dsp_config.get("noise_mu", 0.01))  # LMS adaptasyon katsayısı

        # Veri Kuyrukları
        queue_size = int(config.get("queue_size", 100))
        self.raw_queue_main = Queue(maxsize=queue_size)
        self.raw_queue_noise = Queue(maxsize=queue_size)
        self.processed_queue = Queue(maxsize=queue_size)

        # Threadler
        self.threads: Dict[str, Optional[Thread]] = {}

    class SDRInterface:
        def __init__(self, config: dict):
            """ZMQ iletişimini başlat"""
            # ZMQ İletişim
            self.context = zmq.Context()

            try:
                # Port numaralarını önce al
                output_port = int(config.get("output_port", PORT_SDR_DATA))
                control_port = int(config.get("control_port", PORT_SDR_CONTROL))

                # Soketleri oluştur ve bağla
                self.data_socket = self.context.socket(zmq.PUB)
                self.data_socket.bind(f"tcp://*:{output_port}")

                self.control_socket = self.context.socket(zmq.REP)
                self.control_socket.bind(f"tcp://*:{control_port}")
                self.control_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 saniye zaman aşımı

            except zmq.ZMQError as e:
                self._cleanup()
                raise RuntimeError(f"ZMQ başlatma hatası: {e}")

        def _cleanup(self):
            """Kaynakları temizle"""
            if hasattr(self, 'data_socket'):
                self.data_socket.close()
            if hasattr(self, 'control_socket'):
                self.control_socket.close()
            self.context.term()

        def __del__(self):
            self._cleanup()

        class SDRReceiver:  # Sınıf tanımı eklenmeli
            def __init__(self, config: dict):
                """SDR Alıcı sınıfı başlatıcı"""
                # SDR Donanım Nesneleri
                self.sdr: Optional[SoapySDR.Device] = None
                self.rx_stream_main: Optional[int] = None
                self.rx_stream_noise: Optional[int] = None
                self.tx_stream: Optional[int] = None
                self.test_mode = bool(config.get("test_mode", False)) or not SOAPY_AVAILABLE
                self.test_signal_freq = float(config.get("test_signal_freq_hz", 10e6))
                self.test_signal_amplitude = float(config.get("test_signal_amplitude", 0.5))

                # Varsayılan değerler (eksik olanlar)
                self.default_gain = float(config.get("default_gain", 20.0))
                self.current_freq = float(config.get("start_freq_hz", 100e6))
                self.noise_filter_taps = int(config.get("noise_filter_taps", 64))

                # Performans Metrikleri
                self.metrics: Dict[str, Any] = {
                    "start_time": time.time(),
                    "samples_processed": 0,
                    "dropped_packets_main": 0,
                    "dropped_packets_noise": 0,
                    "processing_time_avg_ms": 0.0,
                    "snr_estimate_db": 0.0,
                    "current_gain_main": self.default_gain,
                    "current_gain_noise": self.default_gain,
                    "current_freq_mhz": self.current_freq / 1e6,
                    "hardware_info": {},
                    "status": "initialized"
                }
                self._processing_times = deque(maxlen=100)  # Son 100 işleme süresi

                # DSP Filtreleri
                self.bandpass_taps: Optional[np.ndarray] = None
                self.notch_b: Optional[np.ndarray] = None
                self.notch_a: Optional[np.ndarray] = None
                self.noise_weights: np.ndarray = np.zeros(self.noise_filter_taps, dtype=np.complex64)
                self._init_filters()

                logger.info(f"SDR Alıcı başlatıldı (Test Modu: {self.test_mode})")

    def _init_filters(self):
        """Yapılandırmaya göre DSP filtrelerini başlatır."""
        # Bandpass Filtre
        try:
            if 0 < self.bandpass_low_hz < self.bandpass_high_hz < self.sample_rate / 2:
                nyq = 0.5 * self.sample_rate
                low = self.bandpass_low_hz / nyq
                high = self.bandpass_high_hz / nyq
                # FIR filtre katsayıları (örnek: 101 taps)
                self.bandpass_taps = signal.firwin(101, [low, high], pass_zero=False, fs=self.sample_rate)
                logger.info(f"Bandpass filtresi oluşturuldu: {self.bandpass_low_hz/1e6:.1f}-{self.bandpass_high_hz/1e6:.1f} MHz")
            else:
                self.bandpass_taps = None
                logger.warning("Geçersiz bandpass frekansları veya örnekleme hızı, filtre devre dışı.")
        except ValueError as e:
            logger.error(f"Bandpass filtresi oluşturulamadı: {e}")
            self.bandpass_taps = None

        # Notch Filtre
        try:
            if 0 < self.notch_freq_hz < self.sample_rate / 2 and self.notch_width_hz > 0:
                nyq = 0.5 * self.sample_rate
                freq_norm = self.notch_freq_hz / nyq
                q_factor = self.notch_freq_hz / self.notch_width_hz
                self.notch_b, self.notch_a = signal.iirnotch(freq_norm, q_factor)
                logger.info(f"Notch filtresi oluşturuldu: {self.notch_freq_hz/1e6:.1f} MHz, Q={q_factor:.1f}")
            else:
                self.notch_b, self.notch_a = None, None
                if self.notch_freq_hz > 0: # Sadece frekans belirtilmişse uyar
                    logger.warning("Geçersiz notch frekansı veya genişliği, filtre devre dışı.")
        except ValueError as e:
            logger.error(f"Notch filtresi oluşturulamadı: {e}")
            self.notch_b, self.notch_a = None, None

    def start(self) -> bool:
        """SDR alıcıyı ve ilgili thread\leri başlatır."""
        if self.running.is_set():
            logger.warning("SDR Alıcı zaten çalışıyor.")
            return True

        if not self.test_mode:
            if not self._init_sdr():
                logger.critical("SDR donanımı başlatılamadı! Sistem başlatılamıyor.")
                self.metrics["status"] = "error_sdr_init"
                return False

        self.running.set()
        self.shutdown_event.clear()

        # Worker thread\lerini başlat
        self.threads["acquisition"] = Thread(target=self._acquisition_worker, name="Acquisition")
        self.threads["processing"] = Thread(target=self._processing_worker, name="Processing")
        self.threads["control"] = Thread(target=self._control_worker, name="Control")
        # self.threads["monitor"] = Thread(target=self._monitor_worker, name="Monitor") # Ana yönetici yapabilir

        for name, thread in self.threads.items():
            if thread:
                thread.daemon = True
                thread.start()
                logger.info(f"{name} thread\i başlatıldı.")

        self.metrics["status"] = "running"
        logger.info(f"SDR Alıcı başlatıldı (Frekans: {self.current_freq / 1e6:.2f} MHz, Örnekleme: {self.sample_rate / 1e6:.2f} MS/s)")
        return True

    def stop(self):
        """SDR alıcıyı ve thread\leri güvenli bir şekilde durdurur."""
        if not self.running.is_set() and self.shutdown_event.is_set():
            logger.info("SDR Alıcı zaten durdurulmuş veya durduruluyor.")
            return

        logger.info("SDR Alıcı durduruluyor...")
        self.running.clear()
        self.shutdown_event.set()

        # Thread\leri durdur
        for name, thread in self.threads.items():
            if thread and thread.is_alive():
                try:
                    thread.join(timeout=2.0)
                    if thread.is_alive():
                        logger.warning(f"{name} thread\i zamanında durmadı.")
                    else:
                        logger.info(f"{name} thread\i durduruldu.")
                except Exception as e:
                    logger.error(f"{name} thread\ini durdururken hata: {e}")
            self.threads[name] = None # Thread nesnesini temizle

        # SDR kaynaklarını kapat
        if not self.test_mode and self.sdr:
            self._close_sdr()

        # ZMQ soketlerini kapat
        try:
            self.data_socket.close(linger=0)
            self.control_socket.close(linger=0)
            self.context.term()
            logger.info("ZMQ soketleri kapatıldı.")
        except Exception as e:
            logger.error(f"ZMQ kapatılırken hata: {e}")

        self.metrics["status"] = "stopped"
        self._log_metrics() # Son metrikleri logla
        logger.info("SDR Alıcı başarıyla durduruldu.")

    def _init_sdr(self) -> bool:
        """SoapySDR kullanarak SDR donanımını başlatır ve yapılandırır."""
        if not SOAPY_AVAILABLE:
            logger.error("SoapySDR kütüphanesi yüklenemediği için SDR başlatılamıyor.")
            return False
        
        with self.lock:
            try:
                logger.info(f"SDR cihazı aranıyor: args=\"{self.device_args}\"")
                devices = SoapySDR.Device.enumerate(self.device_args)
                if not devices:
                    logger.error(f"Belirtilen argümanlarla SDR cihazı bulunamadı: {self.device_args}")
                    return False
                logger.info(f"Bulunan cihazlar: {devices}")
                
                self.sdr = SoapySDR.Device(self.device_args)
                logger.info(f"SDR cihazı açıldı: {self.sdr.getHardwareInfo()}")
                self.metrics["hardware_info"] = self.sdr.getHardwareInfo()

                # --- Ana RX Kanalı Yapılandırması --- 
                logger.info(f"Ana RX yapılandırılıyor (Kanal: {self.channel_rx_main}, Anten: {self.antenna_rx_main})")
                self.sdr.setSampleRate(SOAPY_SDR_RX, self.channel_rx_main, self.sample_rate)
                self.sdr.setFrequency(SOAPY_SDR_RX, self.channel_rx_main, self.current_freq)
                if self.bandwidth > 0:
                    self.sdr.setBandwidth(SOAPY_SDR_RX, self.channel_rx_main, self.bandwidth)
                if self.antenna_rx_main:
                    self.sdr.setAntenna(SOAPY_SDR_RX, self.channel_rx_main, self.antenna_rx_main)
                self.sdr.setGain(SOAPY_SDR_RX, self.channel_rx_main, self.default_gain)
                if self.lna_enabled and "LNA" in self.sdr.listGains(SOAPY_SDR_RX, self.channel_rx_main):
                    self.sdr.setGain(SOAPY_SDR_RX, self.channel_rx_main, "LNA", self.lna_gain)
                    logger.info(f"LNA etkinleştirildi (Kazanç: {self.lna_gain} dB)")
                if self.adaptive_gain_enabled and self.sdr.hasGainMode(SOAPY_SDR_RX, self.channel_rx_main):
                    self.sdr.setGainMode(SOAPY_SDR_RX, self.channel_rx_main, True)
                    logger.info("Otomatik Kazanç Kontrolü (AGC) etkinleştirildi.")
                else:
                    self.sdr.setGainMode(SOAPY_SDR_RX, self.channel_rx_main, False)

                # --- Gürültü RX Kanalı Yapılandırması --- 
                if self.noise_suppression_enabled:
                    logger.info(f"Gürültü RX yapılandırılıyor (Kanal: {self.channel_rx_noise}, Anten: {self.antenna_rx_noise})")
                    if self.channel_rx_main == self.channel_rx_noise:
                        logger.warning("Ana ve Gürültü RX kanalları aynı. Eş zamanlı alım desteklenmeyebilir.")
                    
                    self.sdr.setSampleRate(SOAPY_SDR_RX, self.channel_rx_noise, self.sample_rate)
                    self.sdr.setFrequency(SOAPY_SDR_RX, self.channel_rx_noise, self.current_freq)
                    if self.bandwidth > 0:
                        self.sdr.setBandwidth(SOAPY_SDR_RX, self.channel_rx_noise, self.bandwidth)
                    if self.antenna_rx_noise:
                        self.sdr.setAntenna(SOAPY_SDR_RX, self.channel_rx_noise, self.antenna_rx_noise)
                    self.sdr.setGain(SOAPY_SDR_RX, self.channel_rx_noise, self.default_gain) # Şimdilik aynı kazanç
                    self.sdr.setGainMode(SOAPY_SDR_RX, self.channel_rx_noise, False) # Gürültü kanalında AGC genellikle istenmez

                # --- TX Kanalı Yapılandırması (Placeholder) --- 
                # logger.info(f"TX yapılandırılıyor (Kanal: {self.channel_tx}, Anten: {self.antenna_tx})")
                # self.sdr.setSampleRate(SOAPY_SDR_TX, self.channel_tx, self.sample_rate)
                # ... (diğer TX ayarları)

                # --- Akışları Ayarla --- 
                rx_channels = [self.channel_rx_main]
                if self.noise_suppression_enabled:
                    rx_channels.append(self.channel_rx_noise)
                
                # Çok kanallı akışı dene
                try:
                    logger.info(f"Çok kanallı RX akışı deneniyor: Kanallar={rx_channels}")
                    self.rx_stream_main = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, rx_channels)
                    logger.info("Çok kanallı RX akışı başarılı.")
                    if len(rx_channels) > 1:
                        self.rx_stream_noise = self.rx_stream_main # Aynı akış nesnesi
                except Exception as multi_stream_err:
                    logger.warning(f"Çok kanallı RX akışı başarısız: {multi_stream_err}. Tekli akışlara dönülüyor.")
                    self.rx_stream_main = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [self.channel_rx_main])
                    if self.noise_suppression_enabled:
                        try:
                            self.rx_stream_noise = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [self.channel_rx_noise])
                        except Exception as noise_stream_err:
                            logger.error(f"Gürültü RX akışı ayarlanamadı: {noise_stream_err}. Gürültü bastırma devre dışı.")
                            self.noise_suppression_enabled = False
                            self.rx_stream_noise = None
                    else:
                        self.rx_stream_noise = None

                # TX Akışı (Placeholder)
                # self.tx_stream = self.sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, [self.channel_tx])

                # Akışları etkinleştir
                if self.rx_stream_main:
                    self.sdr.activateStream(self.rx_stream_main)
                    logger.info("Ana RX akışı etkinleştirildi.")
                if self.rx_stream_noise and self.rx_stream_noise != self.rx_stream_main:
                    self.sdr.activateStream(self.rx_stream_noise)
                    logger.info("Gürültü RX akışı etkinleştirildi.")
                # if self.tx_stream:
                #     self.sdr.activateStream(self.tx_stream)

                logger.info("SDR donanımı başarıyla başlatıldı ve yapılandırıldı.")
                return True

            except Exception as e:
                logger.critical(f"SDR başlatma sırasında kritik hata: {e}", exc_info=True)
                if self.sdr:
                    try: SoapySDR.Device.unmake(self.sdr) 
                    except: pass
                    self.sdr = None
                return False

    def _close_sdr(self):
        """SDR akışlarını ve cihazını güvenli bir şekilde kapatır."""
        with self.lock:
            if not self.sdr:
                return
            logger.info("SDR kaynakları kapatılıyor...")
            try:
                # Akışları devre dışı bırak ve kapat
                if self.rx_stream_main:
                    try: self.sdr.deactivateStream(self.rx_stream_main) 
                    except: pass
                    try: self.sdr.closeStream(self.rx_stream_main) 
                    except: pass
                    self.rx_stream_main = None
                # Eğer gürültü akışı ayrıysa onu da kapat
                if self.rx_stream_noise and self.rx_stream_noise != self.rx_stream_main:
                    try: self.sdr.deactivateStream(self.rx_stream_noise) 
                    except: pass
                    try: self.sdr.closeStream(self.rx_stream_noise) 
                    except: pass
                    self.rx_stream_noise = None
                # TX akışını kapat (Placeholder)
                # if self.tx_stream:
                #     try: self.sdr.deactivateStream(self.tx_stream) 
                #     except: pass
                #     try: self.sdr.closeStream(self.tx_stream) 
                #     except: pass
                #     self.tx_stream = None
                
                # Cihazı serbest bırak
                SoapySDR.Device.unmake(self.sdr)
                self.sdr = None
                logger.info("SDR cihazı başarıyla kapatıldı.")
            except Exception as e:
                logger.error(f"SDR kapatılırken hata: {e}", exc_info=True)
                self.sdr = None # Hata olsa bile referansı temizle

    def _acquisition_worker(self):
        """SDR\den sürekli veri okur ve ham veri kuyruklarına yazar."""
        logger.info("Veri alım thread\i başlatıldı.")
        
        # Bufferları oluştur
        rx_buffer_main = np.zeros(RX_BUFFER_SIZE_SAMPLES, dtype=np.complex64)
        rx_buffer_noise = np.zeros(RX_BUFFER_SIZE_SAMPLES, dtype=np.complex64) if self.noise_suppression_enabled else None
        buffers = [rx_buffer_main]
        if rx_buffer_noise is not None: buffers.append(rx_buffer_noise)
        
        stream_handle = self.rx_stream_main # Çok kanallı veya tekli ana akış
        is_multi_channel = self.noise_suppression_enabled and self.rx_stream_noise == self.rx_stream_main
        
        while self.running.is_set():
            if self.test_mode:
                # --- Test Modu --- 
                sim_time = time.time()
                t = np.arange(RX_BUFFER_SIZE_SAMPLES) / self.sample_rate
                signal_main = self.test_signal_amplitude * np.exp(1j * 2 * np.pi * self.test_signal_freq * t)
                noise_main = (np.random.randn(RX_BUFFER_SIZE_SAMPLES) + 1j * np.random.randn(RX_BUFFER_SIZE_SAMPLES)) * 0.1
                rx_buffer_main[:] = signal_main + noise_main
                read_len = RX_BUFFER_SIZE_SAMPLES
                ret_code = read_len
                flags = 0
                timestamp_ns = int(sim_time * 1e9)
                
                if rx_buffer_noise is not None:
                    noise_ref = (np.random.randn(RX_BUFFER_SIZE_SAMPLES) + 1j * np.random.randn(RX_BUFFER_SIZE_SAMPLES)) * 0.15
                    rx_buffer_noise[:] = noise_ref
                
                time.sleep(RX_BUFFER_SIZE_SAMPLES / self.sample_rate * 0.8) # Gerçek zamanlı gibi davran
            else:
                # --- Gerçek SDR Modu --- 
                if not self.sdr or not stream_handle:
                    logger.error("SDR cihazı veya akışı mevcut değil, alım durduruldu.")
                    time.sleep(1)
                    continue
                
                try:
                    # Veriyi oku
                    if is_multi_channel:
                        ret = self.sdr.readStream(stream_handle, buffers, RX_BUFFER_SIZE_SAMPLES, timeoutUs=int(DEFAULT_TIMEOUT_MS * 1000))
                        read_len = ret.ret # Okunan örnek sayısı
                        flags = ret.flags
                        timestamp_ns = ret.timeNs
                    else:
                        # Ana kanalı oku
                        ret_main = self.sdr.readStream(self.rx_stream_main, [rx_buffer_main], RX_BUFFER_SIZE_SAMPLES, timeoutUs=int(DEFAULT_TIMEOUT_MS * 1000))
                        read_len = ret_main.ret
                        flags = ret_main.flags
                        timestamp_ns = ret_main.timeNs
                        # Gürültü kanalını ayrı oku (eğer varsa)
                        if self.rx_stream_noise:
                             ret_noise = self.sdr.readStream(self.rx_stream_noise, [rx_buffer_noise], RX_BUFFER_SIZE_SAMPLES, timeoutUs=int(DEFAULT_TIMEOUT_MS * 1000))
                             # Gürültü okuma hatasını ayrıca ele al?
                             if ret_noise.ret <= 0:
                                 logger.warning(f"Gürültü kanalından veri okunamadı: {ret_noise.ret}")
                    
                    ret_code = read_len # Kolay kontrol için

                except Exception as e:
                    logger.error(f"SDR okuma hatası: {e}", exc_info=True)
                    # Hata durumunda SDR\yi yeniden başlatmayı dene?
                    self.metrics["status"] = "error_sdr_read"
                    time.sleep(1)
                    continue

            # Okuma sonucunu kontrol et
            if ret_code > 0:
                # Veriyi kuyruğa ekle
                data_packet = {
                    "timestamp": timestamp_ns / 1e9,
                    "samples_iq": rx_buffer_main[:read_len].copy(), # Kopyasını al
                    "flags": flags
                }
                try:
                    self.raw_queue_main.put_nowait(data_packet)
                except Full:
                    self.metrics["dropped_packets_main"] += 1
                    logger.warning("Ana RX ham veri kuyruğu dolu, paket atlandı!")

                if rx_buffer_noise is not None:
                    noise_packet = {
                        "timestamp": timestamp_ns / 1e9,
                        "samples_iq": rx_buffer_noise[:read_len].copy(),
                        "flags": flags
                    }
                    try:
                        self.raw_queue_noise.put_nowait(noise_packet)
                    except Full:
                        self.metrics["dropped_packets_noise"] += 1
                        logger.warning("Gürültü RX ham veri kuyruğu dolu, paket atlandı!")
                        
            elif ret_code == SOAPY_SDR_TIMEOUT:
                logger.debug("SDR okuma zaman aşımı.")
            elif ret_code == SOAPY_SDR_OVERFLOW:
                logger.warning("SDR buffer taşması (Overflow)!")
                self.metrics["dropped_packets_main"] += RX_BUFFER_SIZE_SAMPLES # Tahmini kayıp
            elif ret_code == SOAPY_SDR_UNDERFLOW:
                logger.warning("SDR buffer boşalması (Underflow)! (TX için geçerli)")
            elif ret_code < 0:
                logger.error(f"SDR okuma hatası kodu: {ret_code}")
                self.metrics["status"] = "error_sdr_read"
                # Ciddi hata, belki yeniden başlatma gerekir
                time.sleep(0.5)

                logger.info("Veri alım thread\i durduruldu.")

    def _processing_worker(self):
        """Ham veriyi işler (filtreleme, gürültü bastırma) ve yayınlar."""
        logger.info("Veri işleme thread\i başlatıldı.")
        last_scan_time = time.time()
        
        while self.running.is_set() or not self.raw_queue_main.empty(): # Kapatılırken kuyruğu boşalt
            try:
                # Ana sinyali al
                main_packet = self.raw_queue_main.get(timeout=0.1)
                samples_main = main_packet["samples_iq"]
                timestamp = main_packet["timestamp"]
                proc_start_time = time.perf_counter()

                # Gürültü sinyalini al (varsa ve zaman uyumluysa)
                samples_noise = None
                if self.noise_suppression_enabled and not self.raw_queue_noise.empty():
                    try:
                        noise_packet = self.raw_queue_noise.get_nowait()
                        # Zaman damgalarını kontrol et (çok farklıysa atla?)
                        if abs(noise_packet["timestamp"] - timestamp) < 0.01: # 10ms tolerans
                            samples_noise = noise_packet["samples_iq"]
                            # Uzunlukları eşitle
                            min_len = min(len(samples_main), len(samples_noise))
                            samples_main = samples_main[:min_len]
                            samples_noise = samples_noise[:min_len]
                        else:
                            logger.debug("Gürültü paketi zaman uyumsuz, atlandı.")
                            # Uyumsuz paketi geri koyma, kaybolsun
                    except Empty:
                        pass # Gürültü kuyruğu boşsa devam et

                # --- DSP Adımları --- 
                processed_samples = samples_main.copy()

                # 1. Bandpass Filtreleme
                if self.bandpass_taps is not None:
                    processed_samples = signal.lfilter(self.bandpass_taps, 1.0, processed_samples)

                # 2. Notch Filtreleme
                if self.notch_b is not None and self.notch_a is not None:
                    processed_samples = signal.lfilter(self.notch_b, self.notch_a, processed_samples)

                # 3. Gürültü Bastırma (LMS Algoritması)
                if self.noise_suppression_enabled and samples_noise is not None:
                    # LMS algoritmasını uygula
                    output_signal, self.noise_weights = self._apply_lms_filter(
                        processed_samples, samples_noise, self.noise_weights, self.noise_mu
                    )
                    processed_samples = output_signal
                
                # 4. Uyarlanabilir Kazanç (AGC) - Donanım yapmıyorsa burada yapılabilir
                # if not self.adaptive_gain_enabled:
                #     processed_samples = self._apply_agc(processed_samples)

                # --- Veriyi Yayınla --- 
                output_packet = {
                    "timestamp": timestamp,
                    "center_freq": self.current_freq,
                    "sample_rate": self.sample_rate,
                    "samples_iq": [[s.real, s.imag] for s in processed_samples], # JSON uyumlu format
                    "gain": self.metrics["current_gain_main"],
                    "snr_db": self.metrics["snr_estimate_db"] # Son tahmin
                }
                self.data_socket.send_json(output_packet)

                # Metrikleri güncelle
                self.metrics["samples_processed"] += len(processed_samples)
                proc_time_ms = (time.perf_counter() - proc_start_time) * 1000
                self._processing_times.append(proc_time_ms)
                self.metrics["processing_time_avg_ms"] = np.mean(self._processing_times)
                # SNR tahmini (basit)
                signal_power = np.mean(np.abs(processed_samples)**2)
                noise_power_est = np.mean(np.abs(samples_main - processed_samples)**2) if samples_noise is not None else signal_power / 100 # %1 gürültü varsayımı
                if noise_power_est > 1e-9:
                    self.metrics["snr_estimate_db"] = 10 * np.log10(signal_power / noise_power_est)
                
                # Frekans tarama mantığı
                current_time = time.time()
                if self.scan_mode == "sweep" and (current_time - last_scan_time) > self.sweep_dwell_time_sec:
                    self.current_scan_index = (self.current_scan_index + 1) % len(self.scan_frequencies)
                    new_freq = self.scan_frequencies[self.current_scan_index]
                    self.set_frequency(new_freq)
                    last_scan_time = current_time
                    logger.debug(f"Frekans tarama: {new_freq / 1e6:.2f} MHz")

            except Empty:
                if not self.running.is_set(): break # Kapatılıyorsa ve kuyruk boşsa çık
                time.sleep(0.01) # Kuyruk boşsa kısa süre bekle
            except Exception as e:
                logger.error(f"Veri işleme hatası: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("Veri işleme thread\i durduruldu.")


def _apply_lms_filter(self, desired_signal: np.ndarray, noise_signal: np.ndarray,
                      initial_weights: np.ndarray, mu: float) -> Tuple[np.ndarray, np.ndarray]:
    """Least Mean Squares (LMS) adaptif filtresini uygular."""
    n_samples = len(desired_signal)
    n_taps = len(initial_weights)
    weights = initial_weights.copy()
    output_signal = np.zeros(n_samples, dtype=np.complex64)

    # Gürültü sinyalini FIR filtre için uygun formata getir
    noise_padded = np.pad(noise_signal, (n_taps - 1, 0), "constant")
    noise_matrix = np.lib.stride_tricks.as_strided(
        noise_padded,
        shape=(n_samples, n_taps),
        strides=(noise_padded.strides[0], noise_padded.strides[0])
    )[:, ::-1]  # Ters çevirerek konvolüsyonu kolaylaştır

    for n in range(n_samples):
        # Filtre çıkışını hesapla: y[n] = w^H * x[n]
        noise_vector = noise_matrix[n]
        filtered_noise = np.dot(np.conj(weights), noise_vector)

        # Hata sinyalini hesapla: e[n] = d[n] - y[n]
        error = desired_signal[n] - filtered_noise
        output_signal[n] = error  # Hata sinyali temizlenmiş sinyaldir

        # Ağırlıkları güncelle: w[n+1] = w[n] + mu * e[n]^* * x[n]
        weights += mu * np.conj(error) * noise_vector

    return output_signal, weights

    def _control_worker(self):
        """ZMQ üzerinden gelen kontrol komutlarını işler."""
        logger.info("Kontrol thread\i başlatıldı.")
        while not self.shutdown_event.is_set():
            try:
                message = self.control_socket.recv_json()
                response = self._handle_control_command(message)
                self.control_socket.send_json(response)
            except zmq.Again:
                continue # Timeout, tekrar dene
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
            return {"status": "ok", "data": self.get_status()}
        elif command == "set_frequency":
            freq_mhz = message.get("params", {}).get("frequency_mhz")
            if freq_mhz is not None:
                if self.set_frequency(float(freq_mhz) * 1e6):
                    return {"status": "ok"}
                else:
                    return {"status": "error", "message": "Frekans ayarlanamadı."}
            else:
                return {"status": "error", "message": "Frekans parametresi eksik."}
        elif command == "set_gain":
            gain_db = message.get("params", {}).get("gain_db")
            if gain_db is not None:
                if self.set_gain(float(gain_db)):
                    return {"status": "ok"}
                else:
                    return {"status": "error", "message": "Kazanç ayarlanamadı."}
            else:
                return {"status": "error", "message": "Kazanç parametresi eksik."}
        elif command == "set_scan_mode":
            mode = message.get("params", {}).get("mode")  # "single" veya "sweep"
            if mode in ["single", "sweep"]:
                self.scan_mode = mode
                logger.info(f"Tarama modu ayarlandı: {mode}")
                return {"status": "ok"}
            else:
                return {"status": "error", "message": "Geçersiz tarama modu."}
            # Diğer komutlar eklenebilir (örn. set_sample_rate, set_bandwidth, enable_noise_suppression)
        else:
            return {"status": "error", "message": f"Bilinmeyen komut: {command}"}
    except Exception as e:
        logger.error(f"Kontrol komutu işlenirken hata: {e}", command=command)
        return {"status": "error", "message": f"Komut işlenirken hata: {e}"}

    def set_frequency(self, freq_hz: float) -> bool:
        """SDR merkez frekansını ayarlar."""
        with self.lock:
            if not self.test_mode and self.sdr:
                try:
                    self.sdr.setFrequency(SOAPY_SDR_RX, self.channel_rx_main, freq_hz)
                    if self.noise_suppression_enabled:
                        self.sdr.setFrequency(SOAPY_SDR_RX, self.channel_rx_noise, freq_hz)
                    # TX frekansını da ayarla (Placeholder)
                    # self.sdr.setFrequency(SOAPY_SDR_TX, self.channel_tx, freq_hz)
                    self.current_freq = freq_hz
                    self.metrics["current_freq_mhz"] = freq_hz / 1e6
                    logger.info(f"Frekans ayarlandı: {freq_hz / 1e6:.2f} MHz")
                    return True
                except Exception as e:
                    logger.error(f"Frekans ayarlanamadı ({freq_hz / 1e6:.2f} MHz): {e}")
                    return False
            elif self.test_mode:
                 self.current_freq = freq_hz
                 self.metrics["current_freq_mhz"] = freq_hz / 1e6
                 logger.info(f"Test modu frekansı ayarlandı: {freq_hz / 1e6:.2f} MHz")
                 return True
            else:
                logger.error("Frekans ayarlanamadı: SDR başlatılmamış.")
                return False


def set_gain(self, gain_db: float) -> bool:
    """SDR kazancını ayarlar."""
    with self.lock:
        if not self.test_mode and self.sdr:
            try:
                # AGC açıksa kapat
                if self.adaptive_gain_enabled and self.sdr.hasGainMode(SOAPY_SDR_RX, self.channel_rx_main):
                    self.sdr.setGainMode(SOAPY_SDR_RX, self.channel_rx_main, False)
                    logger.info("Manuel kazanç ayarı için AGC kapatıldı.")
                    self.adaptive_gain_enabled = False  # Durumu güncelle

                self.sdr.setGain(SOAPY_SDR_RX, self.channel_rx_main, gain_db)
                self.metrics["current_gain_main"] = gain_db
                # Gürültü kanalının kazancını da ayarla?
                if self.noise_suppression_enabled:
                    self.sdr.setGain(SOAPY_SDR_RX, self.channel_rx_noise, gain_db)
                    self.metrics["current_gain_noise"] = gain_db

                logger.info(f"Kazanç ayarlandı: {gain_db} dB")
                return True
            except Exception as e:
                logger.error(f"Kazanç ayarlanamadı ({gain_db} dB): {e}")
                return False
        elif self.test_mode:
            self.metrics["current_gain_main"] = gain_db
            self.metrics["current_gain_noise"] = gain_db
            logger.info(f"Test modu kazancı ayarlandı: {gain_db} dB")
            return True
        else:
            logger.error("Kazanç ayarlanamadı: SDR başlatılmamış.")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Modülün anlık durumunu ve metriklerini döndürür."""
        # Metrikleri güncelle (örn. CPU kullanımı)
        # self.metrics["cpu_usage"] = psutil.cpu_percent()
        return self.metrics


def _log_metrics(self):
    """Performans metriklerini log dosyasına yazar."""
    logger.info("SDR Alıcı Metrikleri", **self.get_status())

    def is_alive(self) -> bool:
        """Modülün çalışıp çalışmadığını kontrol eder (heartbeat için)."""
        # Basit kontrol: Ana thread çalışıyor mu?
        acq_thread = self.threads.get("acquisition")
        proc_thread = self.threads.get("processing")
        return self.running.is_set() and (acq_thread is not None and acq_thread.is_alive()) and (proc_thread is not None and proc_thread.is_alive())

# --- Komut Satırı Arayüzü (Bağımsız Çalıştırma İçin) --- 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDR Receiver Module")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON configuration file (overrides default)")
    parser.add_argument("--test", action="store_true", help="Run in test mode without SoapySDR")
    parser.add_argument("--freq", type=float, default=DEFAULT_SCAN_FREQ_START_MHZ, help="Initial center frequency in MHz")
    parser.add_argument("--rate", type=float, default=DEFAULT_SDR_SAMPLE_RATE / 1e6, help="Sample rate in MS/s")
    parser.add_argument("--gain", type=float, default=DEFAULT_SDR_GAIN, help="Initial gain in dB")

    args = parser.parse_args()

    # Yapılandırmayı yükle veya varsayılanları kullan
    config = {
        "test_mode": args.test,
        "default_frequency_mhz": args.freq,
        "sample_rate": args.rate * 1e6,
        "default_gain": args.gain,
        # Diğer varsayılanlar config.py\den alınabilir
    }

    if args.config:
        try:
            with open(args.config, "r") as f:  # Ters slash kaldırıldı
                file_config = json.load(f)
                # Dosyadaki sdr_receiver yapılandırmasını al
                sdr_config_from_file = file_config.get("module_configs", {}).get("sdr_receiver", {})
                config.update(sdr_config_from_file)  # Komut satırı argümanlarını override etme
                logger.info(f"Yapılandırma dosyası yüklendi: {args.config}")
        except FileNotFoundError:
            logger.error(
                f"Yapılandırma dosyası bulunamadı: {args.config}. Varsayılanlar ve komut satırı argümanları kullanılıyor.")
        except json.JSONDecodeError as e:
            logger.error(
                f"Yapılandırma dosyası okunamadı (JSON Hatası): {e}. Varsayılanlar ve komut satırı argümanları kullanılıyor.")

    # SDR Alıcıyı başlat
    sdr_receiver = SDRReceiver(config)
    
    # Kapatma sinyali işleyicisi
    def shutdown_handler(signum, frame):
        print("\nKapatma sinyali alındı, SDR Alıcı durduruluyor...")
        sdr_receiver.stop()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    if sdr_receiver.start():
        logger.info("SDR Alıcı çalışıyor. Durdurmak için CTRL+C basın.")
        # Ana thread\in çalışmasını bekle (veya başka bir bekleme mekanizması)
        while sdr_receiver.running.is_set():
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                break # Sinyal işleyici zaten çağrılacak
    else:
        logger.critical("SDR Alıcı başlatılamadı.")

    logger.info("SDR Alıcı programı sonlandı.")

