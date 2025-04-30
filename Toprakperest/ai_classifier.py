#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibration.py - Geliştirilmiş Versiyon

SDR tabanlı yeraltı tespit sistemi için gelişmiş kalibrasyon modülü.
Zemin özelliklerini tespit ederek sistem performansını optimize eder.

Geliştirmeler:
- Eksik işlevler tamamlandı
- Hata yönetimi iyileştirildi
- Test modu geliştirildi
- Veri işleme algoritmaları eklendi
- Daha detaylı zemin analizi
- Gerçek zamanlı veri görselleştirme desteği
"""

import os
import sys
import time
import argparse
import numpy as np
import zmq
import json
import logging
import pickle
from datetime import datetime
from threading import Thread, Event
from queue import Queue
import scipy.signal as signal
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# SDR kütüphaneleri için koşullu import
try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32, SOAPY_SDR_OVERFLOW, SOAPY_SDR_TIMEOUT

    SOAPY_AVAILABLE = True
except ImportError:
    SOAPY_AVAILABLE = False
    print("SoapySDR kütüphanesi bulunamadı. Test modunda çalışılıyor.")

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("calibration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("calibration")


class GroundCalibration:
    """
    Gelişmiş zemin kalibrasyonu ve çevresel veri toplama sınıfı.
    """

    # Geliştirilmiş zemin türleri ve özellikleri
    GROUND_TYPES = {
        "dry_soil": {
            "description": "Kuru Toprak",
            "permittivity": (2.5, 3.5),
            "conductivity": (0.001, 0.01),
            "moisture": (0, 15),
            "mineral_noise": (0, 20),
            "optimal_freq": (1.2e9, 2.0e9),
            "filter_params": {
                "bandpass_low": 800e6,
                "bandpass_high": 2.2e9,
                "noise_threshold": 0.15
            },
            "color": "tan"
        },
        "wet_soil": {
            "description": "Nemli Toprak",
            "permittivity": (15, 30),
            "conductivity": (0.01, 0.1),
            "moisture": (15, 40),
            "mineral_noise": (0, 30),
            "optimal_freq": (800e6, 1.5e9),
            "filter_params": {
                "bandpass_low": 600e6,
                "bandpass_high": 1.8e9,
                "noise_threshold": 0.25
            },
            "color": "darkgreen"
        },
        "limestone": {
            "description": "Kireçli",
            "permittivity": (4, 8),
            "conductivity": (0.001, 0.005),
            "moisture": (5, 15),
            "mineral_noise": (10, 40),
            "optimal_freq": (1.0e9, 2.5e9),
            "filter_params": {
                "bandpass_low": 700e6,
                "bandpass_high": 2.8e9,
                "noise_threshold": 0.2
            },
            "color": "lightgray"
        },
        "iron_oxide": {
            "description": "Demir Oksitli",
            "permittivity": (5, 15),
            "conductivity": (0.05, 0.5),
            "moisture": (5, 25),
            "mineral_noise": (30, 70),
            "optimal_freq": (500e6, 1.2e9),
            "filter_params": {
                "bandpass_low": 400e6,
                "bandpass_high": 1.5e9,
                "noise_threshold": 0.3
            },
            "color": "red"
        },
        "mineral_rich": {
            "description": "Mineral Gürültülü",
            "permittivity": (6, 20),
            "conductivity": (0.01, 0.2),
            "moisture": (10, 30),
            "mineral_noise": (50, 90),
            "optimal_freq": (400e6, 1.0e9),
            "filter_params": {
                "bandpass_low": 350e6,
                "bandpass_high": 1.2e9,
                "noise_threshold": 0.35
            },
            "color": "gold"
        }
    }

    def __init__(self, args):
        self.args = args
        self.device_args = args.device_args
        self.freq = args.freq
        self.sample_rate = args.rate
        self.bandwidth = args.bandwidth
        self.gain = args.gain
        self.antenna = args.antenna
        self.channel = args.channel
        self.test_mode = args.test_mode
        self.calibration_time = args.calibration_time
        self.calibration_sweeps = args.calibration_sweeps
        self.model_path = args.model_path
        self.visualize = args.visualize

        # Kalibrasyon durumu
        self.calibration_complete = False
        self.ground_type = None
        self.ground_params = {}
        self.calibration_progress = 0.0

        # Veri yapıları
        self.calibration_data = []
        self.feature_vectors = []
        self.spectral_data = []
        self.reference_data = self._load_reference_data()

        # İletişim
        self.context = zmq.Context()
        self.result_socket = self.context.socket(zmq.PUB)
        self.result_socket.bind(f"tcp://*:{args.result_port}")
        self.control_socket = self.context.socket(zmq.REP)
        self.control_socket.bind(f"tcp://*:{args.control_port}")

        # SDR cihazı
        self.sdr = None
        self.rx_stream = None

        # İş parçacıkları
        self.running = Event()
        self.data_queue = Queue(maxsize=1000)
        self.acquisition_thread = None
        self.processing_thread = None
        self.control_thread = None

        # Görselleştirme
        self.fig = None
        self.ax = None
        self.ani = None
        self.spectrum_line = None
        self.ground_type_text = None

        logger.info(f"Kalibrasyon modülü başlatıldı: test_mode={self.test_mode}")

        if not self.test_mode and SOAPY_AVAILABLE:
            self._setup_sdr()

    def _setup_sdr(self):
        """SDR cihazını yapılandırır."""
        try:
            self.sdr = SoapySDR.Device(self.device_args)
            self.sdr.setSampleRate(SOAPY_SDR_RX, self.channel, self.sample_rate)
            self.sdr.setFrequency(SOAPY_SDR_RX, self.channel, self.freq)

            if self.bandwidth:
                self.sdr.setBandwidth(SOAPY_SDR_RX, self.channel, self.bandwidth)
            if self.antenna:
                self.sdr.setAntenna(SOAPY_SDR_RX, self.channel, self.antenna)

            self.sdr.setGain(SOAPY_SDR_RX, self.channel, self.gain)
            self.rx_stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [self.channel])

            logger.info(f"SDR başarıyla yapılandırıldı: {self.sdr.getHardwareInfo()}")
            return True
        except Exception as e:
            logger.error(f"SDR kurulum hatası: {str(e)}")
            return False

    def _load_reference_data(self):
        """Referans kalibrasyon verilerini yükler."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Referans veriler yüklendi: {self.model_path}")
                return data
            except Exception as e:
                logger.error(f"Referans veri yükleme hatası: {str(e)}")
        return None

    def _save_reference_data(self):
        """Mevcut kalibrasyonu referans olarak kaydeder."""
        if not self.ground_type:
            return False

        data = {
            "ground_type": self.ground_type,
            "ground_params": self.ground_params,
            "features": self._extract_features(self.calibration_data),
            "timestamp": datetime.now().isoformat()
        }

        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Referans veriler kaydedildi: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Referans veri kaydetme hatası: {str(e)}")
            return False

    def start(self):
        """Kalibrasyon işlemini başlatır."""
        if not self.test_mode and not self._setup_sdr():
            return False

        self.running.set()

        # İş parçacıklarını başlat
        self.acquisition_thread = Thread(target=self._acquisition_worker)
        self.processing_thread = Thread(target=self._processing_worker)
        self.control_thread = Thread(target=self._control_worker)

        for thread in [self.acquisition_thread, self.processing_thread, self.control_thread]:
            thread.daemon = True
            thread.start()

        # Görselleştirme
        if self.visualize:
            self._setup_visualization()

        logger.info("Kalibrasyon başlatıldı")
        return True

    def stop(self):
        """Kalibrasyon işlemini durdurur."""
        self.running.clear()

        # İş parçacıklarını durdur
        for thread in [self.acquisition_thread, self.processing_thread, self.control_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=2.0)

        # SDR'yi kapat
        if not self.test_mode and self.sdr and self.rx_stream:
            self.sdr.deactivateStream(self.rx_stream)
            self.sdr.closeStream(self.rx_stream)

        # Görselleştirmeyi kapat
        if self.visualize and self.ani:
            self.ani.event_source.stop()
            plt.close(self.fig)

        logger.info("Kalibrasyon durduruldu")

    def _acquisition_worker(self):
        """SDR'den veri toplar."""
        if self.test_mode:
            self._test_data_generator()
            return

        try:
            self.sdr.activateStream(self.rx_stream)
            buffer_size = 1024  # Örnek sayısı
            buffer = np.zeros(buffer_size, np.complex64)
            timeout_us = int(1e6)  # 1 saniye timeout

            start_freq = 300e6
            end_freq = 3e9
            freq_step = (end_freq - start_freq) / self.calibration_sweeps

            logger.info(f"Frekans taraması başlatılıyor: {start_freq / 1e6}MHz - {end_freq / 1e6}MHz")

            for sweep in range(self.calibration_sweeps):
                if not self.running.is_set():
                    break

                current_freq = start_freq + sweep * freq_step
                self.sdr.setFrequency(SOAPY_SDR_RX, self.channel, current_freq)
                self.calibration_progress = (sweep + 1) / self.calibration_sweeps

                logger.debug(f"Tarama {sweep + 1}/{self.calibration_sweeps} - Frekans: {current_freq / 1e6:.2f}MHz")

                sweep_time = self.calibration_time / self.calibration_sweeps
                start_time = time.time()

                while time.time() - start_time < sweep_time and self.running.is_set():
                    try:
                        status = self.sdr.readStream(self.rx_stream, [buffer], buffer_size, timeoutUs=timeout_us)

                        if status.ret > 0:
                            data = {
                                "samples": buffer.copy(),
                                "freq": current_freq,
                                "timestamp": time.time()
                            }
                            if not self.data_queue.full():
                                self.data_queue.put(data)
                        elif status.ret == SOAPY_SDR_OVERFLOW:
                            logger.warning("Tampon taşması - veri kaybı")
                        elif status.ret == SOAPY_SDR_TIMEOUT:
                            logger.warning("Zaman aşımı - veri alınamadı")
                        else:
                            logger.error(f"SDR okuma hatası: {status.ret}")

                    except Exception as e:
                        logger.error(f"Veri alma hatası: {str(e)}")
                        time.sleep(0.1)

            logger.info("Frekans taraması tamamlandı")

        except Exception as e:
            logger.error(f"Veri toplama hatası: {str(e)}")
            self.running.clear()

    def _test_data_generator(self):
        """Test modu için sentetik veri üretir."""
        sample_rate = self.sample_rate
        buffer_size = 1024
        t = np.arange(buffer_size) / sample_rate

        start_freq = 300e6
        end_freq = 3e9
        freq_step = (end_freq - start_freq) / self.calibration_sweeps

        # Rastgele bir zemin türü seç
        ground_type = np.random.choice(list(self.GROUND_TYPES.keys()))
        ground_info = self.GROUND_TYPES[ground_type]
        logger.info(f"Test modu: Simüle edilen zemin türü: {ground_info['description']}")

        for sweep in range(self.calibration_sweeps):
            if not self.running.is_set():
                break

            current_freq = start_freq + sweep * freq_step
            self.calibration_progress = (sweep + 1) / self.calibration_sweeps

            sweep_time = self.calibration_time / self.calibration_sweeps
            start_time = time.time()

            while time.time() - start_time < sweep_time and self.running.is_set():
                try:
                    # Zemin özelliklerine göre sinyal üret
                    permittivity = np.random.uniform(*ground_info["permittivity"])
                    conductivity = np.random.uniform(*ground_info["conductivity"])

                    # Temel sinyal
                    signal_freq = current_freq * (1 + 0.01 * (permittivity - 1))
                    main_signal = np.exp(2j * np.pi * signal_freq * t)

                    # Gürültü ekle
                    noise_level = conductivity * 0.1
                    noise = noise_level * (np.random.randn(buffer_size) + 1j * np.random.randn(buffer_size))

                    # Mineral gürültüsü
                    mineral_noise = ground_info["mineral_noise"][0] / 100 * np.random.randn(buffer_size)

                    # Birleştir
                    samples = main_signal * (1 + mineral_noise) + noise

                    data = {
                        "samples": samples,
                        "freq": current_freq,
                        "timestamp": time.time(),
                        "test_params": {
                            "ground_type": ground_type,
                            "permittivity": permittivity,
                            "conductivity": conductivity
                        }
                    }

                    if not self.data_queue.full():
                        self.data_queue.put(data)

                    time.sleep(0.01)

                except Exception as e:
                    logger.error(f"Test verisi üretme hatası: {str(e)}")
                    time.sleep(0.1)

    def _processing_worker(self):
        """Toplanan verileri işler ve zemin türünü belirler."""
        while self.running.is_set() or not self.data_queue.empty():
            try:
                if self.data_queue.empty():
                    time.sleep(0.1)
                    continue

                data = self.data_queue.get()
                self.calibration_data.append(data)

                # Özellik çıkarımı
                features = self._extract_features([data])
                self.feature_vectors.extend(features)

                # Spektral analiz
                spectrum = self._compute_spectrum(data['samples'])
                self.spectral_data.append({
                    'freq': data['freq'],
                    'spectrum': spectrum
                })

                # Zemin türü tahmini
                if len(self.feature_vectors) > 100:  # Yeterli veri toplandığında
                    self._determine_ground_type()

                # Görselleştirme güncelleme
                if self.visualize and len(self.spectral_data) % 10 == 0:
                    self._update_visualization()

                self.data_queue.task_done()

            except Exception as e:
                logger.error(f"Veri işleme hatası: {str(e)}")
                time.sleep(0.1)

    def _extract_features(self, data_chunks):
        """Ham veriden özellikler çıkarır."""
        features = []
        for chunk in data_chunks:
            samples = chunk['samples']

            # Temel istatistikler
            power = np.mean(np.abs(samples) ** 2)
            phase = np.mean(np.angle(samples))
            iq_ratio = np.mean(np.abs(samples.real)) / np.mean(np.abs(samples.imag))

            # Spektral özellikler
            spectrum = self._compute_spectrum(samples)
            spectral_centroid = np.sum(spectrum * np.arange(len(spectrum))) / np.sum(spectrum)
            spectral_spread = np.sqrt(
                np.sum((np.arange(len(spectrum)) - spectral_centroid) ** 2 * spectrum) / np.sum(spectrum))

            # Dalga biçimi özellikleri
            waveform_skew = skew(np.abs(samples))
            waveform_kurtosis = kurtosis(np.abs(samples))

            features.append([
                power, phase, iq_ratio,
                spectral_centroid, spectral_spread,
                waveform_skew, waveform_kurtosis,
                chunk['freq']
            ])

        return features

    def _compute_spectrum(self, samples):
        """Sinyalin güç spektrumunu hesaplar."""
        window = signal.windows.hann(len(samples))
        spectrum = np.abs(np.fft.fft(samples * window)) ** 2
        spectrum = spectrum[:len(spectrum) // 2]  # Tek taraflı spektrum
        spectrum = 10 * np.log10(spectrum + 1e-12)  # dB cinsinden
        return spectrum

    def _determine_ground_type(self):
        """Toplanan verilere göre zemin türünü belirler."""
        try:
            # Özellikleri ölçeklendir
            scaler = StandardScaler()
            X = scaler.fit_transform(self.feature_vectors)

            # Boyut indirgeme
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X)

            # Kümeleme
            kmeans = KMeans(n_clusters=5, random_state=42)
            clusters = kmeans.fit_predict(X_pca)

            # Anomali tespiti
            iso_forest = IsolationForest(contamination=0.05)
            anomalies = iso_forest.fit_predict(X_pca)

            # Temiz veri
            clean_mask = anomalies != -1
            X_clean = X_pca[clean_mask]
            clusters_clean = clusters[clean_mask]

            # Her küme için ortalama özellikler
            cluster_features = []
            for i in range(5):
                cluster_data = X_clean[clusters_clean == i]
                if len(cluster_data) > 0:
                    cluster_features.append(np.mean(cluster_data, axis=0))

            # Referans veri varsa karşılaştır
            if self.reference_data:
                ref_features = self.reference_data['features']
                # Benzerlik hesapla (basit öklid mesafesi)
                distances = []
                for cf in cluster_features:
                    dist = np.mean([np.linalg.norm(cf - rf[:3]) for rf in ref_features])
                    distances.append(dist)

                best_match = np.argmin(distances)
                ground_type = self.reference_data['ground_type']
            else:
                # En büyük kümeyi seç
                cluster_sizes = [np.sum(clusters_clean == i) for i in range(5)]
                best_match = np.argmax(cluster_sizes)

                # Kümeyi zemin türüne eşle
                ground_types = list(self.GROUND_TYPES.keys())
                ground_type = ground_types[best_match % len(ground_types)]

            # Zemin parametrelerini hesapla
            self.ground_type = ground_type
            self.ground_params = self._estimate_ground_parameters(X_clean[clusters_clean == best_match])

            # Sonuçları yayınla
            self._publish_results()
            self.calibration_complete = True

            logger.info(f"Zemin türü belirlendi: {self.GROUND_TYPES[ground_type]['description']}")

            # Referans olarak kaydet
            self._save_reference_data()

            return True

        except Exception as e:
            logger.error(f"Zemin türü belirleme hatası: {str(e)}")
            return False

    def _estimate_ground_parameters(self, cluster_data):
        """Zemin parametrelerini tahmin eder."""
        params = {
            "permittivity": np.mean(cluster_data[:, 0]) * 5 + 5,  # Ölçeklendirilmiş
            "conductivity": np.mean(cluster_data[:, 1]) * 0.1 + 0.05,
            "moisture": np.mean(cluster_data[:, 2]) * 10 + 10,
            "mineral_noise": np.var(cluster_data) * 50,
            "stability": 1 - np.var(cluster_data) / 10
        }
        return params

    def _publish_results(self):
        """Kalibrasyon sonuçlarını yayınlar."""
        if not self.ground_type:
            return

        result = {
            "ground_type": self.ground_type,
            "description": self.GROUND_TYPES[self.ground_type]["description"],
            "params": self.ground_params,
            "filter_params": self.GROUND_TYPES[self.ground_type]["filter_params"],
            "optimal_freq": self.GROUND_TYPES[self.ground_type]["optimal_freq"],
            "timestamp": datetime.now().isoformat(),
            "calibration_data": {
                "samples_processed": len(self.calibration_data),
                "features_extracted": len(self.feature_vectors)
            }
        }

        try:
            self.result_socket.send_json(result)
            logger.debug("Kalibrasyon sonuçları yayınlandı")
        except Exception as e:
            logger.error(f"Sonuç yayınlama hatası: {str(e)}")

    def _control_worker(self):
        """Kontrol komutlarını işler."""
        while self.running.is_set():
            try:
                # Komut bekleyin (bloke olmayan şekilde)
                try:
                    message = self.control_socket.recv_json(flags=zmq.NOBLOCK)
                except zmq.Again:
                    time.sleep(0.1)
                    continue

                # Komut işleme
                response = {"status": "error", "message": "Geçersiz komut"}

                if message.get("command") == "status":
                    response = {
                        "status": "ok",
                        "calibration_complete": self.calibration_complete,
                        "ground_type": self.ground_type,
                        "progress": self.calibration_progress,
                        "samples_processed": len(self.calibration_data)
                    }
                elif message.get("command") == "start":
                    if not self.running.is_set():
                        self.start()
                        response = {"status": "ok", "message": "Kalibrasyon başlatıldı"}
                elif message.get("command") == "stop":
                    if self.running.is_set():
                        self.stop()
                        response = {"status": "ok", "message": "Kalibrasyon durduruldu"}
                elif message.get("command") == "save_reference":
                    success = self._save_reference_data()
                    response = {
                        "status": "ok" if success else "error",
                        "message": "Referans kaydedildi" if success else "Referans kaydedilemedi"
                    }

                # Yanıt gönder
                self.control_socket.send_json(response)

            except Exception as e:
                logger.error(f"Kontrol işleme hatası: {str(e)}")
                self.control_socket.send_json({"status": "error", "message": str(e)})
                time.sleep(0.1)

    def _setup_visualization(self):
        """Gerçek zamanlı veri görselleştirmesini kurar."""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title("Kalibrasyon Verisi - Frekans Spektrumu")
        self.ax.set_xlabel("Frekans (Hz)")
        self.ax.set_ylabel("Güç (dB)")
        self.ax.grid(True)

        # Başlangıç çizgisi
        self.spectrum_line, = self.ax.plot([], [], 'b-', linewidth=1)
        self.ground_type_text = self.ax.text(0.02, 0.95, "Zemin Türü: Belirleniyor...",
                                             transform=self.ax.transAxes, fontsize=12,
                                             bbox=dict(facecolor='white', alpha=0.7))

        # Animasyon başlat
        self.ani = FuncAnimation(self.fig, self._update_visualization, interval=500, blit=False)
        plt.show(block=False)

    def _update_visualization(self, frame=None):
        """Görselleştirmeyi günceller."""
        if not self.spectral_data:
            return

        try:
            # En son spektrumu al
            latest = self.spectral_data[-1]
            freq = latest['freq']
            spectrum = latest['spectrum']

            # Frekans vektörü oluştur
            freqs = np.linspace(freq - self.sample_rate / 2, freq + self.sample_rate / 2, len(spectrum))

            # Çizgiyi güncelle
            self.spectrum_line.set_data(freqs, spectrum)
            self.ax.relim()
            self.ax.autoscale_view()

            # Zemin türü bilgisini güncelle
            if self.ground_type:
                ground_info = self.GROUND_TYPES[self.ground_type]
                text = f"Zemin Türü: {ground_info['description']}\n"
                text += f"Dielektrik: {self.ground_params['permittivity']:.2f}\n"
                text += f"İletkenlik: {self.ground_params['conductivity']:.4f} S/m"

                self.ground_type_text.set_text(text)
                self.ground_type_text.set_bbox(dict(facecolor=ground_info['color'], alpha=0.7))

            self.fig.canvas.draw_idle()

        except Exception as e:
            logger.error(f"Görselleştirme güncelleme hatası: {str(e)}")


def parse_arguments():
    """Komut satırı argümanlarını ayrıştırır."""
    parser = argparse.ArgumentParser(description="SDR Tabanlı Yeraltı Tespit Sistemi - Kalibrasyon Modülü")

    # SDR parametreleri
    parser.add_argument("--device-args", type=str, default="", help="SDR cihaz argümanları")
    parser.add_argument("--freq", type=float, default=1.5e9, help="Merkez frekansı (Hz)")
    parser.add_argument("--rate", type=float, default=20e6, help="Örnekleme hızı (Hz)")
    parser.add_argument("--bandwidth", type=float, default=0, help="Sinyal bant genişliği (Hz)")
    parser.add_argument("--gain", type=float, default=30, help="Kazanç değeri (dB)")
    parser.add_argument("--antenna", type=str, default="", help="Anten seçimi")
    parser.add_argument("--channel", type=int, default=0, help="Kanal numarası")

    # Kalibrasyon parametreleri
    parser.add_argument("--calibration-time", type=float, default=30.0,
                        help="Toplam kalibrasyon süresi (saniye)")
    parser.add_argument("--calibration-sweeps", type=int, default=10,
                        help="Frekans tarama sayısı")
    parser.add_argument("--model-path", type=str, default="calibration_model.pkl",
                        help="Kalibrasyon model dosyası yolu")

    # İletişim parametreleri
    parser.add_argument("--result-port", type=int, default=5557,
                        help="Sonuç yayınlama portu")
    parser.add_argument("--control-port", type=int, default=5558,
                        help="Kontrol portu")

    # Diğer parametreler
    parser.add_argument("--test-mode", action="store_true",
                        help="Test modunda çalıştır (gerçek SDR kullanma)")
    parser.add_argument("--visualize", action="store_true",
                        help="Gerçek zamanlı veri görselleştirmeyi etkinleştir")

    return parser.parse_args()


def main():
    """Ana uygulama giriş noktası."""
    args = parse_arguments()

    try:
        calibrator = GroundCalibration(args)
        calibrator.start()

        # Ana döngü
        while calibrator.running.is_set():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nKullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"Kritik hata: {str(e)}")
    finally:
        calibrator.stop()


if __name__ == "__main__":
    main()