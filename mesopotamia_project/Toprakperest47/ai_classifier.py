#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ai_classifier.py - Yapay Zeka Sınıflandırma Modülü

Version: 2.2
Author: AI Assistant (Manus)
Last Updated: 2025-04-30

Ön işlenmiş SDR verilerini analiz ederek yeraltı hedeflerini (metal, boşluk,
taş, vb.) tespit eder, sınıflandırır ve özelliklerini tahmin eder.
Yanlış pozitifleri (clutter, yüzey yansımaları) azaltmaya yönelik algoritmalar içerir.
Kural tabanlı veya eğitilmiş bir makine öğrenimi modeli kullanabilir.

Geliştirmeler:
- ZMQ üzerinden Ön İşleme modülünden veri alımı.
- Kural tabanlı sınıflandırma mantığı (genlik, faz değişimi temelli).
- Makine öğrenimi modeli kullanımı için placeholder.
- Yanlış pozitif azaltma (FP Reduction) kuralları.
- Türkçe dokümantasyon ve iyileştirilmiş kod yapısı.
- Optimize edilmiş kural tabanlı eşik değerleri.
"""

import os
import sys
import time
import argparse
import numpy as np
import zmq
import json
import logging
import math
import signal
from threading import Thread, Event, Lock
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field

# Yerel modüller
from config import (
    GroundType, PORT_PREPROCESSING_OUTPUT, PORT_AI_OUTPUT, PORT_AI_CONTROL,
    TARGET_TYPES  # Hedef tipleri için sabitler
)

# Varsayılan sabitler
DEFAULT_PERMITTIVITY = 4.0  # Varsayılan toprak geçirgenlik katsayısı

# Logging yapılandırması
logger = logging.getLogger("AI_CLASSIFIER")
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )


@dataclass
class AIDetectionResult:
    """Yapay zeka tarafından tespit edilen bir hedefin bilgilerini tutar."""
    id: str
    type: str
    position: Tuple[float, float, float]
    size_estimate_m: float
    confidence: float
    depth_m: float
    amplitude: float
    phase_change_rad: float
    shape_estimate: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AIClassifier:
    """Ön işlenmiş SDR verilerini kullanarak hedef tespiti ve sınıflandırması yapar."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = config.get("model_path", None)
        self.use_rule_based = self.model_path is None
        self.confidence_threshold = float(config.get("confidence_threshold", 0.5))

        # Yanlış Pozitif Azaltma Ayarları
        fp_config = config.get("false_positive_reduction", {})
        self.min_depth_fp_m = float(fp_config.get("min_depth_m", 0.15))
        self.max_size_clutter_m = float(fp_config.get("max_size_clutter_m", 0.1))
        self.min_amplitude_target = float(fp_config.get("min_amplitude_target", 0.05))
        self.planar_reflection_phase_threshold_rad = float(fp_config.get("planar_reflection_phase_threshold_rad", 0.2))

        # Kural Tabanlı Sınıflandırma Eşikleri
        rb_config = config.get("rule_based_config", {})
        self.metal_phase_thresh_rad = abs(float(rb_config.get("metal_phase_threshold_rad", math.pi * 0.8)))
        self.void_phase_thresh_rad = -abs(float(rb_config.get("void_phase_threshold_rad", math.pi * 0.8)))
        self.stone_max_abs_phase_rad = abs(float(rb_config.get("stone_max_abs_phase_rad", math.pi * 0.6)))

        self.metal_amp_factor = float(rb_config.get("metal_amplitude_factor", 1.5))
        self.void_amp_factor = float(rb_config.get("void_amplitude_factor", 1.0))
        self.stone_amp_factor = float(rb_config.get("stone_amplitude_factor", 1.0))

        self.metal_conf_amp_norm = self.min_amplitude_target * 3.0
        self.void_conf_amp_norm = self.min_amplitude_target * 2.0
        self.stone_conf_amp_norm = self.min_amplitude_target * 2.0
        self.base_confidence_boost = 0.05
        self.confidence_scale = 1.0 - self.base_confidence_boost

        # ZMQ İletişim
        self.context = zmq.Context()

        # Ön İşleme Veri Girişi (SUB socket)
        self.input_socket = self.context.socket(zmq.SUB)
        self.input_socket.connect(f"tcp://localhost:{PORT_PREPROCESSING_OUTPUT}")
        self.input_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.input_socket.setsockopt(zmq.CONFLATE, 1)

        # AI Sonuç Çıkışı (PUB socket)
        self.output_socket = self.context.socket(zmq.PUB)
        self.output_socket.bind(f"tcp://*:{PORT_AI_OUTPUT}")

        # Kontrol Girişi (REP socket)
        self.control_socket = self.context.socket(zmq.REP)
        self.control_socket.bind(f"tcp://*:{PORT_AI_CONTROL}")
        self.control_socket.setsockopt(zmq.RCVTIMEO, 1000)

        # Dahili Durum
        self.running = Event()
        self.shutdown_event = Event()
        self.lock = Lock()
        self.model = self._load_model()
        self.threads: Dict[str, Optional[Thread]] = {}

        logger.info(
            f"AI Sınıflandırıcı başlatıldı (Mod: {'Kural Tabanlı' if self.use_rule_based else 'Model Tabanlı'}).")

        def _load_model(self):
                """Model yükleme fonksiyonu"""
                # Model yükleme işlemleri burada yapılacak
                pass

            # Diğer metodlar buraya eklenecek...
    def _load_model(self) -> Optional[Any]:
        """Eğitilmiş AI modelini diskten yükler."""
        if self.model_path and os.path.exists(self.model_path):
            logger.info(f"AI modeli yükleniyor: {self.model_path}")
            try:
                # Örnek: TensorFlow/Keras modeli yükleme
                # self.model = tf.keras.models.load_model(self.model_path)
                # Örnek: Scikit-learn modeli yükleme
                # self.model = joblib.load(self.model_path)
                logger.warning("Gerçek model yükleme kodu henüz eklenmedi. Placeholder kullanılıyor.")
                # Placeholder model
                model_placeholder = lambda features: [("UNKNOWN", 0.5)] * len(features)
                self.use_rule_based = False # Model yüklendi (placeholder olsa bile)
                return model_placeholder
            except Exception as e:
                logger.error(f"AI modeli yüklenemedi: {e}. Kural tabanlı moda geçiliyor.", exc_info=True)
                self.use_rule_based = True
                return None
        else:
            if self.model_path:
                logger.warning(f"AI model dosyası bulunamadı: {self.model_path}. Kural tabanlı moda geçiliyor.")
            else:
                logger.info("Model yolu belirtilmedi. Kural tabanlı sınıflandırıcı kullanılacak.")
            self.use_rule_based = True
            return None

    def start(self) -> bool:
        """AI Sınıflandırıcıyı ve ilgili thread\leri başlatır."""
        if self.running.is_set():
            logger.warning("AI Sınıflandırıcı zaten çalışıyor.")
            return True
            
        self.running.set()
        self.shutdown_event.clear()

        self.threads["worker"] = Thread(target=self._processing_worker, name="AIWorker")
        self.threads["control"] = Thread(target=self._control_worker, name="AIControl")
        
        for name, thread in self.threads.items():
            if thread:
                thread.daemon = True
                thread.start()
                logger.info(f"{name} thread\i başlatıldı.")
                
        logger.info("AI Sınıflandırıcı başlatıldı ve ZMQ verisi bekleniyor.")
        return True

    def stop(self):
        """AI Sınıflandırıcıyı ve thread\leri güvenli bir şekilde durdurur."""
        if not self.running.is_set() and self.shutdown_event.is_set():
            return
            
        logger.info("AI Sınıflandırıcı durduruluyor...")
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
            self.input_socket.close(linger=0)
            self.output_socket.close(linger=0)
            self.control_socket.close(linger=0)
            # self.context.term() # Context paylaşımlı olabilir
            logger.info("AI ZMQ soketleri kapatıldı.")
        except Exception as e:
            logger.error(f"AI ZMQ kapatılırken hata: {e}")

        logger.info("AI Sınıflandırıcı durduruldu.")

    def _processing_worker(self):
        """Gelen ön işlenmiş verileri alır, sınıflandırır ve sonuçları yayınlar."""
        logger.info("AI işlem worker başlatıldı.")
        while self.running.is_set():
            try:
                message = self.input_socket.recv_json()
                # Mesaj doğrulaması
                if not isinstance(message, dict) or "targets" not in message or "timestamp" not in message:
                    logger.warning(f"Alınan Ön İşleme mesajı geçersiz veya eksik: {list(message.keys())}")
                    continue

                # Özellikleri çıkar (Preprocessing\den gelen hedefler)
                potential_targets_features = self._extract_features_from_message(message)
                if not potential_targets_features:
                    # logger.debug("Mesajda işlenecek hedef bulunamadı.")
                    continue

                # Sınıflandırma yap
                with self.lock: # Model erişimi için kilit (gerekirse)
                    if self.use_rule_based or self.model is None:
                        classified_targets = self._classify_rule_based(potential_targets_features)
                    else:
                        classified_targets = self._classify_with_model(potential_targets_features)
                
                # Yanlış pozitifleri azalt
                final_detections = self._apply_false_positive_reduction(classified_targets)

                # Sonuçları yayınla
                if final_detections:
                    result_message = {
                        "timestamp": time.time(), # Yayınlama zamanı
                        "source_timestamp": message.get("timestamp"), # Orijinal veri zamanı
                        "detections": [d.__dict__ for d in final_detections] # Dataclass\ı dict\e çevir
                    }
                    self.output_socket.send_json(result_message)
                    logger.debug(f"{len(final_detections)} nihai tespit yayınlandı.")

            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM: break # Context kapatıldı
                logger.error(f"ZMQ AI veri alma hatası: {e}")
                time.sleep(1)
            except json.JSONDecodeError:
                logger.warning("Geçersiz JSON formatında Ön İşleme mesajı alındı.")
            except Exception as e:
                logger.error(f"AI işleme hatası: {e}", exc_info=True)
                time.sleep(1) # Hata durumunda kısa süre bekle
        logger.info("AI işlem worker durdu.")

    def _extract_features_from_message(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Ön işleme mesajından gelen hedefleri AI için özellik formatına dönüştürür."""
        targets_in = message.get("targets", {})
        timestamp = message.get("timestamp", time.time())
        center_freq_hz = message.get("center_freq_hz", 0)
        sample_rate_hz = message.get("sample_rate_hz", 0)
        permittivity = message.get("permittivity", DEFAULT_PERMITTIVITY)
        features_list = []
        
        # Gelen hedefleri işle (targets_in bir dict {index: props})
        for index_str, props in targets_in.items():
            # Gerekli alanları kontrol et
            required_keys = ["depth_m", "amplitude", "phase_rad", "time_delay_s"]
            if not all(k in props for k in required_keys):
                logger.warning(f"Ön işleme hedefinde eksik alanlar: index={index_str}, keys={list(props.keys())}")
                continue
            try:
                feature_dict = {
                    "id": f"target_{timestamp:.6f}_{index_str}", # Daha benzersiz ID
                    "depth_m": float(props["depth_m"]),
                    "amplitude": float(props["amplitude"]),
                    "phase_rad": float(props["phase_rad"]), # Preprocessing\den gelen pik fazı
                    "phase_change_rad": float(props.get("phase_change_rad", 0.0)), # Preprocessing hesapladıysa
                    "time_delay_s": float(props["time_delay_s"]),
                    "peak_width_s": float(props.get("peak_width_s", 0.0)),
                    "timestamp": timestamp,
                    "center_freq_hz": center_freq_hz,
                    "sample_rate_hz": sample_rate_hz,
                    "permittivity": permittivity
                    # Model için ek özellikler buraya eklenebilir
                    # Örn: FFT özellikleri, dalgacık katsayıları vb.
                }
                features_list.append(feature_dict)
            except (ValueError, TypeError) as e:
                 logger.warning(f"Özellik çıkarılırken tip hatası: {e}, index={index_str}, props={props}")
                 
        return features_list

    def _classify_rule_based(self, features_list: List[Dict[str, Any]]) -> List[AIDetectionResult]:
        """Özellik listesini kural tabanlı olarak sınıflandırır."""
        detections = []
        for feat in features_list:
            target_type = TARGET_TYPES["UNKNOWN"] # Varsayılan
            confidence = 0.0
            shape = "point" # Varsayılan şekil
            size_m = 0.1 # Varsayılan boyut

            # Özellikleri al
            amplitude = feat["amplitude"]
            # Fazı -pi ile +pi aralığına sar (wrap)
            phase_rad = (feat["phase_rad"] + math.pi) % (2 * math.pi) - math.pi
            depth_m = feat["depth_m"]
            peak_width_s = feat["peak_width_s"]

            # --- Sınıflandırma Kuralları --- 
            
            # Kural 1: Metal (Yüksek genlik, ~ +/- pi faz kayması)
            is_potential_metal = (amplitude > self.min_amplitude_target * self.metal_amp_factor and 
                                  abs(phase_rad) > self.metal_phase_thresh_rad)
            if is_potential_metal:
                target_type = TARGET_TYPES["METAL"]
                # Güven skoru: Genlik ve faz ne kadar ideale yakınsa o kadar yüksek
                conf_amp_part = min(1.0, amplitude / self.metal_conf_amp_norm)
                conf_phase_part = min(1.0, abs(phase_rad) / math.pi)
                confidence = conf_amp_part * conf_phase_part
                shape = "volumetric" if amplitude > self.metal_conf_amp_norm * 0.8 else "point"
                size_m = amplitude * 0.5 # Boyut genlikle orantılı (kaba tahmin)

            # Kural 2: Boşluk (Orta/Yüksek genlik, ~ -pi faz kayması - metalin tersi?)
            # Not: Boşluk fazı modele/deneye göre değişir, bu varsayımsal.
            is_potential_void = (amplitude > self.min_amplitude_target * self.void_amp_factor and 
                                 phase_rad < self.void_phase_thresh_rad) # Negatif büyük faz
            if is_potential_void and target_type == TARGET_TYPES["UNKNOWN"]: # Metalle çakışmadıysa
                target_type = TARGET_TYPES["VOID"]
                conf_amp_part = min(1.0, amplitude / self.void_conf_amp_norm)
                conf_phase_part = min(1.0, abs(phase_rad) / math.pi)
                confidence = conf_amp_part * conf_phase_part
                shape = "volumetric"
                size_m = amplitude * 0.8 # Boşluklar daha büyük olabilir

            # Kural 3: Taş/Mineral (Orta genlik, küçük faz kayması)
            is_potential_stone = (amplitude > self.min_amplitude_target * self.stone_amp_factor and 
                                  abs(phase_rad) < self.stone_max_abs_phase_rad)
            if is_potential_stone and target_type == TARGET_TYPES["UNKNOWN"]: # Diğerleriyle çakışmadıysa
                target_type = TARGET_TYPES["STONE"]
                conf_amp_part = min(1.0, amplitude / self.stone_conf_amp_norm)
                # Faz sıfıra ne kadar yakınsa o kadar güvenli
                conf_phase_part = max(0.0, 1.0 - abs(phase_rad) / self.stone_max_abs_phase_rad)
                confidence = conf_amp_part * conf_phase_part
                shape = "point"
                size_m = amplitude * 0.3

            # Kural 4: Düzlemsel Yansıma / Clutter (Küçük faz kayması, düşük güven)
            if abs(phase_rad) < self.planar_reflection_phase_threshold_rad and confidence < 0.5:
                 target_type = TARGET_TYPES["CLUTTER"]
                 confidence = max(0.1, confidence * 0.5) # Güveni düşür ama sıfır yapma
                 shape = "planar"

            # Güven skorunu son aralığa (0-1) ölçekle ve taban ekle
            final_confidence = max(0.0, min(1.0, confidence * self.confidence_scale + self.base_confidence_boost))

            # Tespit sonucunu oluştur
            detection = AIDetectionResult(
                id=feat["id"],
                type=target_type,
                position=(0, 0, depth_m), # x, y şimdilik 0
                size_estimate_m=max(0.05, size_m), # Minimum boyut 5cm
                confidence=round(final_confidence, 3),
                depth_m=depth_m,
                amplitude=amplitude,
                phase_change_rad=phase_rad, # Kullanılan faz değeri
                shape_estimate=shape,
                timestamp=feat["timestamp"],
                metadata={"center_freq_hz": feat["center_freq_hz"], "peak_width_s": peak_width_s}
            )
            detections.append(detection)
            
        # logger.debug(f"Kural tabanlı sınıflandırma: {len(detections)} potansiyel tespit.")
        return detections

    def _classify_with_model(self, features_list: List[Dict[str, Any]]) -> List[AIDetectionResult]:
        """Özellik listesini eğitilmiş AI modelini kullanarak sınıflandırır."""
        if self.model is None:
            logger.error("Model yüklenemediği için model tabanlı sınıflandırma yapılamıyor.")
            return self._classify_rule_based(features_list) # Fallback
            
        logger.debug(f"{len(features_list)} hedef için model tabanlı sınıflandırma yapılıyor (Placeholder)...")
        detections = []
        
        # --- Placeholder Logic --- 
        # Modelin girdi olarak ne beklediğini varsayalım (örn. numpy array)
        # input_data = np.array([[f["depth_m"], f["amplitude"], f["phase_rad"], f["peak_width_s"]] for f in features_list])
        # predictions = self.model.predict(input_data) # Modelin çıktısı [(type_str, confidence), ...]
        predictions = self.model(features_list) # Placeholder model çağrısı

        for i, feat in enumerate(features_list):
            pred_type_str, pred_confidence = predictions[i]
            
            # Model çıktısını doğrula
            target_type = TARGET_TYPES.get(pred_type_str, TARGET_TYPES["UNKNOWN"])
            confidence = max(0.0, min(1.0, float(pred_confidence)))
            
            # Şekil ve boyut tahmini (model yapmıyorsa kural tabanlıdan alınabilir)
            shape = "volumetric" if confidence > 0.7 else "point"
            size_m = feat["amplitude"] * 0.4 # Kaba tahmin
            
            detection = AIDetectionResult(
                id=feat["id"],
                type=target_type,
                position=(0, 0, feat["depth_m"]),
                size_estimate_m=max(0.05, size_m),
                confidence=round(confidence, 3),
                depth_m=feat["depth_m"],
                amplitude=feat["amplitude"],
                phase_change_rad=feat["phase_rad"],
                shape_estimate=shape,
                timestamp=feat["timestamp"],
                metadata={"center_freq_hz": feat["center_freq_hz"], "classifier": "model"}
            )
            detections.append(detection)
        # --- Placeholder Logic Sonu ---
        
        return detections

    def _apply_false_positive_reduction(self, detections: List[AIDetectionResult]) -> List[AIDetectionResult]:
        """Sınıflandırılmış tespitler üzerinde yanlış pozitif azaltma kurallarını uygular."""
        filtered_detections = []
        num_original = len(detections)
        
        for det in detections:
            is_fp = False
            reason = ""

            # Kural 1: Güven eşiği
            if det.confidence < self.confidence_threshold:
                is_fp = True; reason = f"Düşük güven ({det.confidence:.2f} < {self.confidence_threshold:.2f})"
            # Kural 2: Minimum derinlik
            elif det.depth_m < self.min_depth_fp_m:
                 is_fp = True; reason = f"Çok yüzeyde ({det.depth_m:.2f}m < {self.min_depth_fp_m:.2f}m)"
            # Kural 3: Minimum genlik
            elif det.amplitude < self.min_amplitude_target:
                 is_fp = True; reason = f"Zayıf genlik ({det.amplitude:.3f} < {self.min_amplitude_target:.3f})"
            # Kural 4: Clutter olarak sınıflandırıldıysa
            elif det.type == TARGET_TYPES["CLUTTER"]:
                 is_fp = True; reason = "Clutter olarak sınıflandırıldı"
            # Kural 5: Çok küçük metal tespiti (çöp olabilir)
            elif det.type == TARGET_TYPES["METAL"] and det.size_estimate_m < self.max_size_clutter_m:
                 is_fp = True; reason = f"Küçük metal çöp ({det.size_estimate_m:.2f}m < {self.max_size_clutter_m:.2f}m)"
            # Kural 6: Düzlemsel yansıma şüphesi (düşük faz değişimi)
            # elif abs(det.phase_change_rad) < self.planar_reflection_phase_threshold_rad and det.confidence < 0.6:
            #      is_fp = True; reason = "Düzlemsel yansıma şüphesi"
                 
            # TODO: Gelişmiş FP Azaltma Kuralları:
            # - Ağaç kökü deseni? (Genellikle daha karmaşık, dallanan yapılar)
            # - Tekrarlayan yansımalar (multipaht, anten vb.)
            # - Zemin katmanı tespiti ve filtrelemesi

            if not is_fp:
                filtered_detections.append(det)
            else:
                logger.debug(f"FP elendi: {det.id} ({det.type} @ {det.depth_m:.2f}m) - Sebep: {reason}")

        num_filtered = len(filtered_detections)
        if num_original > num_filtered:
             logger.debug(f"{num_original - num_filtered} tespit FP olarak elendi ({num_filtered} kaldı).")
             
        return filtered_detections

    def _control_worker(self):
        """ZMQ üzerinden gelen kontrol komutlarını işler."""
        logger.info("AI kontrol thread\i başladı.")
        while not self.shutdown_event.is_set():
            try:
                message = self.control_socket.recv_json()
                response = self._handle_control_command(message)
                self.control_socket.send_json(response)
            except zmq.Again:
                continue # Timeout
            except zmq.ZMQError as e:
                 if e.errno == zmq.ETERM: break
                 logger.error(f"ZMQ AI kontrol hatası: {e}")
                 time.sleep(1)
            except json.JSONDecodeError:
                logger.error("Geçersiz JSON formatında kontrol mesajı alındı.")
                try: self.control_socket.send_json({"status": "error", "message": "Invalid JSON"})
                except: pass
            except Exception as e:
                logger.error(f"AI kontrol hatası: {e}", exc_info=True)
                try: self.control_socket.send_json({"status": "error", "message": str(e)})
                except: pass
                time.sleep(0.1)
        logger.info("AI kontrol thread\i durdu.")

    def _handle_control_command(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Gelen kontrol komutunu işler ve yanıt döndürür."""
        command = message.get("command")
        params = message.get("params", {})
        logger.info(f"Kontrol komutu alındı: {command}", params=params)
        
        try:
            if command == "get_status":
                with self.lock:
                    status_data = {
                        "running": self.running.is_set(),
                        "mode": "rule-based" if self.use_rule_based else "model-based",
                        "model_loaded": self.model is not None,
                        "confidence_threshold": self.confidence_threshold
                    }
                return {"status": "ok", "data": status_data}
            
            elif command == "set_parameter":
                param_name = params.get("name")
                param_value = params.get("value")
                if param_name == "confidence_threshold":
                    try:
                        threshold = float(param_value)
                        if 0.0 <= threshold <= 1.0:
                            self.confidence_threshold = threshold
                            logger.info(f"Güven eşiği ayarlandı: {self.confidence_threshold:.2f}")
                            return {"status": "ok"}
                        else:
                            return {"status": "error", "message": "Güven eşiği 0-1 arasında olmalı."}
                    except (ValueError, TypeError):
                         return {"status": "error", "message": "Geçersiz güven eşik değeri."}
                # Diğer ayarlanabilir parametreler (örn. FP kuralları) eklenebilir
                else:
                    return {"status": "error", "message": f"Bilinmeyen veya ayarlanamayan parametre: {param_name}"}

            elif command == "reload_model":
                with self.lock:
                    self.model = self._load_model()
                status = "ok" if not self.model_path or self.model else "error"
                msg = "Model yeniden yüklendi." if status == "ok" else "Model yüklenemedi."
                mode_msg = "Kural tabanlı moda geçildi." if self.use_rule_based else "Model tabanlı moda geçildi."
                return {"status": status, "message": f"{msg} {mode_msg}"}
                
            else:
                return {"status": "error", "message": f"Bilinmeyen komut: {command}"}
                
        except Exception as e:
             logger.error(f"Kontrol komutu işlenirken hata: {e}", command=command, exc_info=True)
             return {"status": "error", "message": f"Komut işlenirken hata: {e}"}

    def is_alive(self) -> bool:
        """Modülün çalışıp çalışmadığını kontrol eder."""
        worker = self.threads.get("worker")
        return self.running.is_set() and (worker is not None and worker.is_alive())

# --- Komut Satırı Arayüzü (Bağımsız Çalıştırma İçin) --- 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Classification Module")
    parser.add_argument("--config", type=str, default=None, help="Path to system JSON configuration file")
    parser.add_argument("--model", type=str, default=None, help="Path to the trained AI model file (overrides config)")

    args = parser.parse_args()

    # Yapılandırmayı yükle
    config = {}
    if args.config:
        try:
            with open(args.config, "r") as f:
                file_config = json.load(f)
                config = file_config.get("module_configs", {}).get("ai_classifier", {})
                logger.info(f"Yapılandırma dosyası yüklendi: {args.config}")
        except FileNotFoundError:
            logger.error(f"Yapılandırma dosyası bulunamadı: {args.config}. Varsayılanlar kullanılıyor.")
        except json.JSONDecodeError as e:
            logger.error(f"Yapılandırma dosyası okunamadı (JSON Hatası): {e}. Varsayılanlar kullanılıyor.")
    else:
        logger.warning("Yapılandırma dosyası belirtilmedi, varsayılan ayarlar kullanılıyor.")

    # Komut satırı model yolunu override et
    if args.model:
        config["model_path"] = args.model
        logger.info(f"Komut satırından model yolu ayarlandı: {args.model}")

    # Modülü başlat
    classifier = AIClassifier(config)

    def shutdown_handler(signum, frame):
        print("\nKapatma sinyali alındı, AI Sınıflandırıcı durduruluyor...")
        classifier.stop()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    if classifier.start():
        logger.info("AI Sınıflandırıcı çalışıyor. Durdurmak için CTRL+C basın.")
        # Ana thread\in çalışmasını bekle
        while classifier.running.is_set():
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                break # Sinyal işleyici zaten çağrılacak
    else:
        logger.critical("AI Sınıflandırıcı başlatılamadı.")

    logger.info("AI Sınıflandırıcı programı sonlandı.")

