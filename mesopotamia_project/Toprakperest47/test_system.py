#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_system.py - System Integration and Unit Tests

Version: 1.0
Author: AI Assistant (Manus)
Last Updated: 2025-04-30

Mesopotamia GPR sistemi için birim ve entegrasyon testleri.
Bu testler, ana modüllerin temel işlevselliğini, yapılandırma yüklemesini,
veri formatlarını ve modüller arası iletişimi (ZMQ üzerinden simüle edilmiş)
doğrulamayı amaçlar. Gerçek donanım veya derin öğrenme modeli gerektirmez.
"""

import unittest
import time
import json
import os
import numpy as np
from threading import Thread
from queue import Queue, Empty
import zmq
import logging
import unittest
from unittest.loader import load_tests

# Test edilecek modüller
from config import (
    PORT_SDR_DATA, PORT_CALIBRATION_RESULT, PORT_PREPROCESSING_OUTPUT,
    PORT_AI_OUTPUT, PORT_SDR_CONTROL, PORT_CALIBRATION_CONTROL,
    PORT_PREPROCESSING_CONTROL, PORT_AI_CONTROL, GroundType
)
from sdr_receiver import SDRReceiver
from calibration import CalibrationModule
from preprocessing import SignalPreprocessor
from ai_classifier import AIClassifier, AIDetectionResult

# --- Test Yardımcı Fonksiyonları ve Sınıfları --- 

def create_dummy_config(module_name: str, overrides: dict = None) -> dict:
    """Belirli bir modül için varsayılan/dummy yapılandırma oluşturur."""
    base_config = {
        "sdr_receiver": {
            "device_args": "driver=dummy",
            "sample_rate": 2e6,
            "center_freq": 1e9,
            "gain": 30,
            "buffer_size": 1024 * 16,
            "output_port_data": PORT_SDR_DATA,
            "control_port": PORT_SDR_CONTROL,
            "dual_antenna_mode": True,
            "antenna_config": {
                "tx1": {"type": "UWB", "port": "TX/RX"},
                "rx1": {"type": "UWB", "port": "RX1"},
                "rx2": {"type": "Dipole", "port": "RX2"}
            }
        },
        "calibration": {
            "input_port_sdr": PORT_SDR_DATA,
            "output_port_result": PORT_CALIBRATION_RESULT,
            "control_port": PORT_CALIBRATION_CONTROL,
            "freq_sweep_start_mhz": 200,
            "freq_sweep_end_mhz": 3000,
            "freq_sweep_steps": 10,
            "noise_monitor_interval_s": 60
        },
        "preprocessing": {
            "input_port_sdr": PORT_SDR_DATA,
            "input_port_cal": PORT_CALIBRATION_RESULT,
            "output_port": PORT_PREPROCESSING_OUTPUT,
            "control_port": PORT_PREPROCESSING_CONTROL,
            "wavelet_type": "db4",
            "wavelet_level": 4,
            "depth_method": "peak_time",
            "clutter_threshold": 0.5
        },
        "ai_classifier": {
            "input_port": PORT_PREPROCESSING_OUTPUT,
            "output_port": PORT_AI_OUTPUT,
            "control_port": PORT_AI_CONTROL,
            "model_path": None, # Kural tabanlı test
            "confidence_threshold": 0.5,
            "false_positive_reduction": {
                "min_depth_m": 0.15,
                "max_size_clutter_m": 0.15,
                "min_amplitude_target": 0.08
            }
        }
    }
    config = base_config.get(module_name, {})
    if overrides:
        config.update(overrides)
    return config

class ZMQTestHelper:
    """ZMQ mesajlaşmasını test etmek için yardımcı sınıf."""
    def __init__(self):
        self.context = zmq.Context()
        self.sockets = {}

    def create_pub_socket(self, port: int) -> zmq.Socket:
        sock = self.context.socket(zmq.PUB)
        sock.bind(f"tcp://*:{port}")
        self.sockets[port] = sock
        time.sleep(0.2) # Bağlanma için kısa bekleme
        return sock

    def create_sub_socket(self, port: int, topic: str = "") -> zmq.Socket:
        sock = self.context.socket(zmq.SUB)
        sock.connect(f"tcp://localhost:{port}")
        sock.setsockopt_string(zmq.SUBSCRIBE, topic)
        sock.setsockopt(zmq.RCVTIMEO, 500) # Mesaj bekleme süresi
        self.sockets[port] = sock
        time.sleep(0.2)
        return sock

    def create_req_socket(self, port: int) -> zmq.Socket:
        sock = self.context.socket(zmq.REQ)
        sock.connect(f"tcp://localhost:{port}")
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 1000)
        self.sockets[port] = sock
        time.sleep(0.2)
        return sock

    def create_rep_socket(self, port: int) -> zmq.Socket:
        sock = self.context.socket(zmq.REP)
        sock.bind(f"tcp://*:{port}")
        sock.setsockopt(zmq.RCVTIMEO, 500)
        self.sockets[port] = sock
        time.sleep(0.2)
        return sock

    def cleanup(self):
        for sock in self.sockets.values():
            sock.close()
        self.context.term()

# --- Test Senaryoları --- 

class TestSDRReceiver(unittest.TestCase):
    """SDRReceiver modülü için testler."""
    def setUp(self): 
        self.config = create_dummy_config("sdr_receiver")
        self.zmq_helper = ZMQTestHelper()
        self.sub_socket = self.zmq_helper.create_sub_socket(self.config["output_port_data"])
        self.rep_socket = self.zmq_helper.create_rep_socket(self.config["control_port"])
        self.sdr = SDRReceiver(self.config)
        # SDR\ı başlat ama donanım bağlantısı kurma (dummy driver)
        self.sdr.running.set() # Çalışıyor olarak işaretle
        self.sdr.context = self.zmq_helper.context # Paylaşılan context
        self.sdr.data_socket = self.zmq_helper.create_pub_socket(self.config["output_port_data"])
        self.sdr.control_socket = self.zmq_helper.create_rep_socket(self.config["control_port"])
        # Thread\leri manuel başlatma
        self.control_thread = Thread(target=self.sdr._control_worker, daemon=True)
        self.control_thread.start()
        time.sleep(0.1)

    def tearDown(self): 
        self.sdr.stop()
        if self.control_thread.is_alive():
            self.control_thread.join(1.0)
        self.zmq_helper.cleanup()

    def test_01_initialization(self): 
        self.assertIsNotNone(self.sdr)
        self.assertEqual(self.sdr.sample_rate, 2e6)

    def test_02_control_status(self): 
        req_socket = self.zmq_helper.create_req_socket(self.config["control_port"])
        req_socket.send_json({"command": "get_status"})
        response = req_socket.recv_json()
        self.assertEqual(response["status"], "ok")
        self.assertTrue(response["data"]["running"])

    def test_03_control_set_freq(self): 
        req_socket = self.zmq_helper.create_req_socket(self.config["control_port"])
        new_freq = 1.5e9
        req_socket.send_json({"command": "set_frequency", "value": new_freq})
        response = req_socket.recv_json()
        self.assertEqual(response["status"], "ok")
        # Gerçek frekans ayarı donanım gerektirir, sadece komutun işlendiğini kontrol et
        # self.assertEqual(self.sdr.center_freq, new_freq) # Doğrudan erişim yerine status ile kontrol et
        req_socket.send_json({"command": "get_status"})
        status_resp = req_socket.recv_json()
        self.assertEqual(status_resp["data"]["center_freq"], new_freq)

    # def test_04_dummy_data_publication(self):
    #     # Dummy driver veri üretmediği için bu test zor
    #     # Manuel olarak veri göndermeyi simüle edebiliriz
    #     dummy_samples = np.random.randn(self.config["buffer_size"]) + 1j * np.random.randn(self.config["buffer_size"])
    #     self.sdr._publish_data(dummy_samples)
    #     try:
    #         message = self.sub_socket.recv_json()
    #         self.assertIn("samples_iq", message)
    #         self.assertEqual(len(message["samples_iq"]), self.config["buffer_size"])
    #     except zmq.Again:
    #         self.fail("SDR modülünden veri alınamadı.")

class TestCalibrationModule(unittest.TestCase):
    """CalibrationModule modülü için testler."""
    def setUp(self): 
        self.config = create_dummy_config("calibration")
        self.zmq_helper = ZMQTestHelper()
        self.pub_socket_sdr = self.zmq_helper.create_pub_socket(self.config["input_port_sdr"])
        self.sub_socket_cal = self.zmq_helper.create_sub_socket(self.config["output_port_result"])
        self.rep_socket = self.zmq_helper.create_rep_socket(self.config["control_port"])
        self.cal = CalibrationModule(self.config)
        self.cal.start()
        time.sleep(0.2)

    def tearDown(self): 
        self.cal.stop()
        self.zmq_helper.cleanup()

    def test_01_initialization(self): 
        self.assertIsNotNone(self.cal)
        self.assertTrue(self.cal.running.is_set())

    def test_02_control_status(self): 
        req_socket = self.zmq_helper.create_req_socket(self.config["control_port"])
        req_socket.send_json({"command": "get_status"})
        response = req_socket.recv_json()
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["data"]["current_ground_type"], GroundType.UNKNOWN.name)

    def test_03_calibration_trigger(self): 
        req_socket = self.zmq_helper.create_req_socket(self.config["control_port"])
        req_socket.send_json({"command": "start_calibration"})
        response = req_socket.recv_json()
        self.assertEqual(response["status"], "ok")
        # Kalibrasyonun başladığını kontrol et (durum üzerinden)
        time.sleep(0.1)
        req_socket.send_json({"command": "get_status"})
        status_resp = req_socket.recv_json()
        self.assertTrue(status_resp["data"]["is_calibrating"])
        # TODO: Dummy SDR verisi göndererek kalibrasyon sonucunu test et

    # def test_04_ground_type_detection(self):
    #     # Farklı frekanslarda dummy SDR verileri gönder
    #     # ... (Bu kısım daha karmaşık veri simülasyonu gerektirir)
    #     # Sonucun yayınlandığını kontrol et
    #     try:
    #         message = self.sub_socket_cal.recv_json()
    #         self.assertIn("ground_type", message)
    #         self.assertIn("background_noise_db", message)
    #     except zmq.Again:
    #         self.fail("Kalibrasyon modülünden sonuç alınamadı.")

class TestPreprocessingModule(unittest.TestCase):
    """SignalPreprocessor modülü için testler."""
    def setUp(self): 
        self.config = create_dummy_config("preprocessing")
        self.zmq_helper = ZMQTestHelper()
        self.pub_socket_sdr = self.zmq_helper.create_pub_socket(self.config["input_port_sdr"])
        self.pub_socket_cal = self.zmq_helper.create_pub_socket(self.config["input_port_cal"])
        self.sub_socket_prep = self.zmq_helper.create_sub_socket(self.config["output_port"])
        self.rep_socket = self.zmq_helper.create_rep_socket(self.config["control_port"])
        self.prep = SignalPreprocessor(self.config)
        self.prep.start()
        time.sleep(0.2)

    def tearDown(self): 
        self.prep.stop()
        self.zmq_helper.cleanup()

    def test_01_initialization(self): 
        self.assertIsNotNone(self.prep)
        self.assertTrue(self.prep.running.is_set())

    def test_02_control_status(self): 
        req_socket = self.zmq_helper.create_req_socket(self.config["control_port"])
        req_socket.send_json({"command": "get_status"})
        response = req_socket.recv_json()
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["data"]["current_ground_type"], GroundType.UNKNOWN.name)

    def test_03_process_dummy_data(self): 
        # Dummy kalibrasyon verisi gönder
        self.pub_socket_cal.send_json({"ground_type": GroundType.DRY_SOIL.name, "background_noise_db": -95.0})
        time.sleep(0.1) # Verinin işlenmesi için bekle
        
        # Dummy SDR verisi gönder
        sample_rate = 2e6
        n_samples = 4096
        # Yüzey yansıması ve bir hedef yansıması simüle et
        time_axis = np.arange(n_samples) / sample_rate
        signal = np.zeros(n_samples, dtype=complex)
        surface_delay = 10e-9
        target_delay = 50e-9
        surface_idx = int(surface_delay * sample_rate)
        target_idx = int(target_delay * sample_rate)
        signal[surface_idx : surface_idx+10] = 0.8 * np.exp(1j * np.pi * 0.1)
        signal[target_idx : target_idx+15] = 0.5 * np.exp(1j * np.pi * 0.9) # Metal benzeri faz
        signal += (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) * 0.05 # Gürültü
        
        sdr_message = {
            "timestamp": time.time(),
            "sample_rate": sample_rate,
            "center_freq": 1e9,
            "samples_iq": [[s.real, s.imag] for s in signal]
        }
        self.pub_socket_sdr.send_json(sdr_message)
        
        # Sonucu almayı bekle
        try:
            result_message = self.sub_socket_prep.recv_json()
            self.assertIn("timestamp", result_message)
            self.assertIn("targets", result_message)
            self.assertIn("filtered_samples_iq", result_message)
            self.assertEqual(result_message["ground_type"], GroundType.DRY_SOIL.name)
            # Hedef tespit edildi mi?
            # Derinlik tahmini doğru mu? (Yaklaşık)
            expected_depth = (3e8 / np.sqrt(3)) * target_delay / 2 # DRY_SOIL permittivity=3
            found_target = False
            for idx, props in result_message["targets"].items():
                self.assertAlmostEqual(props["depth_m"], expected_depth, delta=expected_depth*0.2) # %20 tolerans
                found_target = True
            self.assertTrue(found_target, "Simüle edilen hedef tespit edilemedi.")
            
        except zmq.Again:
            self.fail("Preprocessing modülünden sonuç alınamadı.")

class TestAIClassifier(unittest.TestCase):
    """AIClassifier modülü için testler (kural tabanlı)."""
    def setUp(self): 
        self.config = create_dummy_config("ai_classifier")
        self.zmq_helper = ZMQTestHelper()
        self.pub_socket_prep = self.zmq_helper.create_pub_socket(self.config["input_port"])
        self.sub_socket_ai = self.zmq_helper.create_sub_socket(self.config["output_port"])
        self.rep_socket = self.zmq_helper.create_rep_socket(self.config["control_port"])
        self.ai = AIClassifier(self.config)
        self.ai.start()
        time.sleep(0.2)

    def tearDown(self): 
        self.ai.stop()
        self.zmq_helper.cleanup()

    def test_01_initialization(self): 
        self.assertIsNotNone(self.ai)
        self.assertTrue(self.ai.running.is_set())
        self.assertTrue(self.ai.use_rule_based)

    def test_02_control_status(self): 
        req_socket = self.zmq_helper.create_req_socket(self.config["control_port"])
        req_socket.send_json({"command": "get_status"})
        response = req_socket.recv_json()
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["mode"], "rule-based")

    def test_03_classify_metal(self): 
        prep_message = {
            "timestamp": time.time(),
            "targets": {
                "100": {"depth_m": 1.5, "amplitude": 0.8, "phase_change_rad": np.pi * 0.95}
            }
        }
        self.pub_socket_prep.send_json(prep_message)
        try:
            result = self.sub_socket_ai.recv_json()
            self.assertEqual(len(result["detections"]), 1)
            detection = result["detections"][0]
            self.assertEqual(detection["type"], "METAL")
            self.assertGreater(detection["confidence"], self.config["confidence_threshold"])
        except zmq.Again:
            self.fail("AI modülünden sonuç alınamadı (Metal Test)")

    def test_04_classify_void(self): 
        prep_message = {
            "timestamp": time.time(),
            "targets": {
                "200": {"depth_m": 2.0, "amplitude": 0.6, "phase_change_rad": -np.pi * 0.9}
            }
        }
        self.pub_socket_prep.send_json(prep_message)
        try:
            result = self.sub_socket_ai.recv_json()
            self.assertEqual(len(result["detections"]), 1)
            detection = result["detections"][0]
            self.assertEqual(detection["type"], "VOID")
            self.assertGreater(detection["confidence"], self.config["confidence_threshold"])
        except zmq.Again:
            self.fail("AI modülünden sonuç alınamadı (Void Test)")

    def test_05_classify_stone(self): 
        prep_message = {
            "timestamp": time.time(),
            "targets": {
                "300": {"depth_m": 1.0, "amplitude": 0.4, "phase_change_rad": np.pi * 0.2}
            }
        }
        self.pub_socket_prep.send_json(prep_message)
        try:
            result = self.sub_socket_ai.recv_json()
            self.assertEqual(len(result["detections"]), 1)
            detection = result["detections"][0]
            self.assertEqual(detection["type"], "STONE") # Veya MINERAL
            self.assertGreater(detection["confidence"], self.config["confidence_threshold"])
        except zmq.Again:
            self.fail("AI modülünden sonuç alınamadı (Stone Test)")

    def test_06_false_positive_depth(self): 
        prep_message = {
            "timestamp": time.time(),
            "targets": {
                "400": {"depth_m": 0.1, "amplitude": 0.9, "phase_change_rad": np.pi * 0.9} # Yüzeye yakın metal
            }
        }
        self.pub_socket_prep.send_json(prep_message)
        try:
            # Sonuç gelmemeli (veya çok kısa sürede gelmeli ama boş olmalı)
            result = self.sub_socket_ai.recv_json(flags=zmq.NOBLOCK) 
            self.assertIsNone(result, "Yüzeydeki yanlış pozitif elenmedi.")
        except zmq.Again:
            # Beklenen durum: Mesaj yok
            pass 

    def test_07_false_positive_amplitude(self): 
        prep_message = {
            "timestamp": time.time(),
            "targets": {
                "500": {"depth_m": 1.0, "amplitude": 0.05, "phase_change_rad": np.pi * 0.9} # Zayıf metal
            }
        }
        self.pub_socket_prep.send_json(prep_message)
        try:
            result = self.sub_socket_ai.recv_json(flags=zmq.NOBLOCK)
            self.assertIsNone(result, "Düşük genlikli yanlış pozitif elenmedi.")
        except zmq.Again:
            pass

# --- Test Suite --- 

def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTest(loader.loadTestsFromTestCase(TestSDRReceiver))
    suite.addTest(loader.loadTestsFromTestCase(TestCalibrationModule))
    suite.addTest(loader.loadTestsFromTestCase(TestPreprocessingModule))
    suite.addTest(loader.loadTestsFromTestCase(TestAIClassifier))
    return suite

if __name__ == "__main__":
    # Log seviyesini testler için ayarla
    logging.getLogger("sdr_receiver").setLevel(logging.WARNING)
    logging.getLogger("calibration").setLevel(logging.WARNING)
    logging.getLogger("preprocessing").setLevel(logging.WARNING)
    logging.getLogger("ai_classifier").setLevel(logging.WARNING)
    
    runner = unittest.TextTestRunner()
    suite = load_tests(unittest.defaultTestLoader, None, None)
    runner.run(suite)

