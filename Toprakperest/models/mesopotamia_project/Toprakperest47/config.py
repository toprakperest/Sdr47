#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
config.py - Centralized Configuration, Constants, and Enums

Version: 2.0
Author: AI Assistant (Manus)
Last Updated: 2025-04-30

SDR yeraltı tespit sistemi için merkezi yapılandırma, sabitler, enumlar,
donanımsal özellikler ve varsayılan değerleri tanımlar.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List

# --- Sistem Modları ve Durumları ---

class SystemMode(Enum):
    """Sistem çalışma modları"""
    STANDBY = auto()      # Beklemede, modüller hazır ama aktif değil
    CALIBRATION = auto()  # Kalibrasyon işlemi yapılıyor
    SCANNING = auto()     # Aktif tarama ve veri işleme
    DIAGNOSTIC = auto()   # Donanım/yazılım tanılama modu
    MAINTENANCE = auto()  # Bakım modu
    TEST = auto()         # Test modu (örn. test_system.py için)

class ModuleStatus(Enum):
    """Modül çalışma durumları"""
    STOPPED = auto()      # Durduruldu
    INITIALIZING = auto() # Başlatılıyor
    RUNNING = auto()      # Çalışıyor
    ERROR = auto()        # Hata durumunda
    RECOVERING = auto()   # Kurtarma/Yeniden başlatma deneniyor
    TERMINATING = auto()  # Durduruluyor

# --- Tespit ve Zemin Tipleri --- 

# TARGET_TYPES AI sınıflandırıcının kullanacağı hedef etiketleri
TARGET_TYPES: List[str] = [
    "METAL",
    "VOID",       # Boşluk
    "STONE",      # Taş
    "MINERAL",    # Mineral / Hot Rock
    "WATER",      # Su / Yüksek Nem
    "CLUTTER",    # Metal çöp, kök, düzlemsel yansıma vb.
    "UNKNOWN"     # Bilinmeyen
]

class GroundType(Enum):
    """Zemin türleri (Kalibrasyon ve Ön İşleme için)"""
    DRY_SOIL = "dry_soil"
    WET_SOIL = "wet_soil"
    LIMESTONE = "limestone"     # Kireçli
    IRON_OXIDE = "iron_oxide"   # Demir Oksitli
    MINERAL_RICH = "mineral_rich" # Mineral Gürültülü
    ROCKY = "rocky"         # Kayalık
    SANDY = "sandy"         # Kumlu
    CLAY = "clay"           # Killi
    MIXED = "mixed"         # Karışık
    UNKNOWN = "unknown"     # Bilinmeyen / Kalibre edilmemiş

# --- Donanım Özellikleri --- 

@dataclass
class AntennaSpec:
    """Anten özelliklerini tanımlayan veri sınıfı."""
    name: str
    type: str # Örn: 'dipole', 'uwb_directional'
    role: str # Örn: 'tx', 'rx', 'noise_ref'
    frequency_range_mhz: Tuple[float, float]
    gain_db: float | None = None
    beamwidth_deg: Tuple[float, float] | None = None # (Yatay, Dikey)
    telescopic: bool = False

# Kullanıcı tarafından belirtilen antenlerin tanımlanması
ANTENNA_TX1 = AntennaSpec(
    name="UWB_TX1",
    type="uwb_directional",
    role="tx",
    frequency_range_mhz=(400.0, 8000.0),
    # gain_db ve beamwidth_deg datasheet\ten eklenmeli
)

ANTENNA_RX1 = AntennaSpec(
    name="UWB_RX1",
    type="uwb_directional",
    role="rx",
    frequency_range_mhz=(400.0, 8000.0),
    # gain_db ve beamwidth_deg datasheet\ten eklenmeli
)

ANTENNA_RX2_DIPOLE = AntennaSpec(
    name="Dipole_RX2",
    type="dipole",
    role="noise_ref", # Gürültü referansı ve başlangıç kalibrasyonu için
    frequency_range_mhz=(70.0, 6000.0), # Geniş aralık, teleskopik uzunluğa göre ayarlanır
    telescopic=True
)

HARDWARE_SPECS = {
    "sdr": {
        "name": "HamGeek Zynq7020+AD9363",
        "frequency_range_mhz": (70.0, 6000.0),
        "tx_support": True,
        "rx_support": True,
        "default_driver_args": "" # Biliniyorsa SoapySDR argümanları eklenebilir
    },
    "lna": {
        "name": "HamGeek MT1129 TQP3M9037",
        "gain_db": 20, # Örnek kazanç, datasheet kontrol edilmeli
        "noise_figure_db": 0.6 # Örnek NF, datasheet kontrol edilmeli
    },
    "antennas": {
        "tx1": ANTENNA_TX1,
        "rx1": ANTENNA_RX1,
        "rx2": ANTENNA_RX2_DIPOLE
    }
}

# --- Varsayılan Yollar ve Sabitler --- 

DEFAULT_CONFIG_FILENAME = "system_config.json"
DEFAULT_LOG_DIR = "logs"
DEFAULT_MODEL_DIR = "models"
DEFAULT_CALIBRATION_DIR = "calibration_data"
DEFAULT_SAVE_DIR = "scan_results"

# SDR Varsayılanları (config dosyasından override edilebilir)
DEFAULT_SDR_SAMPLE_RATE = 20e6 # 20 MS/s
DEFAULT_SDR_BANDWIDTH = 15e6 # 15 MHz
DEFAULT_SDR_GAIN = 30 # dB

# Tarama ve Derinlik
DEFAULT_SCAN_FREQ_START_MHZ = 200.0
DEFAULT_SCAN_FREQ_END_MHZ = 3000.0
DEFAULT_MAX_DEPTH_METERS = 5.0
DEFAULT_PERMITTIVITY = 6.0 # Ortalama bir değer

# --- İletişim Portları (ZMQ) --- 
# Tüm modüllerin aynı portları kullanması kritik öneme sahip
PORT_BASE = 5550
PORT_SDR_DATA = PORT_BASE + 0
PORT_SDR_CONTROL = PORT_BASE + 1
PORT_CALIBRATION_RESULT = PORT_BASE + 2
PORT_CALIBRATION_CONTROL = PORT_BASE + 3
PORT_PREPROCESSING_OUTPUT = PORT_BASE + 4 # AI ve UI için girdi
PORT_PREPROCESSING_CONTROL = PORT_BASE + 5
PORT_AI_OUTPUT = PORT_BASE + 6 # UI için girdi
PORT_AI_CONTROL = PORT_BASE + 7
PORT_MONITORING_DATA = PORT_BASE + 8 # UI için girdi
PORT_MONITORING_CONTROL = PORT_BASE + 9
PORT_UI_CONTROL = PORT_BASE + 10 # Ana kontrol için
# PORT_MOBILE_SYNC = PORT_BASE + 11 # Opsiyonel

# --- Görselleştirme Ayarları --- 

# Zemin tipleri için renk haritası (RGB, 0-1 aralığında)
COLOR_MAP_GROUND = {
    GroundType.DRY_SOIL: (0.8, 0.7, 0.5),      # Açık Kahve
    GroundType.WET_SOIL: (0.3, 0.4, 0.2),      # Koyu Kahve/Yeşil
    GroundType.LIMESTONE: (0.85, 0.85, 0.85),  # Açık Gri
    GroundType.IRON_OXIDE: (0.7, 0.3, 0.2),    # Kırmızımsı Kahve
    GroundType.MINERAL_RICH: (0.9, 0.8, 0.4),  # Altın/Sarımsı
    GroundType.ROCKY: (0.5, 0.5, 0.5),         # Gri
    GroundType.SANDY: (0.9, 0.85, 0.7),       # Açık Sarı/Kum
    GroundType.CLAY: (0.6, 0.5, 0.4),         # Kil Kahvesi
    GroundType.MIXED: (0.6, 0.55, 0.45),      # Karışık Kahve
    GroundType.UNKNOWN: (0.7, 0.7, 0.7)       # Orta Gri
}

# Tespit tipleri için renk haritası (RGB, 0-1 aralığında)
COLOR_MAP_DETECTION = {
    "METAL": (1.0, 0.8, 0.0),   # Sarı/Turuncu
    "VOID": (0.2, 0.5, 1.0),    # Mavi
    "STONE": (0.4, 0.3, 0.2),   # Koyu Kahve
    "MINERAL": (1.0, 0.4, 0.8), # Pembe/Magenta (Hot rocks için)
    "WATER": (0.0, 0.8, 1.0),   # Açık Mavi (Cyan)
    "CLUTTER": (0.6, 0.6, 0.6), # Gri
    "UNKNOWN": (1.0, 0.0, 0.0)  # Kırmızı
}

# --- Diğer Sistem Sabitleri --- 

MAX_MODULE_START_TIME_SEC = 15 # Bir modülün başlaması için max süre (sn)
HEARTBEAT_INTERVAL_SEC = 5   # Modül sağlık kontrol aralığı (sn)
MAX_RESTART_ATTEMPTS = 3     # Bir modül için max yeniden başlatma denemesi
RESOURCE_CHECK_INTERVAL_SEC = 10 # Kaynak kullanım kontrol aralığı (sn)

