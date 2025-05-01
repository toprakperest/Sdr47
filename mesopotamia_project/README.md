# Mesopotamia SDR Yeraltı Algılama Sistemi

Version: 1.0 (Geliştirilmiş)
Author: AI Assistant (Manus) & toprakperest
Last Updated: 2025-04-30

## Genel Bakış

Bu proje, HamGeek Zynq7020+AD9363 gibi SDR donanımlarını kullanarak yeraltındaki farklı malzemeleri (metal, boşluk, taş vb.) tespit etmek ve 3D olarak görselleştirmek için geliştirilmiş Python tabanlı bir sistemdir. Sistem, modüler bir yapıya sahip olup ZMQ üzerinden iletişim kuran farklı bileşenlerden oluşur:

- **SDR Alıcı (`sdr_receiver.py`):** SoapySDR kullanarak donanımdan veri alır, temel filtreleme ve kazanç kontrolü yapar.
- **Kalibrasyon (`calibration.py`):** Zemin özelliklerini (dielektrik sabiti vb.) tahmin eder ve arka plan gürültüsünü izler.
- **Ön İşleme (`preprocessing.py`):** Gürültü azaltma (Wavelet), yankı tespiti, derinlik tahmini ve clutter filtreleme yapar.
- **AI Sınıflandırıcı (`ai_classifier.py`):** Tespit edilen yankıları sınıflandırır (metal, boşluk vb.) ve yanlış pozitifleri azaltır.
- **Görselleştirme (`depth_3d_view.py`):** Tespit edilen hedefleri 3D olarak görselleştirir (Matplotlib/Mayavi).
- **Kullanıcı Arayüzü (`radar_ui.py`):** Sistemi kontrol etmek ve sonuçları görüntülemek için PyQt5 tabanlı arayüz.
- **Ana Yönetici (`main.py`):** Tüm modülleri başlatır, yönetir ve aralarındaki iletişimi koordine eder.
- **Yapılandırma (`config.py`, `system_config.json`):** Sistem ve modül ayarlarını içerir.
- **Diğer Yardımcı Modüller:** `mobile_sync.py`, `monitoring.py`, `test_system.py`.

## Kurulum (Windows - PyCharm & Anaconda)

Detaylı kurulum adımları için `windows_setup_guide.md` dosyasına bakın. Genel adımlar:

1.  **Anaconda Kurulumu:** Anaconda'yı [resmi sitesinden](https://www.anaconda.com/products/distribution) indirin ve kurun.
2.  **Yeni Conda Ortamı Oluşturma:**
    ```bash
    conda create --name mesopotamia python=3.9 # Veya uyumlu Python sürümü
    conda activate mesopotamia
    ```
3.  **Gerekli Kütüphanelerin Kurulumu:** Proje kök dizinindeyken:
    ```bash
    pip install -r requirements.txt
    ```
    *Not: `SoapySDR` ve donanımınıza özel sürücülerin (örn. Zynq için) ayrıca kurulması gerekebilir. `windows_setup_guide.md` dosyasına bakın.*
4.  **PyCharm Kurulumu:** PyCharm Community veya Professional sürümünü [JetBrains sitesinden](https://www.jetbrains.com/pycharm/download/) indirin ve kurun.
5.  **Projeyi PyCharm'da Açma:** Proje klasörünü PyCharm ile açın.
6.  **Conda Ortamını Ayarlama:** PyCharm'da `File > Settings > Project: mesopotamia_project > Python Interpreter` yolunu izleyerek oluşturduğunuz `mesopotamia` conda ortamını seçin.

## Kullanım

1.  **Yapılandırma:** `Toprakperest/system_config.json` dosyasını donanımınıza ve tercihlerinize göre düzenleyin (SDR sürücü argümanları, antenler, frekanslar vb.).
2.  **Sistemi Çalıştırma:** Ana yönetici betiğini PyCharm üzerinden veya terminalden çalıştırın:
    ```bash
    python Toprakperest/main.py --config Toprakperest/system_config.json
    ```
3.  **Arayüz:** `radar_ui.py` tarafından sağlanan arayüz otomatik olarak açılacaktır. Buradan kalibrasyonu başlatabilir, taramayı kontrol edebilir ve sonuçları görebilirsiniz.

## Geliştirmeler ve Notlar

- Bu sürüm, orijinal GitHub deposundaki kodlar temel alınarak AI tarafından önemli ölçüde yeniden yapılandırılmış ve geliştirilmiştir.
- Modüller arası iletişim ZMQ kullanılarak daha sağlam hale getirilmiştir.
- Sinyal işleme, sınıflandırma ve yanlış pozitif azaltma algoritmaları eklenmiş veya iyileştirilmiştir.
- Kapsamlı hata yönetimi ve loglama eklenmiştir.
- Kodlar Türkçe olarak belgelendirilmiştir.
- **Önemli:** Sistem, gerçek SDR donanımı ve antenlerle test edilmemiştir. Donanım entegrasyonu, sürücü uyumluluğu ve parametre ayarları (kazanç, filtreler, eşik değerleri) gerçek dünya koşullarında ince ayar gerektirebilir.
- AI modeli yükleme kısmı placeholder olarak bırakılmıştır; gerçek bir model eğitilip entegre edilmelidir.

## Dosya Yapısı

```
mesopotamia_project/
├── Toprakperest/           # Ana kod modülleri
│   ├── __init__.py
│   ├── main.py             # Ana yönetici
│   ├── config.py           # Varsayılan yapılandırma
│   ├── system_config.json  # Kullanıcı yapılandırması
│   ├── sdr_receiver.py     # SDR veri alımı
│   ├── calibration.py      # Zemin kalibrasyonu
│   ├── preprocessing.py    # Sinyal ön işleme
│   ├── ai_classifier.py    # AI sınıflandırma
│   ├── depth_3d_view.py    # 3D Görselleştirme
│   ├── radar_ui.py         # Kullanıcı arayüzü
│   ├── mobile_sync.py      # Mobil senkronizasyon (Temel)
│   ├── monitoring.py       # Sistem izleme
│   └── test_system.py      # Test betikleri
├── requirements.txt        # Gerekli Python kütüphaneleri
├── windows_setup_guide.md  # Windows kurulum rehberi
├── todo.md                 # Geliştirme görev listesi
└── README.md               # Bu dosya
```

