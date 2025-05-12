import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QComboBox,
                             QSpinBox, QDoubleSpinBox, QGroupBox, QTabWidget)
from PyQt5.QtCore import QTimer
from ui.visualization import RadarVisualizationWidget
from hardware.sdr_controller import SDRController
from algorithms.signal_processing import SignalProcessing
from algorithms.hybrid_analysis import HybridAnalysis
from data.calibration import CalibrationData
from algorithms.ai_models import AIModels


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BD-GPR Arayüzü")
        self.setGeometry(100, 100, 1200, 800)

        # Donanım ve işlem modülleri
        self.sdr_controller = SDRController()
        self.signal_processor = SignalProcessing(sample_rate=20e6)
        self.calibration = CalibrationData(sdr_controller_ref=self.sdr_controller)
        self.hybrid_analyzer = HybridAnalysis(
            signal_processor=self.signal_processor,
            ai_model_handler=AIModels(),
            calibration_module=self.calibration
        )

        # Kullanıcı arayüzü
        self.init_ui()

        # Tarama zamanlayıcısı
        self.scan_timer = QTimer()
        self.scan_timer.timeout.connect(self.perform_scan)
        self.scan_data = []
        self.current_position = 0

    def init_ui(self):
        # Ana widget ve layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Sol panel - Kontroller
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)

        # Sağ panel - Görselleştirme
        self.visualization = RadarVisualizationWidget()

        # Kontrol elemanlarını ekle
        self.setup_control_panel(control_layout)

        # Ana layout'a ekle
        main_layout.addWidget(control_panel, stretch=1)
        main_layout.addWidget(self.visualization, stretch=3)

        # Durum çubuğu
        self.statusBar().showMessage("Hazır")

    def setup_control_panel(self, layout):
        # Bağlantı grubu
        connection_group = QGroupBox("SDR Bağlantı")
        connection_layout = QVBoxLayout()

        self.btn_connect = QPushButton("Bağlan")
        self.btn_connect.clicked.connect(self.connect_sdr)
        connection_layout.addWidget(self.btn_connect)

        connection_group.setLayout(connection_layout)
        layout.addWidget(connection_group)

        # Tarama ayarları grubu
        scan_group = QGroupBox("Tarama Ayarları")
        scan_layout = QVBoxLayout()

        self.freq_selector = QDoubleSpinBox()
        self.freq_selector.setRange(50, 6000)
        self.freq_selector.setValue(433)
        self.freq_selector.setSuffix(" MHz")
        scan_layout.addWidget(QLabel("Merkez Frekans:"))
        scan_layout.addWidget(self.freq_selector)

        self.gain_selector = QSpinBox()
        self.gain_selector.setRange(0, 70)
        self.gain_selector.setValue(30)
        self.gain_selector.setSuffix(" dB")
        scan_layout.addWidget(QLabel("RX Kazancı:"))
        scan_layout.addWidget(self.gain_selector)

        self.btn_start = QPushButton("Tarama Başlat")
        self.btn_start.clicked.connect(self.start_scan)
        scan_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Tarama Durdur")
        self.btn_stop.clicked.connect(self.stop_scan)
        self.btn_stop.setEnabled(False)
        scan_layout.addWidget(self.btn_stop)

        scan_group.setLayout(scan_layout)
        layout.addWidget(scan_group)

        # Kalibrasyon grubu
        cal_group = QGroupBox("Kalibrasyon")
        cal_layout = QVBoxLayout()

        self.btn_ground_cal = QPushButton("Zemin Kalibrasyonu")
        self.btn_ground_cal.clicked.connect(self.perform_ground_calibration)
        cal_layout.addWidget(self.btn_ground_cal)

        cal_group.setLayout(cal_layout)
        layout.addWidget(cal_group)

        # Analiz grubu
        analysis_group = QGroupBox("Analiz")
        analysis_layout = QVBoxLayout()

        self.terrain_selector = QComboBox()
        self.terrain_selector.addItems(["Otomatik", "Kalker", "Bazalt", "Kil", "Kumtaşı"])
        analysis_layout.addWidget(QLabel("Zemin Türü:"))
        analysis_layout.addWidget(self.terrain_selector)

        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

    def connect_sdr(self):
        self.statusBar().showMessage("SDR'a bağlanıyor...")
        success, message = self.sdr_controller.connect()
        self.statusBar().showMessage(message)

        if success:
            self.btn_connect.setEnabled(False)
            self.btn_connect.setText("Bağlı")
            self.btn_start.setEnabled(True)

            # Frekans ve kazancı ayarla
            self.sdr_controller.set_center_frequency(self.freq_selector.value() * 1e6)
            self.sdr_controller.set_rx_gain(self.gain_selector.value())

    def start_scan(self):
        self.scan_data = []
        self.current_position = 0
        self.scan_timer.start(100)  # 100 ms'de bir tarama
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.statusBar().showMessage("Tarama başladı...")

    def stop_scan(self):
        self.scan_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.statusBar().showMessage("Tarama durduruldu")

    def perform_scan(self):
        try:
            if not hasattr(self, 'sdr_controller') or not self.sdr_controller.sdr:
                if not self.connect_sdr():
                    return

            # Ham veriyi al (timeout ekli)
            raw_data = self.sdr_controller.capture_data(timeout=5.0)
            if raw_data is None or len(raw_data) == 0:
                self.show_error_message("Veri alınamadı!")
                return

            # İşle
            processed_data = self.signal_processor.process(raw_data)

            if processed_data is not None:
                self.visualize_results(processed_data)
            else:
                self.show_warning_message("Veri işleme başarısız!")

        except Exception as e:
            error_msg = f"Tarama hatası: {str(e)}"
            print(error_msg)
            self.show_error_message(error_msg)

            # Hata durumunda SDR'yi sıfırla (önemli!)
            self.reset_sdr_connection()
            return

        # Sinyal işleme
        processed_data = self.signal_processor.process(raw_data)

        # AI analizi
        ai_predictions = self.hybrid_analyzer.analyze(processed_data)

        # Görselleştirme
        self.visualization.update_bscan(np.abs(processed_data))

        # Katman görselleştirme (örnek veri)
        example_layers = [
            ("Toprak", 0, 45),
            ("Taş", 45, 100),
            ("Boşluk", 100, 210)
        ]
        self.visualization.update_layers(example_layers)

        # Tarama pozisyonunu güncelle
        self.current_position += 1
        self.statusBar().showMessage(f"Tarama devam ediyor... Pozisyon: {self.current_position}")

    def perform_ground_calibration(self):
        self.statusBar().showMessage("Zemin kalibrasyonu başlatıldı...")
        profile = self.calibration.perform_rx2_ground_calibration()

        if profile:
            terrain = profile.get("estimated_terrain_type", "Bilinmiyor")
            self.statusBar().showMessage(f"Kalibrasyon tamamlandı. Zemin: {terrain}")
            self.hybrid_analyzer.set_current_terrain_type(terrain)
        else:
            self.statusBar().showMessage("Kalibrasyon başarısız!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())