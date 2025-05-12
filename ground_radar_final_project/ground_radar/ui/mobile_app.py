from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.network.urlrequest import UrlRequest
from kivy.clock import Clock
import json

# Configuration for connecting to the main desktop application
# This would be the IP address of the machine running main_window.py
# and the port on which it exposes a simple API for mobile interaction.
SERVER_IP = "192.168.1.100" # Placeholder: User needs to configure this
SERVER_PORT = "8080" # Placeholder: Main app needs to run a server on this port
BASE_URL = f"http://{SERVER_IP}:{SERVER_PORT}"

class MobileAppInterface(App):
    """
    Kivy-based mobile application interface for the Ground Radar System.
    Allows basic control and status monitoring of the main GPR application.
    """

    def build(self):
        self.title = "Yeraltı Radar Mobil Kontrol"
        self.main_layout = BoxLayout(orientation="vertical", padding=10, spacing=10)

        # --- Connection Status ---
        self.status_label = Label(text="Bağlantı Durumu: Bağlı Değil", size_hint_y=None, height=40)
        self.main_layout.add_widget(self.status_label)

        # --- Controls ---
        controls_grid = GridLayout(cols=2, size_hint_y=None, height=120, spacing=5)
        self.connect_button = Button(text="Ana Sisteme Bağlan")
        self.connect_button.bind(on_press=self.connect_to_server)
        controls_grid.add_widget(self.connect_button)

        self.start_scan_button = Button(text="Taramayı Başlat", disabled=True)
        self.start_scan_button.bind(on_press=lambda x: self.send_command("/start_scan"))
        controls_grid.add_widget(self.start_scan_button)

        self.stop_scan_button = Button(text="Taramayı Durdur", disabled=True)
        self.stop_scan_button.bind(on_press=lambda x: self.send_command("/stop_scan"))
        controls_grid.add_widget(self.stop_scan_button)
        
        self.get_status_button = Button(text="Durum Al", disabled=True)
        self.get_status_button.bind(on_press=lambda x: self.send_command("/status"))
        controls_grid.add_widget(self.get_status_button)
        self.main_layout.add_widget(controls_grid)

        # --- Log/Response Area ---
        self.log_label = Label(text="Sunucu Yanıtları / Loglar:", size_hint_y=None, height=30)
        self.main_layout.add_widget(self.log_label)

        self.scroll_view = ScrollView(size_hint=(1, 1))
        self.log_output = Label(text="", size_hint_y=None, markup=True)
        self.log_output.bind(texture_size=self.log_output.setter('size'))
        self.scroll_view.add_widget(self.log_output)
        self.main_layout.add_widget(self.scroll_view)
        
        self.log_message("Mobil arayüz başlatıldı. Ana sisteme bağlanın.")
        # Periodically try to check server status if not connected
        # Clock.schedule_interval(self.check_server_status_periodically, 5) 
        return self.main_layout

    def log_message(self, message, is_error=False):
        color = "ff3333" if is_error else "ffffff"
        self.log_output.text += f"[color={color}]{message}[/color]\n"
        # Auto-scroll to bottom
        self.scroll_view.scroll_y = 0
        print(f"MobileUI: {message}")

    def on_request_success(self, req, result):
        self.log_message(f"Sunucu Yanıtı ({req.url}):\n{json.dumps(result, indent=2)}")
        status = result.get("status", "unknown")
        message = result.get("message", "")
        sdr_connected = result.get("sdr_connected", False)
        scan_active = result.get("scan_active", False)

        self.status_label.text = f"Bağlantı Durumu: Bağlı - {message[:30]}"
        self.start_scan_button.disabled = not sdr_connected or scan_active
        self.stop_scan_button.disabled = not sdr_connected or not scan_active
        self.get_status_button.disabled = False
        self.connect_button.disabled = True

    def on_request_failure(self, req, result):
        self.log_message(f"Sunucu Hatası ({req.url}): {result}", is_error=True)
        self.status_label.text = "Bağlantı Durumu: Sunucuya Ulaşılamadı"
        self.reset_button_states()

    def on_request_error(self, req, error):
        self.log_message(f"Ağ Hatası ({req.url}): {error}", is_error=True)
        self.status_label.text = "Bağlantı Durumu: Ağ Hatası"
        self.reset_button_states()

    def send_command(self, endpoint, data=None):
        url = f"{BASE_URL}{endpoint}"
        headers = {"Content-Type": "application/json"}  # Sözlük anahtarlarını ve değerlerini doğru şekilde tanımlayın
        body = json.dumps(data) if data else None  # Eğer data None değilse JSON formatına dönüştürün
        self.log_message(f"Komut gönderiliyor: {url} {body if body else 'No data'}")
        try:
            UrlRequest(
                url,
                on_success=self.on_request_success,
                on_failure=self.on_request_failure,
                on_error=self.on_request_error,
                req_body=body,
                req_headers=headers,
                timeout=10 # seconds
            )
        except Exception as e:
            self.log_message(f"UrlRequest oluşturulurken hata: {e}", is_error=True)

    def connect_to_server(self, instance):
        self.log_message(f"Ana sisteme ({BASE_URL}) bağlanmaya çalışılıyor...")
        self.send_command("/status") # Initial command to check status and connect

    def reset_button_states(self):
        self.start_scan_button.disabled = True
        self.stop_scan_button.disabled = True
        self.get_status_button.disabled = True
        self.connect_button.disabled = False

    # def check_server_status_periodically(self, dt):
    #     # This can be too noisy, better to rely on user action or WebSocket for live status
    #     if self.connect_button.disabled == False: # Only if not already connected
    #         self.log_message("Periyodik sunucu durumu kontrol ediliyor...")
    #         self.send_command("/status")

if __name__ == "__main__":
    # Note: To run this, the main_window.py would need to run a simple HTTP server
    # (e.g., using Flask or http.server) to respond to these /status, /start_scan, /stop_scan requests.
    # This mobile_app.py is the client side.
    MobileAppInterface().run()

