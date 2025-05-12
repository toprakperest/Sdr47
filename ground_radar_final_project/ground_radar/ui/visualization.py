import plotly.graph_objects as go
import colorsys
from PyQt5.QtWebEngineWidgets import QWebEngineView
from collections import defaultdict
import numpy as np
import plotly.io as pio
from PyQt5.QtWidgets import QWidget, QVBoxLayout


class LayerVisualizer:
    """
    AI tarafından belirlenen katmanları Z ekseninde görselleştirir
    """
    DEFAULT_COLORS = {
        "Toprak": "#a0522d",  # Sienna
        "Taş": "#808080",  # Gray
        "Boşluk": "#add8e6",  # LightBlue
        "Metal": "#8b0000",  # DarkRed
        "Su": "#0000ff",  # Blue
        "Kök": "#2e8b57",  # SeaGreen
        "Beton": "#696969"  # DimGray
    }

    def __init__(self):
        self.fig = go.Figure()
        self.max_depth = 0

    def visualize_layers(self, layers, title="Z Ekseninde Katman Görselleştirmesi"):
        """
        Katmanları görselleştirir

        Args:
            layers: [("Katman Adı", başlangıç_derinliği, bitiş_derinliği), ...]
            title: Görsel başlığı
        """
        # Katmanları doğrulayıp sırala
        validated_layers = self._validate_and_sort_layers(layers)
        if not validated_layers:
            raise ValueError("Geçersiz katman verisi")

        # Görsel elemanları oluştur
        self._create_visual_elements(validated_layers)

        # Layout ayarları
        self._configure_layout(title)

        return self.fig

    def _validate_and_sort_layers(self, layers):
        """Katman verisini doğrular ve derinliğe göre sıralar"""
        if not layers:
            return []

        try:
            # Katmanları başlangıç derinliğine göre sırala
            sorted_layers = sorted(
                [(str(name), float(start), float(end)) for name, start, end in layers],
                key=lambda x: x[1]
            )

            # Maksimum derinliği kaydet
            self.max_depth = max(end for _, _, end in sorted_layers)
            return sorted_layers

        except Exception as e:
            print(f"Katman veri hatası: {e}")
            return []

    def _generate_color(self, name):
        """Bilinmeyen katmanlar için rastgele ama tutarlı renk oluşturur"""
        hue = (hash(name) % 360) / 360.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.85)
        return f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"

    def _create_visual_elements(self, layers):
        """Görsel elemanları ve etiketleri oluşturur"""
        annotations = []
        x_pos = 0  # Sabit genişlik için sabit x pozisyonu

        for name, start, end in layers:
            # Katman rengini belirle
            color = self.DEFAULT_COLORS.get(name, self._generate_color(name))

            # Katman çubuğunu ekle
            self.fig.add_trace(go.Bar(
                x=[x_pos],
                y=[end - start],  # Katman yüksekliği
                base=start,  # Başlangıç derinliği
                marker_color=color,
                hoverinfo="text",
                hovertext=self._generate_hover_text(name, start, end),
                width=0.8,  # Sabit genişlik
                name=name,
                showlegend=False
            ))

            # Katman etiketi ekle
            annotations.append(dict(
                x=x_pos,
                y=(start + end) / 2,
                text=f"{name}<br>({int(start)}–{int(end)} cm)",
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="white" if self._is_dark_color(color) else "black")
            ))

        self.fig.update_layout(annotations=annotations)

    def _generate_hover_text(self, name, start, end):
        """Fare üzerine gelindiğinde gösterilecek detaylı bilgi"""
        return (f"<b>{name}</b><br>"
                f"Derinlik: {start:.1f} - {end:.1f} cm<br>"
                f"Kalınlık: {end - start:.1f} cm")

    def _configure_layout(self, title):
        """Görsel düzenini ayarlar"""
        y_tick = max(10, int(self.max_depth / 10)) * 10  # Y ekseni aralığını belirle

        self.fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            barmode="stack",
            xaxis=dict(
                visible=False  # X ekseni gizlendi
            ),
            yaxis=dict(
                title="Derinlik (cm)",
                autorange="reversed",  # En yüzey yukarıda
                tickmode="linear",
                dtick=y_tick,
                range=[self.max_depth * 1.1, 0]  # %10 fazlasına kadar göster
            ),
            height=600,
            margin=dict(l=50, r=50, b=50, t=100),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                align="left"
            )
        )

    def _is_dark_color(self, color_str):
        """Renk koyu mu açık mı belirler (etiket rengi için)"""
        if color_str.startswith('rgb('):
            parts = color_str[4:-1].split(',')
            r, g, b = map(int, parts)
        else:
            r, g, b = tuple(int(color_str.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return luminance < 0.5

    def show_in_qt(self, parent_widget):
        """PyQt5'te göstermek için QWebEngineView oluşturur"""
        web_view = QWebEngineView(parent_widget)
        web_view.setHtml(self.fig.to_html(
            include_plotlyjs='cdn',
            full_html=False,
            default_width='100%',
            default_height='600px'
        ))
        return web_view

    def export(self, filename="katman_goruntusu.png", width=800, height=600):
        """Görseli dosyaya kaydeder"""
        self.fig.write_image(filename, width=width, height=height, scale=2)


class RadarVisualizationWidget(QWidget):
    """
    Radar verilerini ve katman görselleştirmesini gösteren PyQt widget'ı
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # B-scan görselleştirme
        self.bscan_view = QWebEngineView()

        # Katman görselleştirme
        self.layer_view = QWebEngineView()

        # Kontrol paneli
        self.control_panel = QWidget()
        control_layout = QVBoxLayout()
        self.control_panel.setLayout(control_layout)

        # Widget'ları düzenle
        self.layout.addWidget(self.bscan_view)
        self.layout.addWidget(self.layer_view)
        self.layout.addWidget(self.control_panel)

    def update_bscan(self, scan_data):
        """B-scan verisini günceller"""
        fig = go.Figure(data=go.Heatmap(
            z=scan_data,
            colorscale='Viridis',
            hoverongaps=False
        ))
        fig.update_layout(
            title="B-Scan Görünümü",
            xaxis_title="Tarama Noktası",
            yaxis_title="Derinlik (Örnek)"
        )
        self.bscan_view.setHtml(fig.to_html(full_html=False))

    def update_layers(self, layers):
        """Katman verisini günceller"""
        visualizer = LayerVisualizer()
        fig = visualizer.visualize_layers(layers)
        self.layer_view.setHtml(fig.to_html(full_html=False))