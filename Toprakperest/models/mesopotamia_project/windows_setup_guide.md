# Windows Geliştirme Ortamı Kurulumu (Anaconda & PyCharm)

Bu rehber, projenin Windows işletim sisteminde Anaconda ve PyCharm kullanılarak geliştirilmesi için gerekli adımları içerir.

## 1. Gerekli Yazılımlar

*   **Anaconda:** Python ve paket yönetimi için. [Anaconda İndirme Sayfası](https://www.anaconda.com/products/distribution)
*   **PyCharm:** Python için entegre geliştirme ortamı (IDE). Community veya Professional sürümünü kullanabilirsiniz. [PyCharm İndirme Sayfası](https://www.jetbrains.com/pycharm/download/)
*   **Git:** Versiyon kontrol sistemi için. [Git İndirme Sayfası](https://git-scm.com/download/win)
*   **SoapySDR ve Donanım Sürücüleri:** Kullandığınız SDR donanımı (HamGeek Zynq7020+AD9363) için Windows sürücülerini ve SoapySDR kütüphanesini kurmanız gerekmektedir. Bu kurulum donanıma özeldir ve genellikle üreticinin sağladığı talimatları veya Pothos SDR geliştirme ortamını kurmayı içerir:
    *   **Pothos SDR:** SoapySDR ve çeşitli donanım desteklerini içeren bir Windows dağıtımıdır. [Pothos SDR İndirme Sayfası](https://github.com/pothosware/PothosSDR/wiki/Downloads)
    *   SoapySDR'nin sistem PATH'inde veya Anaconda ortamında erişilebilir olduğundan emin olun.

## 2. Anaconda Ortamı Oluşturma

1.  **Anaconda Prompt'u Açın:** Başlat menüsünden "Anaconda Prompt" uygulamasını yönetici olarak çalıştırın.
2.  **Yeni Ortam Oluşturun:** Proje için izole bir Python ortamı oluşturun (Python 3.10 veya 3.11 önerilir):
    ```bash
    conda create --name mesopotamia_sdr python=3.10
    ```
3.  **Ortamı Aktifleştirin:**
    ```bash
    conda activate mesopotamia_sdr
    ```

## 3. Proje Kodlarını Alma

1.  **Proje Dizini Oluşturun:** Proje dosyalarını saklamak için bir klasör oluşturun (örneğin, `C:\Projects\MesopotamiaSDR`).
2.  **Git ile Klonlama:** Anaconda Prompt'ta proje dizinine gidin ve GitHub deposunu klonlayın:
    ```bash
    cd C:\Projects\MesopotamiaSDR
    git clone https://github.com/toprakperest/Sdr47.git .
    ```
    *Not: Eğer kodları zip olarak aldıysanız, bu adıma gerek yoktur. Dosyaları proje dizinine çıkarmanız yeterlidir.*

## 4. Gerekli Python Paketlerini Kurma

1.  **Ortamın Aktif Olduğundan Emin Olun:** Anaconda Prompt'ta `(mesopotamia_sdr)` ifadesinin göründüğünden emin olun.
2.  **`requirements.txt` Kullanarak Kurulum:** Proje dizinindeki `requirements.txt` dosyasını kullanarak gerekli paketleri kurun:
    ```bash
    cd C:\Projects\MesopotamiaSDR\Toprakperest
    pip install -r requirements.txt
    ```
3.  **İsteğe Bağlı Paketler:** Görselleştirme veya GUI için ek paketler gerekiyorsa (Plotly, PyVista, PyQt5, PyQtGraph), bunları da `pip` veya `conda` ile kurun:
    ```bash
    # Örnekler:
    pip install plotly pyvista vtk
    pip install PyQt5 pyqtgraph
    # veya conda ile:
    # conda install plotly pyvista vtk
    # conda install pyqt5 pyqtgraph
    ```
    *Not: PyQtGraph genellikle pip ile daha sorunsuz kurulur.*

## 5. PyCharm Projesini Yapılandırma

1.  **PyCharm'ı Açın.**
2.  **Projeyi Açın:** "Open" seçeneği ile proje dizinini (`C:\Projects\MesopotamiaSDR`) seçin.
3.  **Python Interpreter'ı Ayarlayın:**
    *   `File` > `Settings` (veya `PyCharm` > `Preferences` macOS'ta).
    *   `Project: MesopotamiaSDR` > `Python Interpreter` bölümüne gidin.
    *   Sağ üstteki dişli ikonuna tıklayın ve `Add...` seçeneğini seçin.
    *   Sol menüden `Conda Environment` seçeneğini seçin.
    *   `Existing environment` seçeneğini işaretleyin.
    *   `Interpreter` alanında, Anaconda kurulumunuz altındaki `envs\mesopotamia_sdr\python.exe` dosyasını bulun ve seçin (örneğin, `C:\Users\KullaniciAdi\anaconda3\envs\mesopotamia_sdr\python.exe`).
    *   `OK` butonuna tıklayarak onaylayın.
4.  **Çalıştırma Yapılandırması (Opsiyonel):**
    *   Ana betiği (örneğin, `main.py` veya `run_system.py`) çalıştırmak için bir yapılandırma oluşturabilirsiniz.
    *   `Run` > `Edit Configurations...` seçeneğine gidin.
    *   `+` ikonuna tıklayıp `Python` seçin.
    *   `Script path:` alanına ana betiğin yolunu (`C:\Projects\MesopotamiaSDR\Toprakperest\main.py`) girin.
    *   `Working directory:` alanının proje ana dizini (`C:\Projects\MesopotamiaSDR\Toprakperest`) olduğundan emin olun.
    *   `OK` ile kaydedin.

## 6. Kurulumu Test Etme

1.  **PyCharm Terminalini Açın:** PyCharm içindeki terminali açın. Anaconda ortamının (`mesopotamia_sdr`) otomatik olarak aktifleşmesi gerekir.
2.  **Basit Bir İçe Aktarma Testi:** Python konsolunu açın (`python` yazıp Enter'a basın) ve bazı kütüphaneleri içe aktarmayı deneyin:
    ```python
    import numpy
    import zmq
    import SoapySDR # Eğer SoapySDR kurulumu başarılıysa
    print("Kurulum başarılı!")
    exit()
    ```
3.  **Test Betiğini Çalıştırın:** Depodaki `test_sdr.py` veya `test_system.py` gibi test betiklerini çalıştırmayı deneyin (varsa ve güncelse).

Artık geliştirme ortamınız hazır. PyCharm üzerinden kodları düzenleyebilir, çalıştırabilir ve hata ayıklama yapabilirsiniz.
