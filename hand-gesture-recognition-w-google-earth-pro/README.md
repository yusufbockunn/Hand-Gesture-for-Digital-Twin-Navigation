1. Google Earth Pro indirin ve kurun (Windows için):
   - https://www.google.com/earth/versions/ adresine gidin.
   - "Google Earth Pro for desktop" veya "Download Earth Pro" seçeneğini bulun.
   - Windows sürümünü indirin ve normal şekilde yükleyin.
   - Yükleme tamamlandıktan sonra Google Earth Pro uygulamasını açın ve çalıştığından emin olun.

2. Bu proje klasörüne giderken PowerShell açın.
   cd "c:\Users\cemle\Desktop\ai-projects\Hand-Gesture-for-Digital-Twin-Navigation\hand-gesture-recognition-w-google-earth-pro"

3. Sanal ortam oluşturun ve etkinleştirin:
   python -m venv .venv
   .\.venv\Scripts\Activate

4. Gerekli Python paketlerini yükleyin:
   pip install opencv-python numpy mediapipe tensorflow

5. Google Earth kontrolü için ek paket yükleyin:
   pip install pynput

6. Uygulamayı çalıştırın:
   python app.py

7. Google Earth Pro ile el hareketleriyle kontrol etmek istiyorsanız uygulamayı şu parametrelerle başlatın:
   python app.py --control-earth --earth-focus

Örnek komutlar:
   python app.py
   python app.py --control-earth --earth-focus
   python app.py --control-earth --earth-focus --earth-layout
   python app.py --control-earth --earth-focus --earth-allow-close

Not:
- `--earth-focus` seçeneği, Google Earth penceresini öne getirir.
- Bu proje Windows ortamında Google Earth Pro ile çalışacak şekilde tasarlanmıştır.
