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


## Unity Entegrasyonu

Python uygulaması, tanınan el hareketlerini UDP üzerinden Unity'ye gönderir.
İkisi aynı bilgisayarda çalışır; ekstra bir server kurulumu gerekmez.

### Python tarafı

Ek bir paket gerekmez. Uygulamayı şu parametreyle başlatın:

   python app.py --control-unity

Varsayılan hedef: 127.0.0.1:7777
Port değiştirmek için: python app.py --control-unity --unity-port 9000

Google Earth ve Unity'yi aynı anda kullanmak için:

   python app.py --control-unity --control-earth --earth-focus

### Unity tarafı

1. `unity/GestureReceiver.cs` dosyasını Unity projenizin `Assets/Scripts/` klasörüne kopyalayın.

2. Sahnede bir GameObject'e (örneğin GameManager) GestureReceiver component'ini ekleyin.

3. Inspector'dan Port değerini Python'daki --unity-port ile eşleştirin (varsayılan: 7777).

4. Kamera/navigasyon kodunuzu On Command Started event'ine bağlayın:

   On Command Started → CameraController.HandleGestureCommand

5. Kamera kontrol scriptinizde gelen komutu işleyin:

   public void HandleGestureCommand(string command)
   {
       switch (command)
       {
           case GestureCommand.PanUp:    /* kamerayı ileri taşı */ break;
           case GestureCommand.PanDown:  /* kamerayı geri taşı  */ break;
           case GestureCommand.PanLeft:  /* kamerayı sola taşı  */ break;
           case GestureCommand.PanRight: /* kamerayı sağa taşı  */ break;
           case GestureCommand.ZoomIn:   /* yakınlaştır         */ break;
           case GestureCommand.ZoomOut:  /* uzaklaştır          */ break;
           case GestureCommand.ResetView:/* kamerayı sıfırla    */ break;
       }
   }

### Gelen JSON formatı (her frame)

   {
     "command":        "PAN_UP",
     "confidence":     0.85,
     "hand_sign":      "Open",
     "finger_gesture": "Move",
     "ux_state":       "NAVIGATING"
   }

### Komut listesi

   PAN_UP / PAN_DOWN / PAN_LEFT / PAN_RIGHT   → yön hareketi
   ZOOM_IN / ZOOM_OUT                          → yakınlaştır / uzaklaştır
   ROTATE_LEFT / ROTATE_RIGHT                 → döndür
   RESET_VIEW                                 → kamerayı sıfırla
   STOP                                       → hareketi durdur
   NONE                                       → el görünmüyor / bekleniyor
