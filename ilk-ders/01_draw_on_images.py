# OpenCV'yi import ediyoruz
import cv2
# Numpy'ı import ediyoruz
import numpy as np

# Numpy'ı kullanarak 300*300 boyutlarında bir
# siyah görsel  oluşturuyoruz.
# Numpy array'in boyutlarındaki 3, renk kanallarının sayısını ifade ediyor.
# OpenCV, renkleri BGR (blue, green, red) formatında işliyor.
black_image = np.zeros((300,300,3), dtype='uint8')

# tuple açma yöntemiyle görselin boyutlarını alıyoruz ve konsola yazdırıyoruz.
# Boyutların sırayla yükseklik (rows), genişlik (columns) ve kanal sayısını
# belirttiğine dikkat edelim
height, width, channels = black_image.shape
print('Bu görselin  boyutları: genişlik={}, yükseklik={}. Görsel {} kanallı renklerden oluşuyor.'.format(width, height, channels))

# Verilen görselleri verilen başlık ile gösteren bir fonksiyon tanımlıyoruz.
# .waitKey() metodu programın çalışmasını duraklatıyor ve görseli  görmemize imkan tanıyor.
def show(img, title='Frame'):
    cv2.imshow(title, img)
    cv2.waitKey(0)

show(black_image, 'Black image')

# Kırmızı rengi tanımlayalım
red = (0,0,255)
# Görselin sol üst köşesinden sağ alt köesinde kırmızı bir çizgi çizelim
# Çizginin başlangıç ve bitiş noktalarını (x, y) koordinatı şeklinde veriyoruz
black_image = cv2.line(black_image, (0,0), (300, 300), red)
show(black_image, 'Red line')

# Yeşil rengi tanımlayalım
green = (0, 255, 0)
# Sağ üst köşeden sol alt köşeye kalınlığı 5 olan yeil bir çizgi çizelim
black_image = cv2.line(black_image, (300, 0), (0, 300), green, 5)
show(black_image, 'Green line')

# Mavi renk tanımlayalım
blue = (255, 0, 0)
# Sol üst köşesi (10, 10) ve sağ alt köşesi (50, 50) olan mavi renkte bir dörtgen  çizelim.
# Kalınlık olarak -1 verdiğimiz için içi dolu olacak.
black_image = cv2.rectangle(black_image, (10, 10), (50,50), blue, -1)
show(black_image, 'Blue rectangle')

# Beyaz rengi tanımlayalım
white = (255, 255, 255)
# Merkezi görselin  ortası olan beyaz renkli bir çember çizelim.
# Görselin orta noktasını bulmak için width ve height değerlerini tam sayı sonucu veren // işareti ile 2'ye bölüyoruz.
black_image = cv2.circle(black_image, (width // 2, height // 2), 25, white, 3)
show(black_image, 'White circle')