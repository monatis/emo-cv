# Gerekli dosyaları import edelim
import cv2
import numpy as np

# Verilen görseli verilen başlıkla gösterecek fonksyonumuzu tanımlıyoruz
def show(img, title='Frame'):
    cv2.imshow(title, img)
    cv2.waitKey(0)


# 500x500 boyutlarında 3 kanallı bir görsel oluşturuyoruz.
# Data type olarak 'uint8' seçiyoruz, çünkü renk değerleri 8 bitlik unsigned int veri tipinde saklanıyor.
img = np.ones((500, 500, 3), dtype='uint8')
# Beyaz bir görsel elde etmek için tüm öğeleri 1 olarak başlatılan numpy array'i 255 le çarpıyoruz
img = img * 255
show(img, 'White image')

print('Çemberleri çiziyorum...')

# range metodu ile her dönüşünde yarıçap değerimizi 15'er 15'er arttıracak bir döngü oluşturuyoruz
for radius in range(0, 240, 15):
    # Döngünün her dönüşünde np.random.randint() metodunu kullanarak rastgele bir renk değeri oluşturuyoruz
    # ve .tolist() metodu ile Numpy array'imizi python listesine dönütürüyoruz
    # çünkü OpenCV fonksiyonları renk değerlerini bu veri tipinde bekliyor.
    color = np.random.randint(0, 255, 3).tolist()

    # Yine numpy kullanarak 1:4 aralığında rastgele bir kalınlık değeri oluşturalım
    # Oluşturulan tek boyutlu ve büyüklüğü 1 olan numpy array'in 0'ıncı indeksini thickness değikenine atıyoruz:
    thickness = np.random.randint(1, 4, 1)[0]

    # Görselin orta noktalarını bulup center_x ve center_y değikenlerine atayalım
    height, width, num_channels = img.shape
    center_x = width // 2
    center_y = height // 2

    # Artık çemberlerimizi çizebiliriz.
    # Her bir çemberimizin merkez noktası görselimizin ortası olacak ve
    # döngünün her dönüşünde radius değeri artacağından çemberlerimiz genişleyecek.
    img = cv2.circle(img, (center_x, center_y), radius, color, thickness)


# Bu noktada döngüden çıktık ve tüm çemberlerimiz beyaz görselimizin üzerine çizildi.
# Artık görselimizi ekranda gösterip ayrıca diskimize kaydedebiliriz.
show(img, 'Random circles')

# OpenCV'nin .imwrite fonksiyonu ile görsellerimizi diske kaydedebiliriz.
# Fonksiyona verdiğimiz dosya yolundaki dosya uzantısına göre OpenCV kodlamayı otomatik olarak yapmaktadır.
cv2.imwrite('./random_circles.jpg', img)

print('Görseli diske de kaydettim, ./random_circles.png dosyasına bakabilirsiniz')