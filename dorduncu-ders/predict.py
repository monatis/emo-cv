import argparse
import cv2
import keras
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input input to recognize")
args = vars(ap.parse_args())

# Eğittiğimiz modelimizi yükleyelim
model = keras.models.load_model('flowers.model')

# Kaydettiğimiz labels.txt dosyasını okuyup bir listeye atayalım
labels = []
with open('labels.csv', 'r') as f:
    for line in f.readlines():
        label = line.strip().split(',')
        labels.append(label[1])

# Tanımak istediğimiz görseli yükleyelim
img = cv2.imread(args["image"])

# Modelimizi RGB formatındaki görseller ile eğittiğimiz ve OpenCV görselleri BGR formatında yüklediği için BGR düzeninden RGB düzenine çeviriyoruz
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Yapay zeka modelimizi 224x224 boyutunda görseller ile eğittiğimiz için tanımak istediğimiz görseli de 224x224 olarak yeniden boyutlandırıyoruz
img = cv2.resize(img, (224, 224))

# Yapay zeka modelimizi eğitirken tüm piksel değerlerini 0-1 aralığına normalize ettiğimiz için aynı işlemi tekrarlıyoruz
img = img.astype('float') / 255.0

# Yapay zeka modelimiz 4 boyutlu tensor'ler ile eğitildiğinden tanımak istediğimiz görseli de 4 boyutlu yapmamız lazım
img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

# Artık yapay zeka modelimizin tahminini alabiliriz
pred = model.predict(img)

# Dönen matrisin -1 indeksindeki sütunun maksimum değerinin indeksi, yapay zekanın tahmin ettiği etiketin indeksine eşittir
i = np.argmax(pred, axis=-1)[0]
prob = pred[0, i] * 100
label = labels[i]
print("Yüzde {:.2f} olasılıkla bu görselde {} olduğunu tahmin ediyorum.".format(prob, label))