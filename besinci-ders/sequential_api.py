from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import csv

num_classes = 5
train_dir = 'D:\\datasets\\flower_photos\\training'
validation_dir = 'D:\\datasets\\flower_photos\\validation'

# Sequential API listeye öğe ekler gibi katmanları üst üste sıralamamızı kolaylaştırır
model = Sequential()

# Conv2D, ilk parametresine verdiğimiz sayıda filtre öğrenmek için ikinci parametresine verdiğimiz boyutlardaki parçalara convolutonal (matris çarpımı) işlemi uygular ve skalar bir değer üretir.
# Katman çıktısı matrisi girdi matrisiyle aynı boyutta tutmak için padding argümanına 'same' veriyoruz.
# Keras ilk katmandan sonraki her bir katmanın girdi boyutunu otomatik olarak hesaplayacaktır, ancak ilk katmanın
# boyutunu input_shape argümanına bir tupple vererek kendimizin tanımlaması gerekiyor.
model.add(Conv2D(64, (7, 7), padding='same', input_shape=(224, 224, 3)))
# relu aktivasyon fonksiyonu 0'dan küçük değerleri 0, 0'a eşit veya daha büyük değerleri olduğu gibi döndürür.
model.add(Activation('relu'))

# MaxPooling işlemi her bir adımda pool_size ile belirtilen kısım kadar bir parça üzerinde maksimum işlemi uygular.
model.add(MaxPooling2D(pool_size=(2, 2)))

# Aldığı batch'ten mean değerini çıkarıp std (standard sapma) değerine bölerek modelin başındaki normalizasyon işlemini tekrarlar
model.add(BatchNormalization())

# Sonraki katmanlarda daha ayırt edici özellikleri öğrenmek için filtre sayısını arttırırken filtrenin uygulanacağı alanı küçültüyoruz
model.add(Conv2D(128, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

# fully connected katmanlara geçmeden önce Flatten katmanı ile çok katmanlı yapıdan kurtuluyoruz
model.add(Flatten())

# Fully connected (dense) katmanı kendinden önceki katmanın her bir değeriyle çarpım işlemi uygular
model.add(Dense(256))
model.add(Activation('relu'))

# Dropout belirtilen orana karşılık gelen sayıdaki nöronun bağlantılarını rastgele keserek modelin verisetini ezberlemesini (overfitting)
model.add(Dropout(0.3))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(optimizer=Adam(),
loss='categorical_crossentropy',
metrics=['acc']
)

train_aug = ImageDataGenerator(rescale=1.0 / 255.0,
rotation_range=90,
width_shift_range=0.2,
height_shift_range=0.2,
zoom_range=0.2,
shear_range=0.2,
horizontal_flip=True,
fill_mode='nearest'
)

validation_aug = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_aug.flow_from_directory(train_dir,
target_size=(224, 224),
batch_size=32,
class_mode='categorical',
shuffle=True
)

validation_generator = validation_aug.flow_from_directory(validation_dir,
target_size=(224, 224),
batch_size=32,
class_mode='categorical',
shuffle=False)


labels = dict((v, k) for k, v in train_generator.class_indices.items())

with open('labels.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(labels.items())


model.fit_generator(train_generator,
steps_per_epoch=train_generator.samples // 32,
epochs=50,
validation_data=validation_generator,
validation_steps=validation_generator.samples // 32
)

model.save('flowers.model')