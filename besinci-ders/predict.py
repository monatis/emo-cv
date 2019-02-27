import argparse
import cv2
import keras
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image to recognize")
args = vars(ap.parse_args())

img = cv2.imread(args["image"])
orig = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img.astype('float') / 255.0
img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

model = keras.models.load_model('flowers.model')

labels = []
with open('labels.csv', 'r') as f:
    for line in f.readlines():
        label = line.strip().split(',')
        labels.append(label[1])

pred = model.predict(img)
i = np.argmax(pred, axis=-1)[0]
prob = pred[0, i] * 100
label = labels[i]

print("Yüzde {:.2f} olasılıkla bu görselde {} olduğunu düşünüyorum".format(prob, label))
cv2.imshow('Flower', orig)
cv2.waitKey(0)