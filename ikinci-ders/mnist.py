import keras
import numpy as np
import cv2

(X, y), (X_test, y_test) = keras.datasets.mnist.load_data()

print('{} adet örnek bulunuyor ve boyutları = {}'.format(X.shape[0], X.shape[1:]))

for i in range(10):
    cv2.imshow('Sample mnist digit {}'.format(i + 1), X[i])
    cv2.waitKey(0)