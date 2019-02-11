import numpy as np
import cv2
black_image = np.zeros((300,300,3), dtype='uint8')

red = (0, 0, 255)
black_image = cv2.line(black_image, (0, 0), (300, 300), red, 3)

cv2.imshow('Red line', black_image)
cv2.waitKey(0)

green = (0, 255, 0)
black_image = cv2.rectangle(black_image, (10, 10), (50, 50), green, -1)
cv2.imshow('Green square', black_image)
cv2.waitKey(0)

height, width, num_channels = black_image.shape
blue = (255, 0, 0)
black_image = cv2.circle(black_image, (width // 2, height // 2), 25, blue, 5)
cv2.imshow('Blue circle', black_image)
cv2.waitKey(0)