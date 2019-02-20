import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
import argparse
import csv
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training_dir", required=True, help="Path to training directory")
ap.add_argument("-v", "--validation_dir", required=True, help="Path to validation directory")
ap.add_argument("-c", "--classes", type=int, required=True, help="Number of classes in the dataset")
ap.add_argument("-e", "--epochs", type=int, default=20, help="Numbers of epochs to be trained")
ap.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size as an integer")
args = vars(ap.parse_args())

input_shape = (224, 224, 3)
input_tensor = keras.layers.Input(shape=input_shape)

base_model = keras.applications.MobileNetV2(weights='imagenet',
                  include_top=False,
                  input_tensor=input_tensor,
                                    input_shape=input_shape,
                                                                        pooling='avg')

for layer in base_model.layers:
    layer.trainable = False

x = keras.layers.Dense(1024, activation='relu')(base_model.output)
x = keras.layers.Dropout(0.25)(x)
output_tensor = keras.layers.Dense(args["classes"], activation='softmax')(x)

model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)

train_data_gen = ImageDataGenerator(rescale=1./255,
        rotation_range=90,
                zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

train_generator = train_data_gen.flow_from_directory(
    args["training_dir"],
    target_size=(224, 224),
    batch_size=args["batch_size"],
    class_mode='categorical',
    shuffle=True
)

validation_data_gen = ImageDataGenerator(rescale=1.0 / 255.0)

validation_generator = validation_data_gen.flow_from_directory(
    args["validation_dir"],
    target_size=(224, 224),
    batch_size=args["batch_size"],
    class_mode='categorical',
    shuffle=False)

labels = dict((v, k) for k, v in train_generator.class_indices.items())

with open('labels.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(labels.items())

model.compile(optimizer=keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['acc'])

h = model.fit_generator(
                                                                                train_generator,
                                        steps_per_epoch = train_generator.samples // args["batch_size"],
                    epochs=args["epochs"],
                    validation_data=validation_generator,
                                        validation_steps = validation_generator.samples // args["batch_size"]
)

model.save('flowers.model')

n = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, n), h.history["loss"], label="train_loss")
plt.plot(np.arange(0, n), h.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, n), h.history["acc"], label="train_acc")
plt.plot(np.arange(0, n), h.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy with Fine Tuning")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")