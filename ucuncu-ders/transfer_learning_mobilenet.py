import keras
from keras.preprocessing.image import ImageDataGenerator
import argparse
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

x = keras.layers.Dense(512, activation='relu')(base_model.output)
x = keras.layers.Dropout(0.25)(x)
output_tensor = keras.layers.Dense(args["classes"], activation='softmax')(x)

model = keras.models.Model(inputs=input_tensor, outputs=output_tensor)

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    args["training_dir"],
    target_size=(224, 224),
    batch_size=args["batch_size"],
    class_mode='categorical',
    shuffle=True)

validation_generator = datagen.flow_from_directory(
    args["validation_dir"],
    target_size=(224, 224),
    batch_size=args["batch_size"],
    class_mode='categorical',
    shuffle=False)

model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])

model.fit_generator(train_generator,
                                        steps_per_epoch = len(train_generator) // args["batch_size"],
                    epochs=args["epochs"],
                    validation_data=validation_generator,
                                        validation_steps = len(validation_generator) // args["batch_size"])
