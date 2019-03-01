import argparse
from effnet import EffNet
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training_dir", required=True, help="Path to directory holding training images")
ap.add_argument("-v", "--validation_dir", required=True, help="Path to directory holding validation images")
ap.add_argument("-c", "--classes", type=int, required=True, help="Number of classes in the dataset")
ap.add_argument("-b", "--batch_size", type=int, default=32, help="Number of samples in a batch")
ap.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs to train model")
args = vars(ap.parse_args())

input_size = 208

model = EffNet((input_size, input_size, 3), args["classes"])
model.compile(optimizer=Adam(lr=0.0001, beta_1=0.75), 
loss='categorical_crossentropy',
metrics=['acc'])

train_aug = ImageDataGenerator(rescale=1.0 / 255.0,
rotation_range=90,
width_shift_range=0.2,
height_shift_range=0.2,
zoom_range=0.2,
shear_range=0.2,
horizontal_flip=True,
fill_mode='nearest'
)

train_generator = train_aug.flow_from_directory(args["training_dir"],
target_size=(input_size, input_size),
batch_size=args["batch_size"],
class_mode='categorical',
shuffle=True)

validation_aug = ImageDataGenerator(rescale=1.0 / 255.0)

validation_generator = validation_aug.flow_from_directory(args["validation_dir"],
target_size=(input_size, input_size),
batch_size=args["batch_size"],
class_mode='categorical',
shuffle=False)


model.fit_generator(train_generator,
steps_per_epoch=train_generator.samples // args["batch_size"],
epochs=args["epochs"],
validation_data=validation_generator,
validation_steps=validation_generator.samples // args["batch_size"]
)

model.save('flowers.model')