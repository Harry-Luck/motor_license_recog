import os
import numpy as np
import random
import cv2
from keras import layers, Input, Model, callbacks

input_dir = "dataset/"
target_dir = "annotations/"

input_img_paths = sorted([os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith(".jpg")])
target_paths = sorted([os.path.join(target_dir, fname) for fname in os.listdir(target_dir) if fname.endswith(".jpg")])


img_size = (224, 224)
num_imgs = len(input_img_paths)
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_paths)

input_imgs = np.zeros((num_imgs,) + img_size + (3,),dtype="float32")
targets = np.zeros((num_imgs,) + img_size + (1,), dtype="uint8")

for i in range(num_imgs):
    input_imgs[i] = cv2.resize(cv2.imread(input_img_paths[i]), img_size)
    targets[i] =  cv2.resize(cv2.imread(target_paths[i], 0), img_size).reshape(224,224,1)
input_imgs = input_imgs.astype('float32') / 255
targets = targets / 255

def get_model(img_size, num_classes):
    inputs = Input(shape=img_size + (3,))
    x = layers.Conv2D(64, 3, strides=2, activation="relu",padding="same")(inputs)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu",padding="same")(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, activation="relu",padding="same")(x)
    x = layers.Conv2DTranspose(256, 3, activation="relu",padding="same")(x)
    x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu",padding="same")(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu",padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same", strides=2)(x)
    outputs = layers.Conv2D(num_classes, 3, activation="softmax",padding="same")(x)
    model = Model(inputs, outputs)
    return model
model = get_model(img_size=img_size, num_classes=2)
# model.summary()
model.compile(optimizer="rmsprop",loss="sparse_categorical_crossentropy")
callbacks = [
callbacks.ModelCheckpoint("oxford_segmentation.keras",
save_best_only=True)
]
model.fit(input_imgs, targets, epochs=50, batch_size=64, callbacks=callbacks, validation_split=0.2)

