import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split


# === config ===
IMG_SIZE = 48
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 60
MODEL_OUT = 'fer_model_final.h5'
CSV_PATH = 'fer2013.csv'
print('Loading CSV...')
df = pd.read_csv("fer2013.csv")


def pixels_to_array(pixels_str):
    arr = np.fromstring(pixels_str, dtype=np.uint8, sep=' ')
    return arr.reshape(IMG_SIZE, IMG_SIZE)


print('Parsing images...')
X = np.stack(df['pixels'].apply(pixels_to_array).values)
X = X.astype('float32') / 255.0
X = np.expand_dims(X, -1) # shape: (n, 48, 48, 1)


y_labels = df['emotion'].values # integer labels 0..6
Y = tf.keras.utils.to_categorical(y_labels, NUM_CLASSES)


# split train/val
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.15, stratify=y_labels, random_state=42)


# data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
rotation_range=10,
width_shift_range=0.1,
height_shift_range=0.1,
zoom_range=0.1,
horizontal_flip=True)


train_gen = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)


# build model
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)


    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)


    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)


    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model


model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


cb = [
callbacks.ModelCheckpoint('best_fer.h5', save_best_only=True, monitor='val_loss'),
callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]


steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)
print('Starting training...')
model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, validation_data=(X_val, y_val), callbacks=cb)


print('Saving final model...')
model.save(MODEL_OUT)
print('Done!')