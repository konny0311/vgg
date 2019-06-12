from vgg19 import VGG19
from data_preparation import Data
import cv2
import numpy as np
import os
import glob
import csv
import keras
import keras.callbacks as KC
from keras import regularizers
from keras.preprocessing.image  import ImageDataGenerator

BATCH = 4
EPOCH = 10
VERBOSE = 1
SAVED_MODEL_PATH = os.path.join('models', 'cat_dog_vgg_best_weights.hdf5')
END_MODEL_PATH = os.path.join('models', 'cat_dog_vgg_last_weights.hdf5')
base_dir = 'images'
dog = 'dog'
cat = 'cat'
train = 'train_images'
val = 'valid_images'
test = 'test_images'
dog_train_dir = os.path.join(base_dir, train,dog)
cat_train_dir = os.path.join(base_dir, train,cat)
dog_val_dir = os.path.join(base_dir, val, dog)
cat_val_dir = os.path.join(base_dir, val, cat)
dog_test_dir = os.path.join(base_dir, test, dog)
cat_test_dir = os.path.join(base_dir, test, cat)

train_data = Data([dog_train_dir, cat_train_dir])
train_data.prepare_for_train()
val_data = Data([dog_val_dir, cat_val_dir])
val_data.prepare_for_train()
test_data = Data([dog_test_dir, cat_test_dir], filename_flag=True)
test_data.prepare_for_train()

vgg = VGG19(n_classes=train_data.n_classes)
vgg.model.summary()

callbacks = [KC.TensorBoard(),
            KC.ModelCheckpoint(filepath=SAVED_MODEL_PATH,
            verbose=1,
            save_weights_only=True,
            save_best_only=True,
            period=2)]

history = vgg.model.fit(train_data.images, train_data.answers,
                    batch_size=BATCH, 
                    epochs=EPOCH, 
                    verbose=VERBOSE,
                    validation_data=(val_data.images, val_data.answers), 
                    shuffle=True,
                    callbacks=callbacks)

vgg.model.save_weights(END_MODEL_PATH)

score = vgg.model.evaluate(test_data.images, test_data.answers, verbose=VERBOSE)
print('Test score:', score[0])
print('Test acc:', score[1])

wrong_files = vgg.model.predict(test_data, ret=True)
_ = [print(file) for file in wrong_files]