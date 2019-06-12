import numpy as np
import glob
import os
import cv2
import shutil
import sys
from tqdm import tqdm
from keras.engine.topology import Input
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers import  Conv1D, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import plot_model
import keras.callbacks as KC
from history_checkpoint_callback import HistoryCheckpoint, TargetHistory
from tools import draw_line, overlay

class VGG19():

    def __init__(self, n_classes=2):

        self.n_classes = n_classes
        self.model = self._create_model()

    def _add_batch_norm_activation(self, input_layer, activation_type='relu'):

        layer_b = BatchNormalization()(input_layer)
        layer_a = Activation(activation=activation_type)(layer_b)

        return layer_a

    def _create_four_conv_layers(self, input_layer, filter=256, padding_flag='same'):

        sub_layer_1 = Conv2D(filter, 3, padding=padding_flag)(input_layer)
        sub_layer_1 = self._add_batch_norm_activation(sub_layer_1)
        sub_layer_2 = Conv2D(filter, 3, padding=padding_flag)(sub_layer_1)
        sub_layer_2 = self._add_batch_norm_activation(sub_layer_2)
        sub_layer_3 = Conv2D(filter, 3, padding=padding_flag)(sub_layer_2)
        sub_layer_3 = self._add_batch_norm_activation(sub_layer_3)
        sub_layer_4 = Conv2D(filter, 3, padding=padding_flag)(sub_layer_3)
        sub_layer_4 = self._add_batch_norm_activation(sub_layer_4)
        output_layer = MaxPooling2D()(sub_layer_4)

        return output_layer

    def _create_model(self):
        """
        """
        padding_flag = 'same'
        first_filter = 64
        second_filter = 128
        first_dense_units = 4096
        second_dense_units = 1000
        shape = (224,224,3)

        input_layer = Input(shape=shape)
        layer_1 = Conv2D(first_filter, 3, padding=padding_flag)(input_layer)
        layer_1 = self._add_batch_norm_activation(layer_1)
        layer_2 = Conv2D(first_filter, 3, padding=padding_flag)(layer_1)
        layer_2 = self._add_batch_norm_activation(layer_2)
        layer_3 = MaxPooling2D()(layer_2)
        layer_4 = Conv2D(second_filter, 3, padding=padding_flag)(layer_3)
        layer_4 = self._add_batch_norm_activation(layer_4)
        layer_5 = Conv2D(second_filter, 3, padding=padding_flag)(layer_4)
        layer_5 = self._add_batch_norm_activation(layer_5)
        layer_6 = MaxPooling2D()(layer_5)
        layer_7 = self._create_four_conv_layers(layer_6)
        layer_8 = self._create_four_conv_layers(layer_7, filter=512)
        layer_9 = self._create_four_conv_layers(layer_8, filter=512)
        layer_10 = Dense(first_dense_units)(layer_9)
        layer_11 = Dense(first_dense_units)(layer_10)
        layer_12 = Dense(second_dense_units)(layer_11)
        output = Dense(self.n_classes, activation='sigmoid')(layer_12)
        model = Model(inputs=[input_layer], outputs=[output])
        
        return model

        
if __name__ == '__main__':
    vgg = VGG19(n_classes=4)
    vgg.model.summary()
