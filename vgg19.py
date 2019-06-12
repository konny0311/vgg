import numpy as np
import glob
import os
import cv2
from keras.engine.topology import Input
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers import  Conv1D, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.optimizers import Adam

class VGG19():

    def __init__(self, n_classes=2, lr=0.0001):
        """
        create a VGG19 model.
        """
        self.n_classes = n_classes
        self.lr = lr
        self.model = self._create_model()

    def _add_batch_norm_activation(self, input_layer, activation_type='relu'):
        """
        This is a wrapper method for a layers.
        BatchNormalization -> Activation
        """
        layer_b = BatchNormalization()(input_layer)
        layer_a = Activation(activation=activation_type)(layer_b)

        return layer_a

    def _create_four_conv_layers(self, input_layer, filter=256, padding_flag='same'):
        """
        This is a wrapper method for a layers.
        Conv2D -> Conv2D -> Conv2D -> Conv2D -> MaxPooling2D
        Each Conv2D includes batch normalization and activation
        """
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
        flatten = Flatten()(layer_12)
        output = Dense(self.n_classes, activation='sigmoid')(flatten)
        model = Model(inputs=[input_layer], outputs=[output])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr), metrics=['accuracy'])
        
        return model

    def predict(self, data, ret=False):
        """
        data: Data class
        ret: put True if you need a list of wrong answer files
        """
        answers = np.argmax(self.model.predict(data.images), axis=1)
        matrix = np.zeros((self.n_classes, self.n_classes), dtype=int)
        wrong_files = []
        for i, pre in enumerate(answers):
            correct_answer = data.answers[i]
            matrix[pre][correct_answer] += 1
            if pre != correct_answer:
                wrong_files.append(data.filenames[i])

        for i in range(self.n_classes):
            tp = matrix[i][i]
            precision = tp / np.sum(matrix[i, :])
            print('precision class_id{} is '.format(i+1), round(precision, 4))
            recall = tp / np.sum(matrix[:, i])
            print('recall class_id{} is '.format(i+1), round(recall, 4))

        if ret:
            return wrong_files
                

if __name__ == '__main__':
    vgg = VGG19(n_classes=4)
    vgg.model.summary()
