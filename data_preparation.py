import cv2
import numpy as np
import os
import glob
from keras.utils import np_utils

SIZE = 224

class Data():

    def __init__(self, image_dirs, filename_flag=False):
        """
        image_dirs: list of class image directories
        filename_flag: put True if you need a filename object in an Data instance
        """
        self.filename_flag = filename_flag
        self.image_dirs = image_dirs
        self._prepare_data()
        self.n_classes = len(image_dirs)
    
    def _prepare_data(self):

        if len(self.image_dirs) > 0:
            if self.filename_flag:
                self.filenames = []
            tmp_images = []
            tmp_answers = []
            for i, image_dir in enumerate(self.image_dirs):
                files = glob.glob(os.path.join(image_dir, '*.jpg'))
                files += glob.glob(os.path.join(image_dir, '*.png'))
                tmp_images.extend([self._resize_for_model(cv2.imread(file)) for file in files])
                tmp_answers.extend([i] * len(files))
                if self.filename_flag:
                    self.filenames.extend([os.path.basename(file) for file in files])

            self.images = np.array(tmp_images)
            self.answers = np.array(tmp_answers)

    def _resize_for_model(self, image):
        return cv2.resize(image, (SIZE, SIZE))
    
    def prepare_for_train(self):
        """
        preparation to put data into a model
        """
        self.images = self.images.astype('float32')
        self.images /= 255
        self.answers = np_utils.to_categorical(self.answers, self.n_classes)



