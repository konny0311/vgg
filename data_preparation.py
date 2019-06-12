import cv2
import numpy as np
import os
import glob


SIZE = 224

class Data():

    def __init__(self, image_dirs, filename_flag=False):
        """
        image_dirs: list of class image directories
        """
        self.filename_flag = filename_flag
        self.image_dirs = image_dirs
        self._prepare_data
        self.n_classes = len(image_dirs)
    
    def _prepare_data(self):

        if len(self.image_dirs) > 0:
            self.images = np.array([])
            self.answers = np.array([])
            if self.filename_flag:
                self.filenames = []

            for i, image_dir in enumerate(self.image_dirs):
                files = glob.glob(os.path.join(image_dir, '*.jpg'))
                files += glob.glob(os.path.join(image_dir, '*.png'))
                self.images = np.concatenate([self.images, np.array([self._resize_for_model(cv2.imread(file)) for file in files])])
                self.answers = np.concatenate([self.answers, np.array([i] * len(files))])
                if self.filename_flag:
                    self.filenames += [os.path.basename(file) for file in files]

    def _resize_for_model(self, image):
        # np形式のimageを特定の大きさにresizeする。
        return cv2.resize(image, (SIZE, SIZE))
    
    def prepare_for_train(self):
        
        self.images = self.images.astype('float32')
        self.images /= 255
        self.answers = np_utils.to_categorical(self.answers, self.n_classes)


