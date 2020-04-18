import numpy as np
import keras
import os
import json
import random
import PIL

from io import BytesIO

import cv2

from tqdm import tqdm

from PIL import Image
from jpeg2dct.numpy import load, loads

from keras.applications.vgg16 import preprocess_input

from keras.utils import Sequence


def prepare_imagenet(index_file, data_directory):

    association = {}
    with open(index_file) as index:
        data = json.load(index)
        for id, value in data.items():
            association[value[0]] = id

    # We process the data directory to get all the classes and images
    classes = []
    images_path = []

    for directory in tqdm(os.listdir(data_directory)):
        class_directory = os.path.join(data_directory, directory)
        if os.path.isdir(class_directory):
            classes.append(directory)
            for image in os.listdir(class_directory):
                image_path = os.path.join(class_directory, image)
                images_path.append(image_path)

    return association, classes, images_path


class DCTGeneratorJPEG2DCT(Sequence):
    'Generates data in the DCT space for Keras. This generator makes usage of the [following](https://github.com/uber-research/jpeg2dct) repository to read the jpeg images in the correct format.'

    def __init__(self,
                 data_directory,
                 index_file,
                 input_size=(28, 28),
                 batch_size=32,
                 shuffle=True,
                 seed=333,
                 validation_split=0.0,
                 split_cbcr=False,
                 only_y=False,
                 validation=False,
                 transforms=None):

        if input_size is None and batch_size is not 1:
            raise RuntimeError(
                "The when input_size is None, the batch size should be one.")
        # Process the index dictionary to get the matching name/class_id
        self.association, self.classes, self.images_path = prepare_imagenet(
            index_file, data_directory)

        self.images_path = self.images_path

        # External data
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._number_of_data_samples = len(self.images_path)

        # Internal data
        self.input_size = input_size
        self.split_cbcr = split_cbcr
        self.only_y = only_y

        # If no validation split, all in test
        if validation_split == 0 or validation_split == 1:
            self.indexes = np.arange(len(self.images_path))
        else:
            np.random.seed(seed)
            full_indexes = np.arange(len(self.images_path))
            np.random.shuffle(full_indexes)
            split_index = int(validation_split * len(self.images_path))
            if validation:
                self.indexes = full_indexes[split_index:]
            else:
                self.indexes = full_indexes[:split_index]

            # Re-set the seed to random
            np.random.seed(None)

        self.transforms = transforms
        self.number_of_classes = len(self.classes)
        # An epoch sees all the images
        self.batches_per_epoch = len(self.indexes) // self._batch_size

        # Initialization of the first batch
        self.on_epoch_end()

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
        self.batches_per_epoch = len(self.indexes) // self._batch_size

    @property
    def number_of_data_samples(self):
        return self._number_of_data_samples

    @number_of_data_samples.setter
    def number_of_data_samples(self, value):
        self._number_of_data_samples = value

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, value):
        self._shuffle = value
        if self.shuffle == False:
            self.indexes = np.arange(len(self.indexes))
        else:
            self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # We have to use modulo to avoid overflowing the index size if we have too many batches per epoch
        index = index % self.batches_per_epoch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self._batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self._shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'

        # Two inputs for the data of one image.
        if self.input_size is not None:
            X_y = np.empty(
                (self._batch_size, *self.input_size, 64), dtype=np.int32)
            if not self.split_cbcr:
                X_cbcr = np.empty(
                    (self._batch_size, self.input_size[0] // 2, self.input_size[1] // 2, 128), dtype=np.int32)
            else:
                X_cb = np.empty(
                    (self._batch_size, self.input_size[0] // 2, self.input_size[1] // 2, 64), dtype=np.int32)
                X_cr = np.empty(
                    (self._batch_size, self.input_size[0] // 2, self.input_size[1] // 2, 64), dtype=np.int32)
        y = np.zeros((self._batch_size, self.number_of_classes),
                     dtype=np.int32)

        # iterate over the indexes to get the correct values
        for i, k in enumerate(indexes):

            # Get the index of the class for later usage
            last_slash = self.images_path[k].rfind("/")
            second_last_slash = self.images_path[k][:last_slash].rfind("/")
            index_class = self.images_path[k][second_last_slash + 1:last_slash]

            # Load the image in RGB
            img = cv2.imread(self.images_path[k])

            if self.transforms:
                for transform in self.transforms:
                    img = transform(image=img)['image']

            _, buffer = cv2.imencode(".jpg", img)
            io_buf = BytesIO(buffer)

            dct_y, dct_cb, dct_cr = loads(io_buf.getvalue())

            if self.input_size is None:
                if not self.split_cbcr:
                    X_cbcr = np.empty(
                        (self._batch_size, dct_cb.shape[0], dct_cb.shape[1], dct_cb.shape[2] * 2), dtype=np.int32)
                    X_cbcr[i] = np.concatenate([dct_cb, dct_cr], axis=-1)
                else:
                    X_cb = np.empty(
                        (self._batch_size, dct_cb.shape[0], dct_cb.shape[1], dct_cb.shape[2]), dtype=np.int32)
                    X_cr = np.empty(
                        (self._batch_size, dct_cb.shape[0], dct_cb.shape[1], dct_cb.shape[2]), dtype=np.int32)
                    X_cb[i] = dct_cb
                    X_cr[i] = dct_cr

                X_y = np.zeros(
                    (self._batch_size, dct_cb.shape[0] * 2, dct_cb.shape[1] * 2, dct_cb.shape[2]), dtype=np.int32)
                X_y[i, :dct_y.shape[0], :dct_y.shape[1], :] = dct_y

            else:
                try:
                    X_y[i] = dct_y
                    if not self.split_cbcr:
                        X_cbcr[i] = np.concatenate([dct_cb, dct_cr], axis=-1)
                    else:
                        X_cb[i] = dct_cb
                        X_cr[i] = dct_cr
                except Exception as e:
                    raise Exception(str(e) + str(self.images_path[k]))

            # Setting the target class to 1
            y[i, int(self.association[index_class])] = 1
        if not self.split_cbcr:
            if self.only_y:
                return X_y, y
            else:
                return [X_y, X_cbcr], y
        else:
            return [X_y, X_cb, X_cr], y


class RGBGenerator(Sequence):
    """ Generator for RGB images for the Imagenet dataset. The generator needs a folder with all the classes as well as the index file to generate the data.

    # Arguments
        - data_directory: The folder containing all the classes' folders. One folder per class.
        - index_file: The file containing the index of all the classes
        - input_size: The size of the input, if None the batch_size should be one
        - batch_size: The size of the batches to be generated.
        - shuffle: If the batch should be shuffled. The validation batch is never shuffled.
        - seed: The seed to use for shuffling the data. Should be the same for the training and validation generators.
        - validation_split: The size of the split for the validation, in the range [0;1].
        - validation: If this generator should use the validation split.
        - transforms: The transformation to apply to the images.
    """

    def __init__(self,
                 data_directory,
                 index_file,
                 input_size=(224, 224),
                 batch_size=32,
                 shuffle=True,
                 seed=333,
                 validation_split=0.0,
                 validation=False,
                 transforms=None):

        if input_size is None and batch_size is not 1:
            raise RuntimeError(
                "The when input_size is None, the batch size should be one.")
        # Process the index dictionary to get the matching name/class_id
        self.association, self.classes, self.images_path = prepare_imagenet(
            index_file, data_directory)

        # self.classes = self.classes[:1000]
        # self.images_path = self.images_path[:1000]

        # External data
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._number_of_data_samples = len(self.images_path)

        # Internal data
        self.input_size = input_size

        # If no validation split, all in test
        if validation_split == 0 or validation_split == 1:
            self.indexes = np.arange(len(self.images_path))
        else:
            np.random.seed(seed)
            full_indexes = np.arange(len(self.images_path))
            np.random.shuffle(full_indexes)
            split_index = int(validation_split * len(self.images_path))
            if validation:
                self.indexes = full_indexes[split_index:]
            else:
                self.indexes = full_indexes[:split_index]

            # Re-set the seed to random
            np.random.seed(None)

        self.transforms = transforms
        self.number_of_classes = len(self.classes)
        # An epoch sees all the images
        self.batches_per_epoch = len(self.indexes) // self._batch_size

        # Initialization of the first batch
        self.on_epoch_end()

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def number_of_data_samples(self):
        return self._number_of_data_samples

    @number_of_data_samples.setter
    def number_of_data_samples(self, value):
        self._number_of_data_samples = value

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, value):
        self._shuffle = value
        if self._shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # We have to use modulo to avoid overflowing the index size if we have too many batches per epoch
        index = index % self.batches_per_epoch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self._batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self._shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'

        # Two inputs for the data of one image.
        if self.input_size is not None:
            X = np.empty((self._batch_size, 224, 224, 3), dtype=np.int32)
        y = np.zeros((self._batch_size, self.number_of_classes),
                     dtype=np.int32)

        # iterate over the indexes to get the correct values
        for i, k in enumerate(indexes):

            # Get the index of the class for later usage
            last_slash = self.images_path[k].rfind("/")
            second_last_slash = self.images_path[k][:last_slash].rfind("/")
            index_class = self.images_path[k][second_last_slash + 1:last_slash]

            # Load the image in RGB

            img = Image.open(self.images_path[k])
            img = img.convert("RGB")
            img = np.asarray(img)
            if self.transforms:
                for transform in self.transforms:
                    img = transform(image=img)['image']

            # If no input size is provided, we keep the size of the image (we have a batch size of one then).
            if self.input_size is None:
                X = np.empty((self._batch_size, *img.shape), dtype=np.int32)

            X[i] = preprocess_input(img)

            # Setting the target class to 1
            y[i, int(self.association[index_class])] = 1

        return np.array(X), np.array(y)

    def get_raw_input_label(self, index):
        """ Provide with the raw data, i.e displayable. Here we return the RGB image, same as the original __getitem__ function, without the preprocess input.

        # Argument:
            - index: The index of the batch of data to retreive from the generator.
        
        # Return:
            Two values, the images and the associated labels.

        """
        index = index % self.batches_per_epoch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self._batch_size]

        # Generate data
        if self.input_size is not None:
            X = np.empty((self._batch_size, 224, 224, 3), dtype=np.uint8)
        y = np.zeros((self._batch_size, self.number_of_classes),
                     dtype=np.int32)

        # iterate over the indexes to get the correct values
        for i, k in enumerate(indexes):

            # Get the index of the class for later usage
            last_slash = self.images_path[k].rfind("/")
            second_last_slash = self.images_path[k][:last_slash].rfind("/")
            index_class = self.images_path[k][second_last_slash + 1:last_slash]

            # Load the image in RGB

            img = Image.open(self.images_path[k])
            img = img.convert("RGB")
            img = np.asarray(img)
            if self.transforms:
                for transform in self.transforms:
                    img = transform(image=img)['image']

            # If no input size is provided, we keep the size of the image (we have a batch size of one then).
            if self.input_size is None:
                X = np.empty((self._batch_size, *img.shape), dtype=np.uint8)

            X[i] = img
            # Setting the target class to 1
            y[i, int(self.association[index_class])] = 1

        return np.array(X), np.array(y)