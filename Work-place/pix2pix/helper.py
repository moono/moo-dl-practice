import numpy as np
import os
import glob
from PIL import Image
from scipy.misc import imresize

# class for loading images & split image of the form [A|B] ==> (A, B)
class Dataset(object):
    def __init__(self, input_dir, convert_to_lab_color=False, direction='AtoB', scale_to=256):
        if not os.path.exists(input_dir):
            raise Exception('input directory does not exists!!')

        # search for images(*.jpg or *.png)
        self.image_files = glob.glob(os.path.join(input_dir, '*.jpg'))
        if len(self.image_files) == 0:
            self.image_files = glob.glob(os.path.join(input_dir, '*.png'))

        if len(self.image_files) == 0:
            raise Exception('input directory does not contain any images!!')

        def get_name(path):
            name, _ = os.path.splitext(os.path.basename(path))
            return name

        # if the image names are numbers, sort by the value rather than asciibetically
        # having sorted inputs means that the outputs are sorted in test mode
        if all(get_name(path).isdigit() for path in self.image_files):
            self.image_files = sorted(self.image_files, key=lambda path: int(get_name(path)))
        else:
            self.image_files = sorted(self.image_files)

        # set class attributes
        self.n_images = len(self.image_files)
        self.convert_to_lab_color = convert_to_lab_color
        self.direction = direction
        self.scale_to = scale_to
        self.batch_index = 0
        self.image_max_value = 255

    def get_next_batch(self, batch_size):
        if (self.batch_index + batch_size) > self.n_images:
            self.batch_index = 0

        current_files = self.image_files[self.batch_index:self.batch_index + batch_size]
        splitted = self.load_images(current_files)

        self.batch_index += batch_size

        return splitted

    def load_images(self, files):
        splitted = []
        for im in files:
            # open images with PIL
            im = Image.open(im)

            if self.convert_to_lab_color:
                raise Exception('Lab color conversion: Not implemented yet!!')
            else:
                # convert to np array
                im = np.array(im.convert('RGB')).astype(np.float32)
                width = im.shape[1]  # [height, width, channels]
                a_image = im[:, :width // 2, :]
                b_image = im[:, width // 2:, :]

                # normalize input [0 ~ 255] ==> [-1 ~ 1]
                a_image = (a_image / self.image_max_value - 0.5) * 2
                b_image = (b_image / self.image_max_value - 0.5) * 2

            if not (a_image.shape[1] == self.scale_to and b_image.shape[1] == self.scale_to):
                a_image = imresize(a_image, (self.scale_to, self.scale_to))
                b_image = imresize(b_image, (self.scale_to, self.scale_to))

            if self.direction == 'AtoB':
                inputs, targets = [a_image, b_image]
            elif self.direction == 'BtoA':
                inputs, targets = [b_image, a_image]
            else:
                raise Exception('Invalid direction')


            splitted.append((inputs, targets))
        return splitted

