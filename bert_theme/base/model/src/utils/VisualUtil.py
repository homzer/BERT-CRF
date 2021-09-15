import os

import numpy
from PIL import Image


def clean_or_make_dir(output_dir):
    if os.path.exists(output_dir):
        def del_file(path):
            dir_list = os.listdir(path)
            for item in dir_list:
                item = os.path.join(path, item)
                del_file(item) if os.path.isdir(item) else os.remove(item)
        try:
            del_file(output_dir)
        except Exception as e:
            print(e)
            print('please remove the files of output dir.')
            exit(-1)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


def draw_array_img(array, output_dir):
    """
    draw images of array
    :param output_dir: directory of images.
    :param array: 3-D array [batch, width, height]
    """
    clean_or_make_dir(output_dir)

    def array2Picture(arr, name):
        img = Image.fromarray(arr * 255)
        img = img.convert('L')
        img.save(os.path.join(output_dir, "img-%d.jpg" % name))

    for i, mtx in enumerate(array):
        array2Picture(numpy.array(mtx), i)
