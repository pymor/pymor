# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.functions.basic import FunctionBase


class BitmapFunction(FunctionBase):
    """Define a 2D |Function| via a grayscale image.

    Parameters
    ----------
    filename
        Path of the image representing the function.
    bounding_box
        Lower left and upper right coordinates of the domain of the function.
    range
        A pixel of value p is mapped to `(p / 255.) * range[1] + range[0]`.
    """

    dim_domain = 2
    shape_range = ()

    def __init__(self, filename, bounding_box=[[0., 0.], [1., 1.]], range=[0., 1.]):
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL is needed for loading images. Try 'pip install pillow'")
        img = Image.open(filename)
        assert img.mode == "L", "Image " + filename + " not in grayscale mode"
        rawdata = np.array(img.getdata())
        assert rawdata.shape[0] == img.size[0]*img.size[1]
        self.bitmap = rawdata.reshape(img.size[0], img.size[1]).T[:, ::-1]
        self.bounding_box = bounding_box
        self.lower_left = np.array(bounding_box[0])
        self.size = np.array(bounding_box[1] - self.lower_left)
        self.range = range

    def evaluate(self, x, mu=None):
        indices = np.maximum(np.floor((x - self.lower_left) * np.array(self.bitmap.shape) / self.size).astype(int), 0)
        F = (self.bitmap[np.minimum(indices[..., 0], self.bitmap.shape[0] - 1),
                         np.minimum(indices[..., 1], self.bitmap.shape[1] - 1)]
             * ((self.range[1] - self.range[0]) / 255.)
             + self.range[0])
        return F
