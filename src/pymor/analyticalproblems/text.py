# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)
import warnings

from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, LincombFunction, BitmapFunction
from pymor.core.defaults import defaults
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.tools.io import safe_temporary_filename


@defaults('font_name')
def text_problem(text='pyMOR', font_name=None):
    import numpy as np
    from PIL import Image, ImageDraw

    def _getsize(font_object, string):
        try:
            # pillow >= 9.2.0
            left, top, right, bottom = font_object.getbbox(string)
            return (right-left, bottom-top)
        except AttributeError:
            return font_object.getsize(string)

    font = _get_font(font_name)
    size = _getsize(font, text)  # compute width and height of rendered text
    size = (size[0] + 20, size[1] + 20)  # add a border of 10 pixels around the text

    def make_bitmap_function(char_num):
        # we need to generate a BitmapFunction for each character
        img = Image.new('L', size)  # create new Image object of given dimensions
        d = ImageDraw.Draw(img)  # create ImageDraw object for the given Image

        # in order to position the character correctly, we first draw all characters from the first
        # up to the wanted character
        d.text((10, 10), text[:char_num + 1], font=font, fill=255)

        # next we erase all previous character by drawing a black rectangle
        if char_num > 0:
            d.rectangle(((0, 0), (_getsize(font, text[:char_num])[0] + 10, size[1])), fill=0, outline=0)

        # open a new temporary file
        # after leaving this 'with' block, the temporary file is automatically deleted
        with safe_temporary_filename(name='letter_bitmap.png') as f:
            img.save(f, format='png')
            return BitmapFunction(f, bounding_box=[(0, 0), size], range=[0., 1.])

    # create BitmapFunctions for each character
    dfs = [make_bitmap_function(n) for n in range(len(text))]

    # create an indicator function for the background
    background = ConstantFunction(1., 2) - LincombFunction(dfs, np.ones(len(dfs)))

    # form the linear combination
    dfs = [background] + dfs
    coefficients = [1] + [ProjectionParameterFunctional('diffusion', len(text), i) for i in range(len(text))]
    diffusion = LincombFunction(dfs, coefficients)

    return StationaryProblem(
        domain=RectDomain(dfs[1].bounding_box, bottom='neumann'),
        neumann_data=ConstantFunction(-1., 2),
        diffusion=diffusion,
        parameter_ranges=(0.1, 1.)
    )


def _get_font(font_name):
    from PIL import ImageFont
    font_list = [font_name] if font_name else ['DejaVuSansMono.ttf', 'VeraMono.ttf', 'UbuntuMono-R.ttf', 'Arial.ttf']
    for filename in font_list:
        try:
            return ImageFont.truetype(filename, 64)  # load some font from file of given size
        except (OSError, IOError):
            pass
    warnings.warn(f'Falling back to pillow default font. Could not load any of {font_list}',
                  ResourceWarning, stacklevel=1)
    return ImageFont.load_default()
