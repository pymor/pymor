# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import matplotlib.pyplot as plt

import pymor.discretizations.iosys as iosys


def H(s):
    return np.exp(-s)

def dH(s):
    return -np.exp(-s)

tf = iosys.TF(1, 1, H, dH)

w = np.logspace(-1, 1, 100)
tfw = tf.bode(w)
plt.loglog(w, np.abs(tfw[0, 0, :]), '.-')
plt.show()
