#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:38:41 2020

@author: gerard
"""

import scipy.ndimage as ndim
from skimage.feature import register_translation

def cross_correlation_fourier(images):
    """
    params:
        images: list of frames that form an observation
    return
        images aligned using cross correlation in fourier for faster computation
    """
    print("Aligning images...")
    # The register_translation function uses cross-correlation in Fourier space
    referenceImage = images[0]
    newImages = [referenceImage]
    for i in range(1,len(images)):
        shift, error, diffphase = register_translation(referenceImage, images[i], 100)
        tmp = ndim.shift(images[i], shift)
        newImages.append(tmp)
    print("Done!")
    return newImages