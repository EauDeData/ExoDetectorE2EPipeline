#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:38:41 2020

@author: gerard
"""

import skyCleaner
import scipy.ndimage as ndim
import numpy as np
from skimage.feature import register_translation
import cv2
import os

#imagesOriginals = skyCleaner.images
#images = skyCleaner.cleanImages(skyCleaner.images, skyCleaner.df, skyCleaner.ff, skyCleaner.fdf)
#imagesOriginals = skyCleaner.images

def cross_correlation_fourier(images):
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

def create_video(video, filename):
    print("Creating video...")
    x, y = video[0].shape
    local_path = os.path.dirname(os.path.abspath('__file__'))
    out = cv2.VideoWriter(local_path + "/{}.avi".format(filename), cv2.VideoWriter_fourcc(*'DIVX'), 20, (y, x), False)        
    for i in range(len(video)):
        tmp = video[i]
        tmp = skyCleaner.normalize(tmp)
        out.write(tmp.astype(np.uint8))
    out.release()
    return np.array(video)


