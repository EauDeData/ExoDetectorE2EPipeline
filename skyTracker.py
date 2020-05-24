#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 17:08:43 2020

@author: gerard
"""


import cv2
import numpy as np
from skimage.filters import threshold_triangle

def getStarCoords(image):
    """
    params:
        image: first image of the observation
    return
        coordinates of the stars detected
    """
    print("Getting stars...")
    t = threshold_triangle(image)  # Selected with skimage.filters.try_all_thresholds
    print("Threshold:", t)
    thresh = cv2.threshold(image, t, image.max(), cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]     
    star_coords = []
    star_coords_x = []
    star_coords_y = []
    for star in contours:
        star = star.reshape((-1,2))
        mean_x = np.round(np.mean(star[:,0]))
        mean_y = np.round(np.mean(star[:,1]))
        star_coords_x.append(mean_x)
        star_coords_y.append(mean_y)
        star_coords.append([mean_x, mean_y])
    print("Done!")
    return star_coords, [star_coords_x, star_coords_y]