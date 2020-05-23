#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:41:26 2020

@author: gerard
"""
import skyCleaner, skyAlignment, skyTracker, skyBrian, utils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import png
import random
import copy


obs = 'observation/'
df = 'df/'
ff = 'ff/'
fdf = 'fdf/'

images, df, ff, fdf = skyCleaner.loadImages(obs, df, ff, fdf)

cleanImages = skyCleaner.cleanImages(images, df, ff, fdf)
    
alignedImages = skyAlignment.cross_correlation_fourier(cleanImages)
starCoords , viewCoords = skyTracker.getStarCoords(images[0])
print("Getting curves...")
lightCurve = skyBrian.getBrians(images, starCoords)
lightCurves = skyBrian.getBetterBrians(lightCurve)
print("Done!")

