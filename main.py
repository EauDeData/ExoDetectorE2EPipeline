#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:41:26 2020

@author: gerard
"""
import skyCleaner, skyAlignment, skyTracker, skyBrian
import matplotlib.pyplot as plt
import numpy as np

obs = df = ff = fdf = './cleanObservations/'

images, df, ff, fdf = skyCleaner.loadImages(obs, df, ff, fdf)
#cleanImages = skyCleaner.cleanImages(images, df, ff, fdf)
cleanImages = images
alignedImages = skyAlignment.cross_correlation_fourier(cleanImages)
starCoords , viewCoords = skyTracker.getStarCoords(images[0])
print("Getting curves...")
lightCurve = skyBrian.getBrians(images, starCoords,n=10) #n parell
lightCurves = np.array(skyBrian.getBetterBrians(lightCurve))
print("Done!")
for i in range(len(lightCurves)):
    plt.plot(lightCurves[i])
plt.show()

