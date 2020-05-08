#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:15:31 2020

@author: gerard
"""

import skyTracker, skyCleaner
import numpy as np
import matplotlib.pyplot as plt

images = skyTracker.alignedImages
starCoords = skyTracker.starCoords

def getBrians(images, starCoords):
    ohHiBrian = []
    indx = 0
    for star in starCoords:
        n = 1
        star_value = []
        for im in images:
            mask = im[int(star[1])-n:int(star[1])+n, int(star[0])-n:int(star[0])+n]
            star_value.append(np.mean(mask))
            
        res = []
        for i in range(1,len(star_value)-1):
            mean = np.mean(star_value[i-1:i+1])
            res.append(mean)
        
        skyTracker.skyCleaner.time_serie_plot(res, indx)
        indx += 1
        ohHiBrian.append(res)
        
    return ohHiBrian

myLovelyBrians = getBrians(images, starCoords)
